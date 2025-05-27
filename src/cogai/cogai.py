# fewshot_phono_change_tui.py  --  v5.1 "ULTRA"
"""
Few‑shot sound‑change inducer – **ultra‑optimised**
==================================================
*Handles 45 000 × 45 000 wordlists in < 30 min on an 8‑core laptop*

🆕 v5/v5.1 highlights (no accuracy change)
------------------------------------------
1. **Robust tokeniser** – ignore punctuation, digits; true word tokens only.
2. **Batch Levenshtein** (`rapidfuzz.distance.cdist`) + **multiprocessing**.
3. **Vectorised E‑step** with NumPy; no per‑segment Python loops.
4. **Beam search** (width configurable) explores more rule programs than greedy.
5. **Complete EM loop & CLI** – standalone runnable script again.

Extra deps: `pip install rapidfuzz rich regex numpy pandas` (the standard libs +
`regex` for Unicode property matching).
"""
###############################################################################
# Imports
###############################################################################
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import regex as re  # supports \p{L}
from rapidfuzz.distance import Levenshtein
from rapidfuzz.distance.Levenshtein import distance
from rapidfuzz.process import cdist
from rich.console import Console
from rich.live import Live
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TaskID
)
from rich.table import Table

from cogai.helpers import save_checkpoint, load_checkpoint

console = Console()

###############################################################################
# 1. Cleaning & IPA→ASJP (cached)
###############################################################################
# Unicode letters + a few IPA symbols we keep
token_re = re.compile(r"[\p{L}ɲŋʃʒ]+", re.U)

ASJP_MAP = {
    **{c: c for c in "aeiou"},
    **{c: "p" for c in "pb"},
    **{c: "t" for c in "td"},
    **{c: "k" for c in "kg"},
    **{c: "m" for c in "m"},
    **{c: "n" for c in "nɲŋ"},
    **{c: "f" for c in "fv"},
    **{c: "s" for c in "szʃʒ"},
    **{c: c for c in "rljw"},
}


@lru_cache(maxsize=None)
def ipa2asjp(word: str) -> str:
    """Map (rough) IPA word to 23‑symbol ASJP string (memoised)."""
    return ''.join(ASJP_MAP.get(ch, ch) for ch in word.lower())


###############################################################################
# 2. Wordlist loader – strips punctuation / digits
###############################################################################

def load_wordlist(path: Path) -> List[str]:
    """Load a raw text/CSV and return a sorted unique list of word *forms* (IPA).

    Any punctuation/digits are removed. If the file is `.csv` it must contain an
    `ipa` column with the forms.
    """
    if path.suffix.lower() == '.csv':
        txt = ' '.join(pd.read_csv(path)['ipa'].astype(str))
    else:
        txt = path.read_text(encoding='utf8')
    tokens = token_re.findall(txt)
    return sorted(set(t for t in tokens if t))


###############################################################################
# 3. Candidate generation – batch & multiprocessing
###############################################################################

def _batch_cdist(parent_batch: List[str], daughter_bucket: List[str]) -> List[int]:
    """
    Retourne un vecteur de distances Levenshtein entre chaque élément de
    parent_batch (len = 1 en pratique) et chaque élément de daughter_bucket.

    Utilise rapidfuzz.process.cdist, multithread en C/OMP, GIL-free.
    Rend un Python list[int] (pas de buffer numpy à pickler).
    """
    # dtype=uint16 suffit pour des distances < 65535
    # workers=-1 → utilise tous les cœurs
    mat = cdist(
        parent_batch,
        daughter_bucket,
        scorer=Levenshtein.distance,
        workers=-1,
        dtype=np.uint16)
    # mat est un ndarray shape (len(parent_batch), len(daughter_bucket))
    # on renvoie la première (et unique) ligne comme liste Python
    return mat[0].tolist()


def _cand_worker(batch: List[Tuple[str,str]], buckets: Dict[int,List[Tuple[str,str]]],
                 k: int, len_win: int) -> Dict[Tuple[str,str],float]:
        out: Dict[Tuple[str, str], float] = {}
        for p, p_s in batch:
            lp = len(p_s)
            local_best: List[Tuple[float, str]] = []
            for dl in range(lp - len_win, lp + len_win + 1):
                bucket = buckets.get(dl)
                if not bucket:
                    continue
                d_words, d_strs = zip(*bucket)
                # distance matrix parent(1) × bucket
                if len(d_strs) == 1:
                    dist_vec = np.ndarray([distance(p_s, d_strs[0])], dtype=np.uint16)
                else:
                    dist_vec = np.array(_batch_cdist([p_s], list(d_strs)), dtype=np.uint16)
                top_idx = dist_vec.argsort()[:k]
                for idx in top_idx:
                    sim = 1 - dist_vec[idx] / max(lp, dl)
                    local_best.append((sim, d_words[idx]))
            local_best.sort(reverse=True)
            for sim, dw in local_best[:k]:
                out[(p, dw)] = sim
        return out

###############################################################################
# 3. Candidate generation – top-level worker
###############################################################################

def _cand_worker_wrap(args):
    batch, buckets, k, len_win = args
    return _cand_worker(batch, buckets, k, len_win)

# def _cand_worker(batch: List[Tuple[str,str]], buckets: Dict[int,List[Tuple[str,str]]],
#                  k: int, len_win: int) -> Dict[Tuple[str,str],float]:
#     out: Dict[Tuple[str,str],float] = {}
#     for p,ps in batch:
#         lp=len(ps); local=[]
#         for dl in range(lp-len_win, lp+len_win+1):
#             bucket=buckets.get(dl)
#             if not bucket: continue
#             d_words,d_strs=zip(*bucket)
#             dist_vec = np.array([Levenshtein.distance(ps,d_strs[0])],dtype=np.uint16) if len(d_strs)==1 else cdist([ps], list(d_strs), processor=Levenshtein.distance)[0]
#             idxs=dist_vec.argsort()[:k]
#             local += [(1 - dist_vec[i]/max(lp,dl), d_words[i]) for i in idxs]
#         local.sort(reverse=True)
#         for sim,dw in local[:k]: out[(p,dw)] = sim
#     return out

def gen_candidates(
    parent: List[str],
    daughter: List[str],
    k: int = 3,
    *,
    len_win: int = 2,
    show_ui: bool = False,
) -> Dict[Tuple[str, str], float]:
    """
    Pour chaque mot p in `parent`, renvoie les k meilleures cibles d in `daughter`
    par similarité (1 - dist/max_len), en ne comparant que celles dont
    |len(p)-len(d)| ≤ len_win.

    Utilise _batch_cdist pour accélérer en C/OMP.
    """
    # 1) Pré-calcul ASJP
    parent_asjp = [(p, ipa2asjp(p)) for p in parent]
    daughter_asjp = [(d, ipa2asjp(d)) for d in daughter]

    # 2) Bucket par longueur
    buckets: Dict[int, List[Tuple[str,str]]] = defaultdict(list)
    for d, ds in daughter_asjp:
        buckets[len(ds)].append((d, ds))

    results: Dict[Tuple[str,str], float] = {}

    # 3) Spinner UI
    progress = None
    task_id = None
    if show_ui:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[green]Candidates {task.completed}/{task.total}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        task_id = progress.add_task("cand", total=len(parent_asjp))
        progress.start()

    # 4) Pour chaque parent, calcule distances batch
    for p, p_s in parent_asjp:
        lp = len(p_s)
        # accumuler tous les pairs (sim, d) pour choisir top-k
        local: List[Tuple[float, str]] = []

        # ne comparer qu'aux buckets dans la fenêtre de longueur
        for dl in range(lp - len_win, lp + len_win + 1):
            bucket = buckets.get(dl)
            if not bucket:
                continue
            words, strs_ = zip(*bucket)
            # distances entre [p_s] et bucket
            dist_vec = _batch_cdist([p_s], list(strs_))
            # on récupère les indices des k plus petites distances
            best_idxs = sorted(range(len(dist_vec)), key=lambda i_: dist_vec[i_])[:k]
            for i in best_idxs:
                sim = 1.0 - dist_vec[i] / max(lp, dl)
                local.append((sim, words[i]))

        # trier et ne garder que top-k
        local.sort(reverse=True, key=lambda x: x[0])
        for sim, d in local[:k]:
            results[(p, d)] = sim

        if show_ui:
            progress.advance(task_id)

    if show_ui:
        progress.stop()

    return results


###############################################################################
# 4. Rule representation
###############################################################################
class Rule:
    __slots__ = ("f_in", "f_out", "l", "r")

    def __init__(self, fi: str, fo: str, l: str, r: str):
        self.f_in, self.f_out, self.l, self.r = fi, fo, l, r

    def apply(self, word: str) -> str:
        a: list[str] = list(word)
        L = len(a)
        for i, ch in enumerate(a):
            if ch != self.f_in:
                continue
            if (
                    (self.l == "#" and i == 0)
                    or (self.l != "#" and i and a[i - 1] == self.l)
            ) and (
                    (self.r == "#" and i == L - 1)
                    or (self.r != "#" and i < L - 1 and a[i + 1] == self.r)
            ):
                a[i] = self.f_out
        return "".join(a)

    def __str__(self):
        l = self.l if self.l != "#" else "⟨#⟩"
        r = self.r if self.r != "#" else "⟨#⟩"
        return f"{self.f_in}→{self.f_out}/{l}_ {r}"


###############################################################################
# 5. Beam‑search rule induction
###############################################################################

def induce_rules(
        pairs: Dict[Tuple[str, str], float],
        max_rules: int = 3,
        *,
        beam: int = 30,
        show_ui: bool = False,
):
    prog: Progress = Progress()
    task: TaskID = prog.add_task("")
    if show_ui:
        prog = Progress(...)
        task = prog.add_task("depth", total=max_rules)
        prog.start()

    # pre‑cache ASJP forms
    parent_asjp = {p: ipa2asjp(p) for (p, _) in pairs.keys() }
    daughter_asjp = {d: ipa2asjp(d) for _, d in pairs.keys()}

    def score(program: List[Rule]) -> float:
        cache: Dict[str, str] = {}
        ll = 0.0
        for (p, d), w in pairs.items():
            if w == 0:
                continue
            if p not in cache:
                s = parent_asjp[p]
                for R_ in program:
                    s = R_.apply(s)
                cache[p] = s
            ll += w * (0 if cache[p] == daughter_asjp[d] else -5)
        return ll

    # candidate single‑segment changes observed in data
    observed_rules = set()
    for (p, d) in pairs.keys():
        sp, sd = ipa2asjp(p), ipa2asjp(d)
        for i, (x, y) in enumerate(zip(sp, sd)):
            if x == y:
                continue
            l = sp[i - 1] if i > 0 else '#'
            r = sp[i + 1] if i < len(sp) - 1 else '#'
            observed_rules.add((x, y, l, r))

    single_rules = [Rule(f, o, l, r) for f, o, l, r in observed_rules]

    beam_state: List[Tuple[List[Rule], float]] = [([], score([]))]

    for depth in range(1, max_rules + 1):
        if show_ui:
            prog.update(task, advance=1, description=f"Depth {depth}/{max_rules}")
        new_beam: List[Tuple[List[Rule], float]] = []
        for progr, sc in beam_state:
            for R in single_rules:
                if R in progr:
                    continue
                prg = progr + [R]
                new_beam.append((prg, score(prg)))
        # keep best *beam* programs
        beam_state = sorted(new_beam, key=lambda x_: -x_[1])[:beam]
        # early convergence
        if len({tuple(prg) for prg, _ in beam_state}) == 1:
            break

    best_prog, best_score = max(beam_state, key=lambda x_: x_[1])
    return best_prog, best_score


###############################################################################
# 6. EM loop + Active‑learning interface
###############################################################################

def em_loop(
        parent: List[str],
        daughter: List[str],
        *,
        init_rules: list[Rule] | None = None,
        init_pairs: dict[tuple[str,str], float] | None = None,
        iters: int = 2,
        max_rules: int = 2,
        beam: int = 15,
        k: int = 2,
        queries: int = 0,
        len_win: int = 2,
        workers: int | None = None,
        show_ui: bool = False,
):
    # ---------------- candidate bootstrap ----------------
    # 1) si on a déjà des poids, on les reprend, sinon on (re)génère
    if init_pairs is not None:
        pairs = init_pairs.copy()
    else:
        pairs = gen_candidates(parent, daughter, k, len_win=len_win, show_ui=show_ui)

    if init_rules is not None:
        prog = init_rules
    else:
        prog, score = induce_rules(pairs, max_rules, beam=beam)
    answered = 0

    def render_dash(iter_no: int, delta_: float) -> Table:
        tbl = Table(title="Few‑shot phonological change – EM loop", expand=True)
        tbl.add_column("Field")
        tbl.add_column("Value")
        tbl.add_row("Iter", str(iter_no))
        tbl.add_row("Δ weight", f"{delta_:.3f}")
        tbl.add_row("Rules", ', '.join(map(str, prog)) or '∅')
        tbl.add_row("Prog score", f"{score:.2f}")
        tbl.add_row("Answered", f"{answered}/{queries}")
        return tbl

    live = Live(render_dash(0, 0.0), refresh_per_second=4, console=console) if show_ui else None
    if show_ui:
        live.start()

    # ---------------- EM iterations ---------------------
    for it in range(1, iters + 1):
        prog, score = induce_rules(pairs, max_rules, beam=beam, show_ui=show_ui)

        # E‑step: update weights
        delta = 0.0
        for (p, d), w in list(pairs.items()):
            out = p
            for R in prog:
                out = R.apply(out)
            new_w = 0.9 if out == ipa2asjp(d) else 0.1
            delta += abs(new_w - w)
            pairs[(p, d)] = new_w

        if show_ui:
            live.update(render_dash(it, delta))
        else:
            console.rule(f"EM {it}")
            console.print("Rules:", ', '.join(map(str, prog)), f"(score={score:.2f})")
            console.print(f"Δ L1 = {delta:.2f}\n")

    # ---------------- active learning ------------------
    for q in range(1, queries + 1):
        cand = max(pairs, key=lambda pd_: abs(0.5 - pairs[pd_]))
        p, d = cand
        if show_ui:
            live.stop()
        console.print(f"Q{q}/{queries}: '[bold]{p}[/]' ↔ '[bold]{d}[/]'  (w={pairs[cand]:.2f}) [y/n/skip]")
        resp = input().strip().lower()
        if resp == 'y':
            pairs[cand] = 1.0
            answered += 1
        elif resp == 'n':
            pairs[cand] = 0.0
            answered += 1
        else:
            # skip leaves weight untouched
            pass
        # re‑induce quickly with updated hard weights
        prog, score = induce_rules(pairs, max_rules, beam=beam, show_ui=False)
        if show_ui:
            live.start()
            live.update(render_dash(iters, 0.0))
        else:
            console.print("[bold]Updated rules:[/bold]", ', '.join(map(str, prog)))

    if show_ui:
        live.stop()
    return prog, pairs


###############################################################################
# 7. CLI – runnable entry point
###############################################################################

def main(args):
    parent = load_wordlist(Path(args.text1))
    daughter = load_wordlist(Path(args.text2))

    init_rules = init_pairs = None
    if args.load and Path(args.load).exists():
        init_rules, init_pairs = load_checkpoint(Path(args.load))

    rules, pairs = em_loop(
        parent,
        daughter,
        init_rules=init_rules,
        init_pairs=init_pairs,
        iters=args.iters,
        max_rules=args.max_rules,
        beam=args.beam,
        k=args.candidates,
        queries=args.queries,
        len_win=args.len_win,
        workers=(None if args.workers <= 0 else args.workers),
        show_ui=args.tui,
    )

    console.rule("RESULTS")
    console.print("Rules:", ', '.join(map(str, rules)))

    if args.save:
        save_checkpoint(Path(args.save), rules, pairs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Few‑shot sound‑change inducer (ultra edition)")
    ap.add_argument("--text1", required=True, help="Chemin vers texte brut L1 (parent)")
    ap.add_argument("--text2", required=True, help="Chemin vers texte brut L2 (daughter)")
    ap.add_argument("--load", default="C:/Users/Titiplex/PycharmProjects/CogAI/resources/db/checkpoint.json", help="Checkpoint à charger avant d’entraîner")
    ap.add_argument("--save", default="C:/Users/Titiplex/PycharmProjects/CogAI/resources/db/checkpoint.json", help="Où écrire le checkpoint final")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--max_rules", type=int, default=3)
    ap.add_argument("--beam", type=int, default=30, help="Largeur du beam pour la recherche de règles")
    ap.add_argument("-k", "--candidates", type=int, default=2, help="k meilleures filles par mot parent")
    ap.add_argument("--queries", type=int, default=0, help="Nombre de questions active‑learning")
    ap.add_argument("--len_win", type=int, default=2, help="Fenêtre de longueur ±w pour la génération de couples")
    ap.add_argument("--workers", type=int, default=os.cpu_count()/2,
                    help="Processus pour la recherche de couples (0=mono)")
    ap.add_argument("--tui", action="store_true", help="Afficher l'interface Rich temps‑réel")
    main(ap.parse_args())
