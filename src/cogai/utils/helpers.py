import json
from pathlib import Path
from .rule import  Rule

def save_checkpoint(path: Path,
                    rules: list[Rule],
                    pairs: dict[tuple[str,str], float]) -> None:
    ckpt = {
        "rules": [(r.f_in, r.f_out, r.l, r.r) for r in rules],
        # clé concaténée pour pouvoir la passer en JSON
        "pairs": {f"{p}|||{d}": w for (p, d), w in pairs.items()}
    }
    path.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), "utf8")


def load_checkpoint(path: Path) -> tuple[list[Rule], dict[tuple[str,str], float]]:

    ckpt = json.loads(path.read_text("utf8"))
    rules: list[Rule] = [Rule(*t) for t in ckpt["rules"]]

    pairs: dict[tuple[str,str], float] = {
        # on décompose la clé en deux variables p et d, toutes deux str
        (p, d): float(w)
        for k, w in ckpt["pairs"].items()
        for p, d in [k.split("|||", 1)]
    }

    return rules, pairs

