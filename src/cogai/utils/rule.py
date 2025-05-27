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