"""General correctness and performance fixtures for the remote testbench."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Case:
    case_id: str
    b: int
    s: int
    h: int
    iters: int
    eps: float = 1e-6
    distribution: str = "random"
    purpose: str = ""

    @property
    def shape(self):
        return [self.b, self.s, self.h * (self.h + 2)]

    def metadata(self):
        return dict(asdict(self), shape=self.shape, dtype="float32")


CASES = {c.case_id: c for c in (
    Case("P01", 1, 1, 2, 1, purpose="minimum task"),
    Case("P02", 1, 1, 128, 32, purpose="one large matrix"),
    Case("P03", 4, 4096, 4, 20, purpose="business workload"),
    Case("P04", 128, 8192, 4, 1, purpose="maximum tokens, short iteration"),
    Case("P05", 128, 8192, 4, 32, purpose="maximum tokens, long iteration"),
    Case("P06", 1, 4093, 7, 20, purpose="below alignment boundary"),
    Case("P07", 1, 4093, 8, 20, purpose="aligned control"),
    Case("P08", 1, 4093, 9, 20, purpose="above alignment boundary"),
    Case("P09", 1, 257, 127, 32, purpose="large ragged matrix"),
    Case("P10", 1, 257, 128, 32, purpose="maximum H control"),
    Case("P11", 1, 4096, 128, 1, purpose="large workset, short iteration"),
    Case("P12", 1, 4096, 128, 32, purpose="large workset, long iteration"),
    Case("C01", 1, 5, 3, 2, distribution="mixed", purpose="direction and I=2"),
    Case("C02", 1, 5, 9, 1, distribution="mixed", purpose="stable initialization"),
    Case("C03", 1, 5, 17, 20, distribution="slow", purpose="fixed iteration count"),
    Case("C04", 1, 5, 65, 32, eps=1e-3, distribution="mixed", purpose="nondefault positive eps"),
    # Append new cases so full-suite indices (and existing fixture seeds) stay stable.
    Case("S01", 1, 257, 63, 32, purpose="secondary: below 64-element reduction boundary"),
    Case("S02", 1, 257, 64, 32, purpose="secondary: 64-element reduction control"),
    Case("S03", 1, 257, 65, 32, purpose="secondary: above 64-element reduction boundary"),
)}


def fixture_seed(case_id, base_seed):
    if case_id not in CASES:
        raise ValueError("case ID must be known")
    return base_seed + list(CASES).index(case_id)


SUITES = {
    "smoke": ("P01", "P02", "C01", "C02", "C03", "C04"),
    "quick": ("P03", "P05", "P08", "P10", "P11", "P12"),
    "correctness": tuple(k for k in CASES if k.startswith("C")),
    "performance": tuple(k for k in CASES if k.startswith("P")),
    "secondary": ("S01", "S02", "S03"),
    "full": tuple(CASES),
}


def select_cases(suite="smoke", ids=None):
    names = [s.strip().upper() for s in ids.split(",")] if ids else SUITES[suite]
    if not names or any(n not in CASES for n in names) or len(set(names)) != len(names):
        raise ValueError("case IDs must be known, nonempty and unique")
    return [CASES[n] for n in names]
