"""Chance-corrected agreement, because raw agreement flatters a judge.

"The judge agreed with the human on N of M verdicts" has a known defect: two
graders who both score most calls "about a 4" agree often by accident, and
the raw rate cannot tell accident from skill. So the calibration report
prints Cohen's kappa beside it.

Quadratic weights, the standard for ordinal scales. Unweighted kappa treats
a 6-vs-7 disagreement like a 0-vs-10 one, which throws away what a 0-10
scale is for.

    kappa_w = 1 - sum(w_ij * O_ij) / sum(w_ij * E_ij)

    O_ij  observed count of (judge=i, human=j)
    E_ij  expected count under independence, row_i * col_j / N
    w_ij  (i - j)^2 / (K - 1)^2

What it does NOT tell you, measured rather than assumed (and pinned in
tests/test_kappa.py): on human scores spanning 3 to 9, a judge that is a
constant one point harsh scores 0.86, two points 0.60, three points 0.40. A
constant offset is the easiest defect there is to fix (one guidance edit)
and kappa still marks it down hard, because kappa measures distance, not
consistency. So kappa is the cross-check, not the primary diagnostic. The
direction-of-miss analysis in `calibrate.py` is: a one-sided gap is the
fixable signature whatever the kappa says.

Undefined cases (fewer than two pairs, or no variance on either side to
correct for) return None, and the report prints "n/a". A zero is a claim.
"""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from .rubric import SCORE_MAX, SCORE_MIN

CATEGORIES = tuple(range(SCORE_MIN, SCORE_MAX + 1))


def quadratic_weighted_kappa(judge: Sequence[int], human: Sequence[int], *,
                             categories: Sequence[int] = CATEGORIES
                             ) -> float | None:
    if len(judge) != len(human):
        raise ValueError("judge and human must be the same length")
    n = len(judge)
    if n < 2:
        return None
    lo, hi = min(categories), max(categories)
    span = float(hi - lo)
    if span <= 0:
        return None

    def clamp(v: int) -> int:
        # Clamped, not dropped: a clamp is a visible rounding, a drop would
        # silently shrink N.
        return max(lo, min(hi, int(v)))

    pairs = Counter((clamp(j), clamp(h)) for j, h in zip(judge, human))
    row: Counter = Counter()
    col: Counter = Counter()
    for (j, h), c in pairs.items():
        row[j] += c
        col[h] += c

    def w(i: int, j: int) -> float:
        return ((i - j) ** 2) / (span ** 2)

    observed = sum(w(j, h) * c for (j, h), c in pairs.items())
    expected = sum(w(j, h) * row[j] * col[h] / n for j in row for h in col)
    if expected <= 0:
        return None
    kappa = 1.0 - observed / expected
    # A judge that ignores the input lands on exactly zero in exact
    # arithmetic and on 2e-16 in floating point. Snap it.
    return 0.0 if abs(kappa) < 1e-12 else kappa


def landis_koch(kappa: float | None) -> str:
    """The conventional band. A label, not a verdict."""
    if kappa is None:
        return "n/a"
    if kappa < 0:
        return "worse than chance"
    if kappa <= 0.20:
        return "slight"
    if kappa <= 0.40:
        return "fair"
    if kappa <= 0.60:
        return "moderate"
    if kappa <= 0.80:
        return "substantial"
    return "almost perfect"


def format_kappa(kappa: float | None) -> str:
    return "n/a" if kappa is None else f"{kappa:.2f} ({landis_koch(kappa)})"


def paired_scores(judge_by_case: Mapping, human_by_case: Mapping
                  ) -> tuple[list[int], list[int]]:
    """Align two {case_id: score} maps. Only numeric-on-both-sides survive.

    A decline on either side ("ungradable", None) is not a score and belongs
    in the overreach table, never in a kappa.
    """
    js: list[int] = []
    hs: list[int] = []
    for cid, j in judge_by_case.items():
        h = human_by_case.get(cid)
        if isinstance(j, bool) or isinstance(h, bool):
            continue
        if isinstance(j, (int, float)) and isinstance(h, (int, float)):
            js.append(int(round(j)))
            hs.append(int(round(h)))
    return js, hs
