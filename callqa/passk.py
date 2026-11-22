"""pass^k: the number a supervisor actually experiences.

pass@k asks "could this ever work": did it succeed in AT LEAST ONE of k
tries. It rises toward 1.0 the more retries you allow. pass^k asks "does
this work every time": did it succeed in ALL k. It decays as p^k. For
anything a person sees end to end, the second is the honest one.

At a 90 percent per-call rate:

    k=1   90.0%   one scored call
    k=5   59.0%   one agent's shift
    k=10  34.9%   a supervisor's weekly review

The pass threshold is a domain definition and this module will not invent
one. Where "a good call" sits on 0-10 is a question for whoever runs the
floor. `passk_table` sweeps a range of thresholds so the bar can be picked by
looking at what it implies.

`passk_table_with_checks` counts a call as a success only if it ALSO passes
every deterministic check. A call that scored 8 on a transcript with nobody
on the line is not a success, and an UNKNOWN check is not a pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

DEFAULT_KS = (1, 5, 10)


@dataclass(frozen=True)
class PassKRow:
    threshold: float
    n: int
    passes: int

    @property
    def per_trial(self) -> float:
        return self.passes / self.n if self.n else 0.0

    def pass_at_k(self, k: int) -> float:
        return 1.0 - (1.0 - self.per_trial) ** k

    def pass_pow_k(self, k: int) -> float:
        return self.per_trial ** k


def passk_table(scores: Sequence[float], thresholds: Sequence[float]) -> list[PassKRow]:
    return [PassKRow(float(t), len(scores), sum(1 for s in scores if s >= t))
            for t in thresholds]


def passk_table_with_checks(items: Sequence, thresholds: Sequence[float],
                            score_of: Callable[[object], float | None],
                            check_ok: Callable[[object], bool]) -> list[PassKRow]:
    rows = []
    for t in thresholds:
        passes = 0
        for it in items:
            s = score_of(it)
            if s is not None and s >= t and check_ok(it):
                passes += 1
        rows.append(PassKRow(float(t), len(items), passes))
    return rows


def render_markdown(rows: Sequence[PassKRow], ks: Sequence[int] = DEFAULT_KS,
                    *, title: str = "pass^k", note: str = "") -> str:
    L = [f"### {title}", ""]
    if note:
        L += [note, ""]
    L.append("| threshold | n | per-call | "
             + " | ".join([f"pass@{k}" for k in ks] + [f"pass^{k}" for k in ks]) + " |")
    L.append("|" + "---|" * (3 + 2 * len(ks)))
    for r in rows:
        cells = [f"overall >= {r.threshold:g}", str(r.n), f"{r.per_trial:.1%}"]
        cells += [f"{r.pass_at_k(k):.1%}" for k in ks]
        cells += [f"**{r.pass_pow_k(k):.1%}**" for k in ks]
        L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def explain_gap(per_trial: float, k: int = 10) -> str:
    """One sentence a non-technical reader can act on."""
    return (f"At {per_trial:.0%} per call, a supervisor reviewing {k} scored "
            f"calls has a {1 - per_trial ** k:.0%} chance of seeing at least "
            f"one they would overturn.")
