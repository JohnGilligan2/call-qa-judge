"""The regression gate for a rubric change. Ties lose.

What is under test is the RUBRIC: twelve guidance strings and their weights,
each one a claim about quality that can be wrong. A candidate rubric is
replayed through the judge on frozen golden cases, several times each, and is
promoted only if it beats the incumbent on BOTH mean agreement and pass^k.

Golden cases carry the human's verdict as a BAND per criterion (absent, poor,
adequate, good, excellent, or ungradable), not a point score. Two careful
people will not agree a call is a 6 and not a 7, and a gate built on that
disagreement fails on taste. They will agree it is "poor" and not "good".

A replay's score is the share of criteria where the judge landed inside the
human's band, on 0-10. Declining a criterion the human called ungradable
counts as agreement. A criterion the candidate rubric dropped counts as a
miss, so deleting the hard criteria cannot win the gate.

Scoring against the human bands, and not against the judge's own mean, is
what stops the obvious exploit: a rubric reworded to be generous raises
every score and would sail through a gate that rewards higher scores.

Two more rules:

  * A structurally broken candidate (see `rubric.validate_rubric`) is
    rejected before anything is replayed, and the problems are printed.
  * `break_rubric` builds a realistically broken candidate, the kind a hasty
    edit produces. The gate's own self-test requires that it both REJECTS
    that candidate and scores it materially worse. A gate that rejects
    everything would pass the weaker test while protecting nothing.

A winner is a recommendation. Publish it switched off and let a person
flip it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

from .calibrate import UNGRADABLE, HumanPass
from .judge import Judge
from .rubric import Rubric, validate_rubric
from .transcript import Call

BANDS = {
    "absent": (0, 1),
    "poor": (2, 4),
    "adequate": (5, 6),
    "good": (7, 8),
    "excellent": (9, 10),
}
PASS_THRESHOLD = 8.0
MATERIALLY_WORSE = 1.0


def band_for(score: int) -> str:
    for name, (lo, hi) in BANDS.items():
        if lo <= score <= hi:
            return name
    raise ValueError(f"score {score} is off the 0-10 scale")


def in_band(score: int | None, band: str) -> bool:
    if band == UNGRADABLE:
        return score is None
    if score is None:
        return False
    lo, hi = BANDS[band]
    return lo <= score <= hi


@dataclass(frozen=True)
class GoldenCase:
    call: Call
    bands: dict[str, str]


def golden_from_human(calls: Sequence[Call], human: HumanPass) -> list[GoldenCase]:
    by_id = {c.call_id: c for c in calls}
    cases = []
    for call_id, scores in human.scores.items():
        if call_id not in by_id:
            continue
        bands = {k: (UNGRADABLE if v == UNGRADABLE else band_for(int(v)))
                 for k, v in scores.items()}
        cases.append(GoldenCase(by_id[call_id], bands))
    return cases


@dataclass
class CaseResult:
    case_id: str
    scores: list[float] = field(default_factory=list)
    passes: int = 0
    replays: int = 0


@dataclass
class RunResult:
    name: str
    rubric_version: str
    problems: list[str] = field(default_factory=list)
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        s = [x for c in self.cases for x in c.scores]
        return sum(s) / len(s) if s else 0.0

    @property
    def per_trial(self) -> float:
        total = sum(c.replays for c in self.cases)
        return sum(c.passes for c in self.cases) / total if total else 0.0

    def pass_at_k(self, k: int) -> float:
        return 1.0 - (1.0 - self.per_trial) ** k

    def pass_pow_k(self, k: int) -> float:
        return self.per_trial ** k


def agreement_score(scores: dict[str, int | None], bands: dict[str, str]) -> float:
    if not bands:
        return 0.0
    hit = sum(1 for k, band in bands.items()
              if k in scores and in_band(scores[k], band))
    return round(10.0 * hit / len(bands), 2)


def evaluate(name: str, cases: Sequence[GoldenCase], judge: Judge, rubric: Rubric,
             *, replays: int = 3, threshold: float = PASS_THRESHOLD) -> RunResult:
    run = RunResult(name=name, rubric_version=rubric.version,
                    problems=validate_rubric(rubric))
    for case in cases:
        cr = CaseResult(case.call.call_id)
        for _ in range(replays):
            v = judge.score(case.call, rubric)
            s = agreement_score(v.scores, case.bands)
            cr.scores.append(s)
            cr.replays += 1
            if s >= threshold:
                cr.passes += 1
        run.cases.append(cr)
    return run


@dataclass
class GateDecision:
    promote: bool
    reason: str
    incumbent: RunResult
    candidate: RunResult
    k: int

    def to_text(self) -> str:
        i, c = self.incumbent, self.candidate
        lines = [
            f"gate: candidate {c.rubric_version} vs incumbent {i.rubric_version} (k={self.k})",
            f"  mean agreement  incumbent {i.mean_score:5.2f}   candidate {c.mean_score:5.2f}",
            f"  per-trial pass  incumbent {i.per_trial:5.2f}   candidate {c.per_trial:5.2f}",
            f"  pass@{self.k}          incumbent {i.pass_at_k(self.k):5.3f}   candidate {c.pass_at_k(self.k):5.3f}",
            f"  pass^{self.k}          incumbent {i.pass_pow_k(self.k):5.3f}   candidate {c.pass_pow_k(self.k):5.3f}",
            "  decision: " + ("PROMOTE (publish switched off; a person flips it on)"
                              if self.promote else "REJECT"),
            f"  reason: {self.reason}",
        ]
        return "\n".join(lines)


def gate(incumbent: RunResult, candidate: RunResult, k: int = 5) -> GateDecision:
    if candidate.problems:
        return GateDecision(False, "candidate rubric is structurally broken: "
                            + "; ".join(candidate.problems), incumbent, candidate, k)
    better_mean = candidate.mean_score > incumbent.mean_score
    better_pow = candidate.pass_pow_k(k) > incumbent.pass_pow_k(k)
    if better_mean and better_pow:
        return GateDecision(True, "beats the incumbent on mean agreement and on pass^k",
                            incumbent, candidate, k)
    if better_mean:
        return GateDecision(False, "higher mean but pass^k did not improve",
                            incumbent, candidate, k)
    if better_pow:
        return GateDecision(False, "more consistent but lower mean",
                            incumbent, candidate, k)
    return GateDecision(False, "no improvement on either axis (a tie loses)",
                        incumbent, candidate, k)


def break_rubric(rubric: Rubric) -> Rubric:
    """A broken candidate of the kind a hasty edit produces.

    The two heaviest criteria get their guidance emptied, as if someone
    cleared them meaning to rewrite them later. Still valid JSON, still
    loads, plausible at a glance.
    """
    heaviest = sorted(rubric.criteria, key=lambda c: -c.weight)[:2]
    keys = {c.key for c in heaviest}
    crits = tuple(replace(c, guidance="") if c.key in keys else c
                  for c in rubric.criteria)
    return Rubric(version=rubric.version + "-broken", criteria=crits)


def self_test(cases: Sequence[GoldenCase], judge: Judge, rubric: Rubric,
              *, replays: int = 3, k: int = 5) -> tuple[bool, GateDecision]:
    """Does the gate actually bite? Rejects the broken rubric AND scores it
    at least MATERIALLY_WORSE points lower, or the self-test fails."""
    inc = evaluate("incumbent", cases, judge, rubric, replays=replays)
    broken = break_rubric(rubric)
    cand = evaluate("broken", cases, judge, broken, replays=replays)
    # Score the broken rubric's replays even though gate() rejects it on
    # structure first: the self-test wants BOTH signals.
    decision = gate(inc, cand, k)
    worse = cand.mean_score <= inc.mean_score - MATERIALLY_WORSE
    return (not decision.promote) and worse, decision
