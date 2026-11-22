"""Compare the judge with a blind human pass and say which guidance to edit.

Three numbers, because one is not enough to act on.

1.  CALL-LEVEL agreement on the weighted overall. The headline, and nearly
    useless for fixing anything, because a gap in a weighted sum cannot say
    which criterion caused it.

2.  PER-CRITERION agreement, one case per (call, criterion). Ten calls become
    about 120 comparison points. This is the diagnostic: it ranks criteria by
    how far the judge sits from the human and in which direction, and a
    criterion is a guidance string someone can go and edit.

3.  JUDGE OVERREACH: criteria the human marked ungradable that the judge
    scored anyway. Not a score disagreement, and averaging it as one would be
    wrong in both directions. It is the judge answering a question it should
    have declined.

The direction of the misses is the finding, not the rate. Random misses
scatter both ways. Three or more misses on one criterion all leaning the same
way is a rubric defect, and the report says so and quotes the guidance.

Tolerances are fixed before the human pass and printed beside every number:
2 points per criterion on the 0-10 scale, 1.5 on the overall (tighter,
because independent noise partly cancels in a weighted mean). A tolerance
chosen after seeing the results is not a tolerance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .judge import CallVerdict
from .kappa import format_kappa, paired_scores, quadratic_weighted_kappa
from .rubric import Rubric, weighted_overall

UNGRADABLE = "ungradable"
CRITERION_TOLERANCE = 2
OVERALL_TOLERANCE = 1.5
SYSTEMATIC_MIN_MISSES = 3


class CohortMismatch(ValueError):
    """Judge and human graded against different rubric versions."""


@dataclass
class HumanPass:
    rubric_version: str
    blind: bool
    scores: dict[str, dict[str, Any]]      # call_id -> {key: int | "ungradable"}
    notes: dict[str, str] = field(default_factory=dict)


def load_human(path: str | Path) -> HumanPass:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return human_from_dict(doc)


def human_from_dict(doc: Mapping[str, Any]) -> HumanPass:
    scores, notes = {}, {}
    for v in doc.get("verdicts", []):
        scores[str(v["call_id"])] = dict(v.get("scores", {}))
        notes[str(v["call_id"])] = str(v.get("note", ""))
    return HumanPass(rubric_version=str(doc.get("rubric_version", "")),
                     blind=bool(doc.get("blind", True)), scores=scores, notes=notes)


@dataclass
class Rollup:
    key: str
    n: int
    agreements: int
    harsh: int
    lenient: int
    judge_mean: float
    human_mean: float
    overreach: int
    kappa: float | None

    @property
    def delta(self) -> float:
        """Judge minus human. Negative means the judge is harsher."""
        return self.judge_mean - self.human_mean

    @property
    def one_sided(self) -> bool:
        misses = self.harsh + self.lenient
        return misses >= SYSTEMATIC_MIN_MISSES and (
            self.harsh == misses or self.lenient == misses)

    @property
    def direction(self) -> str:
        if self.harsh > self.lenient:
            return "judge harsh"
        if self.lenient > self.harsh:
            return "judge lenient"
        return "mixed"


@dataclass
class CallRow:
    call_id: str
    judge: float
    human: float
    agree: bool


@dataclass
class Calibration:
    rubric_version: str
    blind: bool
    calls: list[CallRow]
    rollups: list[Rollup]
    overreach: list[tuple[str, str, int]]      # (call_id, key, judge score)
    judge_declined: list[tuple[str, str, int]]  # (call_id, key, human score)
    pooled_kappa: float | None
    pooled_n: int
    crit_tol: float = CRITERION_TOLERANCE
    overall_tol: float = OVERALL_TOLERANCE

    @property
    def call_agreements(self) -> int:
        return sum(1 for r in self.calls if r.agree)

    @property
    def crit_n(self) -> int:
        return sum(r.n for r in self.rollups)

    @property
    def crit_agreements(self) -> int:
        return sum(r.agreements for r in self.rollups)

    @property
    def suspect(self) -> list[Rollup]:
        return [r for r in self.rollups if r.one_sided]


def calibrate(verdicts: Sequence[CallVerdict], human: HumanPass, rubric: Rubric,
              *, crit_tol: float = CRITERION_TOLERANCE,
              overall_tol: float = OVERALL_TOLERANCE,
              reuse_human_pass: bool = False) -> Calibration:
    """Judge vs human on the same calls.

    By default the judge's rubric version must equal the one the human
    graded against. After a guidance-only edit you re-score the same calls
    and re-check against the same blind human pass; `reuse_human_pass=True`
    allows that, but only when the criteria keys are unchanged, and the
    report says which two versions were compared.
    """
    versions = {v.rubric_version for v in verdicts}
    if len(versions) != 1:
        raise CohortMismatch(f"judge verdicts span rubric versions {sorted(versions)}")
    if versions != {human.rubric_version}:
        human_keys = {k for s in human.scores.values() for k in s}
        if not reuse_human_pass:
            raise CohortMismatch(
                f"judge verdicts are on rubric {sorted(versions)}, the human pass "
                f"is on {human.rubric_version!r}. A disagreement across rubric "
                f"versions cannot tell a judge defect from a rubric change.")
        if not human_keys <= set(rubric.keys):
            raise CohortMismatch(
                "the human pass graded criteria this rubric no longer has: "
                + ", ".join(sorted(human_keys - set(rubric.keys))))
    weights = rubric.weights
    by_id = {v.call_id: v for v in verdicts}

    slots: dict[str, dict[str, Any]] = {
        k: {"j": [], "h": [], "agree": 0, "harsh": 0, "lenient": 0, "over": 0}
        for k in rubric.keys}
    overreach: list[tuple[str, str, int]] = []
    declined: list[tuple[str, str, int]] = []
    calls: list[CallRow] = []
    pooled_j: dict[str, Any] = {}
    pooled_h: dict[str, Any] = {}

    for call_id, hscores in human.scores.items():
        v = by_id.get(call_id)
        if v is None:
            continue
        for key in rubric.keys:
            j = v.scores.get(key)
            h = hscores.get(key)
            cid = f"{call_id}:{key}"
            if h == UNGRADABLE:
                if j is not None:
                    overreach.append((call_id, key, j))
                    slots[key]["over"] += 1
                continue
            if h is None:
                continue
            if j is None:
                declined.append((call_id, key, h))
                continue
            s = slots[key]
            s["j"].append(j)
            s["h"].append(h)
            pooled_j[cid], pooled_h[cid] = j, h
            if abs(j - h) <= crit_tol:
                s["agree"] += 1
            elif j < h:
                s["harsh"] += 1
            else:
                s["lenient"] += 1

        # Call level: the human's overall is rebuilt with the SAME weights,
        # over only what both sides actually scored, so the two differ only
        # in the scores and never in the weighting.
        shared = {k for k in rubric.keys
                  if isinstance(hscores.get(k), (int, float))
                  and v.scores.get(k) is not None}
        if shared:
            w = {k: weights[k] for k in shared}
            jo = weighted_overall({k: v.scores[k] for k in shared}, w)
            ho = weighted_overall({k: hscores[k] for k in shared}, w)
            if jo is not None and ho is not None:
                calls.append(CallRow(call_id, jo, ho, abs(jo - ho) <= overall_tol))

    rollups = [
        Rollup(key=k, n=len(s["j"]), agreements=s["agree"], harsh=s["harsh"],
               lenient=s["lenient"],
               judge_mean=mean(s["j"]) if s["j"] else 0.0,
               human_mean=mean(s["h"]) if s["h"] else 0.0,
               overreach=s["over"],
               kappa=quadratic_weighted_kappa(s["j"], s["h"]))
        for k, s in slots.items()]
    rollups.sort(key=lambda r: (r.delta, r.key))
    js, hs = paired_scores(pooled_j, pooled_h)
    judge_version = next(iter(versions))
    label = (judge_version if judge_version == human.rubric_version
             else f"{judge_version} (human pass graded on {human.rubric_version})")
    return Calibration(rubric_version=label, blind=human.blind,
                       calls=calls, rollups=rollups, overreach=overreach,
                       judge_declined=declined,
                       pooled_kappa=quadratic_weighted_kappa(js, hs),
                       pooled_n=len(js), crit_tol=crit_tol, overall_tol=overall_tol)


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({a / b:.0%})" if b else "0/0"


def render_markdown(cal: Calibration, rubric: Rubric) -> str:
    L: list[str] = []
    a = L.append
    a(f"# Judge calibration, rubric {cal.rubric_version}")
    a("")
    if cal.blind:
        a("Human pass: blind (the grader never saw the judge's scores).")
    else:
        a("Human pass: MIXED PROVENANCE. Some verdicts were recorded with the "
          "judge's scores on screen, so agreement below is an upper bound, not "
          "an independent measurement.")
    a(f"Agreement means within {cal.overall_tol} on the overall and within "
      f"{cal.crit_tol} per criterion, both fixed before the human pass.")
    a("")
    a(f"- Call-level agreement: {_pct(cal.call_agreements, len(cal.calls))}")
    a(f"- Per-criterion agreement: {_pct(cal.crit_agreements, cal.crit_n)}")
    a(f"- Quadratic-weighted kappa, pooled over {cal.pooled_n} pairs: "
      f"{format_kappa(cal.pooled_kappa)}")
    a(f"- Judge overreach (human said ungradable, judge scored anyway): "
      f"{len(cal.overreach)}")
    a(f"- Judge declined where the human scored: {len(cal.judge_declined)}")
    a("")
    if cal.suspect:
        a("## Verdict: suspect the rubric")
        a("")
        for r in cal.suspect:
            a(f"`{r.key}`: {r.harsh + r.lenient} misses, all {r.direction}, "
              f"mean gap {r.delta:+.1f}. The guidance the judge is reading:")
            a("")
            a(f"> {rubric.guidance(r.key)}")
            a("")
    else:
        a("## Verdict: no one-sided criterion")
        a("")
        a("Misses, if any, scatter in both directions. Read them anyway.")
        a("")
    a("## Per criterion, most harsh first")
    a("")
    a("| criterion | n | agree | harsh | lenient | judge | human | gap | kappa | overreach |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for r in cal.rollups:
        flag = " **one-sided**" if r.one_sided else ""
        a(f"| `{r.key}`{flag} | {r.n} | {r.agreements} | {r.harsh} | {r.lenient} "
          f"| {r.judge_mean:.1f} | {r.human_mean:.1f} | {r.delta:+.1f} "
          f"| {format_kappa(r.kappa)} | {r.overreach} |")
    if cal.overreach:
        a("")
        a("## Overreach, by call")
        a("")
        a("| call | criteria scored that the human called ungradable | judge scores |")
        a("|---|---|---|")
        by_call: dict[str, list[tuple[str, int]]] = {}
        for call_id, key, j in cal.overreach:
            by_call.setdefault(call_id, []).append((key, j))
        for call_id, items in by_call.items():
            spread = sorted({j for _, j in items})
            a(f"| {call_id} | {len(items)} | {', '.join(str(s) for s in spread)} |")
    a("")
    a("Read the gap first and kappa second. A one-sided gap is fixable with a "
      "guidance edit whatever kappa says; kappa marks a constant offset down "
      "hard because it measures distance, not consistency.")
    return "\n".join(L) + "\n"
