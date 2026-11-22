"""The call-QA rubric, as data a supervisor can edit.

The rubric lives in a JSON file (see `rubrics/default.json`), not in code.
Every criterion is a key, a label, a relative weight and a guidance string,
and the guidance string is the part that matters: it is what the judge reads,
and in practice it is where most judge defects turn out to live.

Two rules travel with it.

1.  Weights are relative. They are renormalized over whatever criteria are
    present, so a supervisor who deletes a criterion does not silently lose
    its share off the top of every score. A perfect call still scores 10.

2.  Any edit changes what a score means, so it carries a version bump.
    Scores from two rubric versions are not comparable and nothing in this
    package averages them together.

The weighted `overall` is computed here, in code, never taken from a model.
The arithmetic is not a judgment call, and a weight change must be able to
re-score history without paying for another model run.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

SCORE_MIN = 0
SCORE_MAX = 10
MIN_GUIDANCE_CHARS = 20

DEFAULT_RUBRIC_PATH = Path(__file__).resolve().parent / "rubrics" / "default.json"

_KEY_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class Criterion:
    key: str
    label: str
    weight: float
    guidance: str


@dataclass(frozen=True)
class Rubric:
    version: str
    criteria: tuple[Criterion, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(c.key for c in self.criteria)

    def get(self, key: str) -> Criterion | None:
        for c in self.criteria:
            if c.key == key:
                return c
        return None

    def guidance(self, key: str) -> str:
        c = self.get(key)
        return c.guidance if c else ""

    @property
    def weights(self) -> dict[str, float]:
        """Fractional weights, renormalized over the criteria present."""
        raw = {c.key: max(0.0, float(c.weight)) for c in self.criteria}
        total = sum(raw.values())
        if total <= 0:
            return {k: 1.0 / len(raw) for k in raw} if raw else {}
        return {k: v / total for k, v in raw.items()}

    def with_guidance(self, key: str, guidance: str, version: str) -> "Rubric":
        """A candidate rubric with one guidance string rewritten.

        The version is required, not defaulted: a wording change that keeps
        the old version string is how two different measurements end up
        averaged on one leaderboard.
        """
        if version == self.version:
            raise ValueError("a guidance edit needs a new version string")
        crits = tuple(replace(c, guidance=guidance) if c.key == key else c
                      for c in self.criteria)
        return Rubric(version=version, criteria=crits)

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version,
                "criteria": [c.__dict__.copy() for c in self.criteria]}


def rubric_from_dict(doc: Mapping[str, Any]) -> Rubric:
    crits = []
    for c in doc.get("criteria", []):
        try:
            weight = float(c.get("weight", 0))
        except (TypeError, ValueError):
            weight = float("nan")
        crits.append(Criterion(key=str(c.get("key", "")),
                               label=str(c.get("label", "")),
                               weight=weight,
                               guidance=str(c.get("guidance", ""))))
    return Rubric(version=str(doc.get("version", "")), criteria=tuple(crits))


def load_rubric(path: str | Path = DEFAULT_RUBRIC_PATH) -> Rubric:
    return rubric_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_rubric(rubric: Rubric) -> list[str]:
    """Everything about a rubric that is decidable without a model.

    Returns a list of problems, empty when the rubric is sound. Each one is a
    plausible result of a careless edit in a JSON file, and each one makes
    scores meaningless or incomparable.
    """
    problems: list[str] = []
    if not rubric.version.strip():
        problems.append("the rubric has no version string")
    if not rubric.criteria:
        return problems + ["the rubric has no criteria at all"]

    keys = [c.key for c in rubric.criteria]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        problems.append(f"duplicate criterion keys: {', '.join(dupes)}")

    for c in rubric.criteria:
        if not _KEY_RE.fullmatch(c.key or ""):
            problems.append(f"criterion key {c.key!r} is not a valid slug")
        if not c.label.strip():
            problems.append(f"{c.key}: no label")
        if len(c.guidance.strip()) < MIN_GUIDANCE_CHARS:
            problems.append(
                f"{c.key}: guidance is {len(c.guidance.strip())} chars, under "
                f"the {MIN_GUIDANCE_CHARS} a criterion needs to be gradeable")
        if c.weight != c.weight:  # NaN
            problems.append(f"{c.key}: weight is not a number")
        elif c.weight < 0:
            problems.append(f"{c.key}: negative weight {c.weight}")

    if all((c.weight != c.weight) or c.weight <= 0 for c in rubric.criteria):
        problems.append("no criterion has a positive weight")
    return problems


def clamp_score(value: Any) -> int | None:
    """Coerce a model's score onto the scale. None for anything unreadable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return int(round(max(SCORE_MIN, min(SCORE_MAX, v))))


def weighted_overall(scores: Mapping[str, int | None],
                     weights: Mapping[str, float]) -> float | None:
    """Collapse per-criterion scores into one 0-10 number.

    A criterion that was declined (None) or is missing is treated as absent,
    not as zero: the remaining weights are renormalized. A call where every
    criterion was declined has no overall at all, and returns None rather
    than 0.0, because zero is a claim that the agent performed terribly.
    """
    total = 0.0
    total_w = 0.0
    for key, w in weights.items():
        s = scores.get(key)
        if s is None:
            continue
        total += float(s) * w
        total_w += w
    if total_w <= 0:
        return None
    return round(total / total_w, 2)


def checkable_keys(rubric: Rubric, keys: Sequence[str]) -> list[str]:
    """The subset of `keys` this rubric actually scores."""
    have = set(rubric.keys)
    return [k for k in keys if k in have]
