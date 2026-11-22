from __future__ import annotations

import pytest

from callqa.calibrate import (CohortMismatch, HumanPass, calibrate, render_markdown)
from callqa.judge import CallVerdict
from callqa.rubric import Criterion, Rubric

R = Rubric("v1", (
    Criterion("greet", "Greeting", 1, "Did the agent greet the caller properly?"),
    Criterion("name", "Name", 1, "Score low if the name was given and never used again."),
    Criterion("close", "Close", 2, "Did the call end with a clear next step?"),
))


def verdicts(rows, version="v1"):
    return [CallVerdict(cid, version, "j", scores=dict(s)) for cid, s in rows.items()]


def human(rows, version="v1", blind=True):
    return HumanPass(version, blind, {cid: dict(s) for cid, s in rows.items()})


JUDGE = {f"c{i}": {"greet": 8, "name": 2, "close": 7} for i in range(5)}
HUMAN = {f"c{i}": {"greet": 8, "name": 7, "close": 7} for i in range(5)}


def test_finds_the_planted_harsh_criterion_and_only_that_one():
    cal = calibrate(verdicts(JUDGE), human(HUMAN), R)
    assert [r.key for r in cal.suspect] == ["name"]
    name = next(r for r in cal.rollups if r.key == "name")
    assert name.direction == "judge harsh" and name.delta == pytest.approx(-5)


def test_gap_inside_tolerance_is_not_a_disagreement():
    j = {"c0": {"greet": 6, "name": 7, "close": 7}}
    cal = calibrate(verdicts(j), human({"c0": {"greet": 8, "name": 7, "close": 7}}), R)
    assert cal.crit_agreements == cal.crit_n == 3


def test_two_misses_are_not_called_systematic():
    j = {k: v for k, v in list(JUDGE.items())[:2]}
    h = {k: v for k, v in list(HUMAN.items())[:2]}
    assert calibrate(verdicts(j), human(h), R).suspect == []


def test_human_ungradable_is_overreach_not_disagreement():
    j = {"c0": {"greet": 0, "name": 0, "close": 0}}
    h = {"c0": {"greet": "ungradable", "name": "ungradable", "close": "ungradable"}}
    cal = calibrate(verdicts(j), human(h), R)
    assert len(cal.overreach) == 3 and cal.crit_n == 0 and cal.calls == []


def test_judge_decline_is_counted_separately():
    j = {"c0": {"greet": None, "name": 7, "close": 7}}
    cal = calibrate(verdicts(j), human({"c0": {"greet": 8, "name": 7, "close": 7}}), R)
    assert cal.judge_declined == [("c0", "greet", 8)] and cal.crit_n == 2


def test_overall_uses_rubric_weights_over_shared_criteria():
    j = {"c0": {"greet": 10, "name": 10, "close": 0}}
    h = {"c0": {"greet": 10, "name": 10, "close": 0}}
    cal = calibrate(verdicts(j), human(h), R)
    assert cal.calls[0].judge == pytest.approx(5.0)   # close carries half the weight


def test_mismatched_rubric_versions_are_refused():
    with pytest.raises(CohortMismatch):
        calibrate(verdicts(JUDGE, "v2"), human(HUMAN), R)


def test_reusing_a_human_pass_after_a_guidance_edit_is_explicit():
    r2 = R.with_guidance("name", "Using the name once is enough here.", "v2")
    cal = calibrate(verdicts(JUDGE, "v2"), human(HUMAN), r2, reuse_human_pass=True)
    assert "human pass graded on v1" in cal.rubric_version


def test_report_quotes_the_guidance_to_edit_and_carries_kappa():
    cal = calibrate(verdicts(JUDGE), human(HUMAN), R)
    md = render_markdown(cal, R)
    assert "suspect the rubric" in md
    assert "never used again" in md
    assert "kappa, pooled over 15 pairs" in md


def test_mixed_provenance_is_labeled_an_upper_bound():
    md = render_markdown(calibrate(verdicts(JUDGE), human(HUMAN, blind=False), R), R)
    assert "MIXED PROVENANCE" in md and "upper bound" in md
