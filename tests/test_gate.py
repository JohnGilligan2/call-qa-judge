from __future__ import annotations

import json
from pathlib import Path

import pytest

from callqa.calibrate import load_human
from callqa.gate import (GoldenCase, agreement_score, band_for, break_rubric, evaluate,
                         gate, golden_from_human, in_band, self_test)
from callqa.judge import ModelJudge
from callqa.rubric import load_rubric, validate_rubric
from callqa.stub import rule_based_model
from callqa.transcript import load_calls

DATA = Path(__file__).resolve().parents[1] / "example" / "data"
ORG = json.loads((DATA / "calls.json").read_text(encoding="utf-8"))["org_names"]


@pytest.fixture(scope="module")
def golden():
    return golden_from_human(load_calls(DATA / "calls.json"), load_human(DATA / "human_verdicts.json"))


def fixed_judge(seed=None):
    return ModelJudge(rule_based_model(ORG, no_decline=False, credit_far_end=False, noise_seed=seed))


@pytest.mark.parametrize("score,band", [(0, "absent"), (3, "poor"), (6, "adequate"), (8, "good"), (10, "excellent")])
def test_bands(score, band):
    assert band_for(score) == band and in_band(score, band)


def test_declining_an_ungradable_criterion_is_agreement():
    assert in_band(None, "ungradable") and not in_band(0, "ungradable")
    assert not in_band(None, "good")


def test_a_dropped_criterion_counts_as_a_miss():
    assert agreement_score({"a": 8}, {"a": "good", "b": "good"}) == 5.0


def test_golden_set_comes_from_the_human_pass(golden):
    assert len(golden) == 11
    hold = next(g for g in golden if g.call.call_id == "c03")
    assert set(hold.bands.values()) == {"ungradable"}


def test_identical_rubrics_tie_and_a_tie_loses(golden):
    r = load_rubric()
    a = evaluate("a", golden, fixed_judge(), r)
    b = evaluate("b", golden, fixed_judge(), r)
    d = gate(a, b)
    assert not d.promote and "tie loses" in d.reason


def test_the_name_guidance_fix_is_promoted(golden):
    r = load_rubric()
    r2 = r.with_guidance("caller_name_used", "Use the name at least once; once is enough.", "v2")
    d = gate(evaluate("inc", golden, fixed_judge(), r), evaluate("cand", golden, fixed_judge(), r2))
    assert d.promote, d.to_text()
    assert "switched off" in d.to_text()


def test_a_generous_rewording_does_not_win_by_inflating_scores(golden):
    """Scoring against human bands is what blocks this exploit."""
    r = load_rubric()
    from callqa.rubric import Criterion, Rubric
    generous = Rubric("v-generous", tuple(
        Criterion(c.key, c.label, c.weight, c.guidance) for c in r.criteria))

    class Generous:
        name = "generous"

        def score(self, call, rubric):
            v = fixed_judge().score(call, rubric)
            v.scores = {k: (None if s is None else 10) for k, s in v.scores.items()}
            return v

    inc = evaluate("inc", golden, fixed_judge(), r)
    cand = evaluate("cand", golden, Generous(), generous)
    assert not gate(inc, cand).promote


def test_broken_rubric_is_rejected_and_materially_worse(golden):
    r = load_rubric()
    assert validate_rubric(break_rubric(r))
    ok, decision = self_test(golden, fixed_judge(), r)
    assert ok, decision.to_text()
    assert "structurally broken" in decision.reason


def test_replays_are_honored_and_noise_shows_up(golden):
    run = evaluate("x", golden, fixed_judge(seed=7), load_rubric(), replays=4)
    assert all(c.replays == 4 for c in run.cases)
    assert 0 < run.per_trial < 1


def test_pass_pow_k_decays_and_pass_at_k_rises(golden):
    run = evaluate("x", golden, fixed_judge(seed=7), load_rubric(), replays=4)
    assert run.pass_pow_k(5) < run.per_trial < run.pass_at_k(5)
