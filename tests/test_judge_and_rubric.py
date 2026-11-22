from __future__ import annotations

import json

import pytest
from helpers import TWO_SIDED, call

from callqa.judge import AnthropicJudge, ModelJudge, parse_verdict, verdict_schema
from callqa.rubric import (Criterion, Rubric, load_rubric, validate_rubric,
                           weighted_overall)
from callqa.stub import rule_based_model


def test_shipped_rubric_is_sound_and_editable_data():
    r = load_rubric()
    assert validate_rubric(r) == []
    assert len(r.keys) == 12
    assert sum(r.weights.values()) == pytest.approx(1.0)


def test_weights_are_relative_and_renormalized():
    r = Rubric("v", (Criterion("a", "A", 3, "x" * 30), Criterion("b", "B", 1, "y" * 30)))
    assert r.weights == {"a": 0.75, "b": 0.25}


@pytest.mark.parametrize("crits,needle", [
    ((), "no criteria"),
    ((Criterion("a", "A", 1, "short"),), "guidance is 5 chars"),
    ((Criterion("a", "A", 1, "x" * 30), Criterion("a", "A", 1, "x" * 30)), "duplicate"),
    ((Criterion("Bad Key", "A", 1, "x" * 30),), "not a valid slug"),
    ((Criterion("a", "A", -1, "x" * 30),), "negative weight"),
    ((Criterion("a", "A", 0, "x" * 30),), "no criterion has a positive weight"),
])
def test_structural_defects_are_caught(crits, needle):
    assert any(needle in p for p in validate_rubric(Rubric("v1", crits)))


def test_guidance_edit_requires_a_new_version():
    r = load_rubric()
    with pytest.raises(ValueError):
        r.with_guidance("caller_name_used", "x" * 40, r.version)
    r2 = r.with_guidance("caller_name_used", "x" * 40, "v2")
    assert r2.guidance("caller_name_used") == "x" * 40
    assert r.guidance("caller_name_used") != r2.guidance("caller_name_used")


def test_declined_criterion_is_absent_not_zero():
    w = {"a": 0.5, "b": 0.5}
    assert weighted_overall({"a": 8, "b": None}, w) == 8.0
    assert weighted_overall({"a": None, "b": None}, w) is None


def test_model_call_is_injected_and_gets_the_schema_and_guidance():
    seen = {}
    r = load_rubric()

    def fake(system, user, schema):
        seen.update(system=system, user=user, schema=schema)
        crits = {k: {"gradeable": True, "score": 7, "evidence": "q"} for k in r.keys}
        crits["caller_name_used"] = {"gradeable": False, "score": 0, "evidence": ""}
        return json.dumps({"criteria": crits, "outcome": "booked"})

    v = ModelJudge(fake, name="fake").score(call(turns=TWO_SIDED), r)
    assert set(seen["schema"]["properties"]["criteria"]["required"]) == set(r.keys)
    assert r.guidance("needs_discovery") in seen["system"]
    assert "AGENT: Hi, thanks for calling" in seen["user"]
    assert v.scores["caller_name_used"] is None and v.declined == ["caller_name_used"]
    assert v.outcome == "booked" and v.problems == []


def test_missing_and_garbage_responses_decline_loudly():
    r = load_rubric()
    c = call(turns=TWO_SIDED)
    v = parse_verdict("not json", c, r, "x")
    assert all(s is None for s in v.scores.values()) and v.problems
    v = parse_verdict(json.dumps({"criteria": {"empathy_tone": {"gradeable": True, "score": 14, "evidence": ""}}}), c, r, "x")
    assert v.scores["empathy_tone"] == 10                     # clamped
    assert v.scores["needs_discovery"] is None                # missing, never zero
    assert any("needs_discovery: missing" in p for p in v.problems)


def test_schema_is_closed():
    s = verdict_schema(load_rubric())
    assert s["additionalProperties"] is False
    assert s["properties"]["criteria"]["additionalProperties"] is False


def test_stub_defects_reproduce_the_production_failures():
    r = load_rubric()
    hold = call(direction="outbound", turns=[
        ("system", "This call may be recorded for quality assurance."),
        ("system", "Please hold."), ("agent", "Still holding.")])
    as_found = ModelJudge(rule_based_model()).score(hold, r)
    assert as_found.scores["empathy_tone"] == 0
    assert as_found.scores["compliance_recording_disclosure"] == 10
    fixed = ModelJudge(rule_based_model(no_decline=False, credit_far_end=False)).score(hold, r)
    assert all(s is None for s in fixed.scores.values())
    assert fixed.overall(r.weights) is None


def test_stub_reads_the_guidance_like_a_model_would():
    r = load_rubric()
    one_use = call(turns=[
        ("agent", "Hi, Bayside Garage Doors, this is Dana."),
        ("caller", "Hi, this is Tom."),
        ("agent", "Thanks, Tom. What can I do for you?"),
        ("caller", "Door's stuck.")])
    judge = ModelJudge(rule_based_model())
    harsh = judge.score(one_use, r).scores["caller_name_used"]
    r2 = r.with_guidance("caller_name_used", "Use the name at least once; once is enough.", "v2")
    fair = judge.score(one_use, r2).scores["caller_name_used"]
    assert harsh <= 3 < 7 <= fair


def test_stub_noise_is_reproducible():
    r = load_rubric()
    c = call(turns=TWO_SIDED)
    a = [ModelJudge(rule_based_model(noise_seed=3)).score(c, r).scores for _ in range(1)]
    b = [ModelJudge(rule_based_model(noise_seed=3)).score(c, r).scores for _ in range(1)]
    assert a == b


class _Block:
    type = "text"
    text = json.dumps({"criteria": {}, "outcome": "unknown"})


class _Resp:
    stop_reason = "end_turn"
    content = [_Block()]


class _Client:
    def __init__(self):
        self.kwargs = None
        self.messages = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _Resp()


def test_anthropic_judge_request_shape_without_network():
    client = _Client()
    j = AnthropicJudge("some-model", client=client)
    v = j.score(call(turns=TWO_SIDED), load_rubric())
    assert client.kwargs["output_config"]["format"]["type"] == "json_schema"
    assert client.kwargs["model"] == "some-model"
    assert all(s is None for s in v.scores.values())   # empty criteria: all declined, loudly
    assert v.problems
