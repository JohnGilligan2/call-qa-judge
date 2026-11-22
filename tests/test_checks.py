from __future__ import annotations

import pytest
from helpers import TWO_SIDED, call

from callqa.checks import (FAIL, PASS, UNKNOWN, agent_identified_self, all_passed,
                           checks_for_rubric, gradeable_conversation,
                           greeting_names_company, org_name_matches,
                           recording_disclosure, run_checks)

ORG = ["Bayside Garage Doors"]


def test_agent_disclosure_passes():
    c = call(turns=[("agent", "Just so you know, this call is recorded for quality purposes.")] + TWO_SIDED)
    assert recording_disclosure()(c).outcome == PASS


def test_far_end_menu_on_outbound_is_not_our_disclosure():
    c = call(direction="outbound", turns=[
        ("system", "Thank you for calling Northpoint Supply. This call may be recorded for quality assurance.")
    ] + TWO_SIDED)
    r = recording_disclosure()(c)
    assert r.outcome == UNKNOWN and "far end" in r.detail
    assert not r.passed


def test_our_own_announcement_on_inbound_is_ours():
    c = call(direction="inbound", turns=[
        ("system", "Calls are recorded for quality and training.")] + TWO_SIDED)
    assert recording_disclosure()(c).outcome == PASS


@pytest.mark.parametrize("text", [
    "We have to take the monitor off the wall first.",
    "Please record your message at the tone.",
    "The keypad monitor light is off.",
    "Leave a message and we'll record your name.",
])
def test_words_that_are_not_disclosures(text):
    c = call(turns=[("agent", text)] + TWO_SIDED)
    assert recording_disclosure()(c).outcome == FAIL


def test_no_speaker_labels_is_unknown_not_a_guess():
    c = call(text="This call may be recorded for quality assurance. Hi, how can I help?")
    assert recording_disclosure()(c).outcome == UNKNOWN
    assert gradeable_conversation()(c).outcome == UNKNOWN


def test_phone_menu_and_hold_only_is_not_gradeable():
    c = call(direction="outbound", turns=[
        ("system", "For orders, press 1."), ("system", "Please hold."),
        ("agent", "Still holding.")])
    r = gradeable_conversation()(c)
    assert r.outcome == FAIL and "made-up verdict" in r.detail


def test_two_sided_conversation_is_gradeable():
    assert gradeable_conversation()(call(turns=TWO_SIDED)).outcome == PASS


@pytest.mark.parametrize("text,name", [
    ("Hi, my name is Priya.", "Priya"),
    ("Bayside Garage Doors, this is Dana.", "Dana"),
    ("You're speaking with Luis.", "Luis"),
    ("Marcus speaking, how can I help?", "Marcus"),
])
def test_real_self_introductions_pass(text, name):
    r = agent_identified_self()(call(turns=[("agent", text)]))
    assert r.outcome == PASS and name in r.detail


@pytest.mark.parametrize("text", [
    "I'm going to check on that.", "This is what we need to do.",
    "It's going to be a while.", "This is The best option.",
])
def test_phrases_that_are_not_introductions(text):
    assert agent_identified_self()(call(turns=[("agent", text)])).outcome == FAIL


def test_no_agent_turns_is_unknown():
    assert agent_identified_self()(call(turns=[("caller", "Hello?")])).outcome == UNKNOWN


def test_company_named_in_opening_passes():
    assert greeting_names_company(ORG)(call(turns=TWO_SIDED)).outcome == PASS


def test_company_named_late_is_not_a_greeting():
    turns = [("agent", "Hello?"), ("caller", "Hi."), ("agent", "Yes?"), ("caller", "Door."),
             ("agent", "Okay."), ("agent", "Sure."), ("agent", "Right."),
             ("agent", "Anyway this is Bayside Garage Doors.")]
    assert greeting_names_company(ORG)(call(turns=turns)).outcome == FAIL


def test_unconfigured_company_names_is_unknown_not_a_failed_agent():
    assert greeting_names_company([])(call(turns=TWO_SIDED)).outcome == UNKNOWN


def test_legal_suffix_and_case_do_not_break_the_match():
    assert org_name_matches("bayside garage doors, LLC", ORG)
    assert not org_name_matches("Bay", ORG)


def test_unknown_is_never_a_pass():
    c = call(text="no labels here")
    results = run_checks(checks_for_rubric(["compliance_recording_disclosure"], ORG), c)
    assert all(r.outcome == UNKNOWN for r in results)
    assert not all_passed(results)


def test_disabled_criterion_gets_no_check():
    names = [c(call(turns=TWO_SIDED)).name for c in checks_for_rubric(["empathy_tone"], ORG)]
    assert names == ["gradeable_conversation"]
