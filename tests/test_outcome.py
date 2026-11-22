from __future__ import annotations

import pytest
from helpers import call

from callqa.judge import CallVerdict
from callqa.outcome import (CONFIRMED, NO_RUNWAY, NOT_CLAIMED, NOT_OBSERVED, UNVERIFIABLE,
                            LedgerRow, check_outcomes, last10, own_numbers,
                            render_markdown, summarize)

D = 86400
AGENT_LINE = "+12065550101"
CUSTOMER = "+12065550150"


def busy_agent_line(start=0):
    """Enough outbound calls from the agent's line for it to read as ours."""
    return [LedgerRow(f"x{i}", start + i * 3600, "outbound", AGENT_LINE, f"+120655501{60 + i}")
            for i in range(6)]


def verdict(cid, commit=8, outcome="follow_up"):
    return CallVerdict(cid, "v1", "j", scores={"commitment_or_booking_attempted": commit},
                       outcome=outcome)


def run(rows, calls, verdicts, days=7):
    return {r.call_id: r for r in check_outcomes(calls, verdicts, rows, days=days)}


def test_the_agents_own_line_does_not_confirm_anything():
    """The production bug: counterparty read as the originator of an
    outbound call, i.e. our own agent, who is on every later call."""
    scored = LedgerRow("s1", 0, "outbound", AGENT_LINE, CUSTOMER)
    ledger = [scored, *busy_agent_line(start=3600), LedgerRow("end", 10 * D, "inbound", "2065550199", "+12065550100")]
    out = run(ledger, [call("s1", "outbound", started_at=0)], [verdict("s1")])
    assert out["s1"].verdict == NOT_OBSERVED


def test_a_real_callback_confirms():
    ledger = [LedgerRow("s1", 0, "outbound", AGENT_LINE, CUSTOMER),
              LedgerRow("f1", 2 * D, "inbound", "2065550150", "+12065550100"),
              LedgerRow("end", 10 * D, "inbound", "2065550199", "+12065550100")]
    out = run(ledger, [call("s1", "outbound", started_at=0)], [verdict("s1")])
    assert out["s1"].verdict == CONFIRMED and out["s1"].first_gap_hours == pytest.approx(48)


def test_counterparty_comes_from_direction():
    ledger = [LedgerRow("s1", 0, "inbound", "(206) 555-0150", "+12065550100"),
              LedgerRow("f1", D, "outbound", AGENT_LINE, CUSTOMER),
              LedgerRow("end", 10 * D, "inbound", "2065550199", "+12065550100")]
    out = run(ledger, [call("s1", "inbound", started_at=0)], [verdict("s1")])
    assert out["s1"].verdict == CONFIRMED


def test_follow_up_outside_the_window_does_not_count():
    ledger = [LedgerRow("s1", 0, "outbound", AGENT_LINE, CUSTOMER),
              LedgerRow("f1", 8 * D, "inbound", CUSTOMER, "+12065550100"),
              LedgerRow("end", 10 * D, "inbound", "2065550199", "+12065550100")]
    assert run(ledger, [call("s1", "outbound")], [verdict("s1")])["s1"].verdict == NOT_OBSERVED


def test_a_call_before_the_scored_one_is_not_a_follow_up():
    ledger = [LedgerRow("p0", 0, "inbound", CUSTOMER, "+12065550100"),
              LedgerRow("s1", D, "outbound", AGENT_LINE, CUSTOMER),
              LedgerRow("end", 10 * D, "inbound", "2065550199", "+12065550100")]
    out = run(ledger, [call("s1", "outbound", started_at=D)], [verdict("s1")])
    assert out["s1"].verdict == NOT_OBSERVED


def test_no_claim_means_nothing_to_verify():
    ledger = [LedgerRow("s1", 0, "outbound", AGENT_LINE, CUSTOMER)]
    out = run(ledger, [call("s1", "outbound")], [verdict("s1", commit=3)])
    assert out["s1"].verdict == NOT_CLAIMED
    out = run(ledger, [call("s1", "outbound")], [verdict("s1", commit=None)])
    assert out["s1"].verdict == NOT_CLAIMED


def test_scored_at_the_end_of_the_ledger_is_no_runway_not_a_failure():
    ledger = [LedgerRow("s1", 10 * D - 3600, "outbound", AGENT_LINE, CUSTOMER),
              LedgerRow("end", 10 * D, "inbound", "2065550199", "+12065550100")]
    out = run(ledger, [call("s1", "outbound", started_at=10 * D - 3600)], [verdict("s1")])
    assert out["s1"].verdict == NO_RUNWAY


def test_missing_ledger_row_is_unverifiable():
    out = run([LedgerRow("end", 10 * D, "inbound", "2065550199", "+12065550100")],
              [call("s1", "outbound")], [verdict("s1")])
    assert out["s1"].verdict == UNVERIFIABLE


def test_own_numbers_finds_the_busy_lines():
    assert last10(AGENT_LINE) in own_numbers(busy_agent_line())


@pytest.mark.parametrize("raw,expected", [
    ("sip:12065550150@pbx.example.com", "2065550150"),
    ("+12065550150", "2065550150"),
    ("2065550150", "2065550150"),
    ("104", "104"),
])
def test_last10_normalizes_the_usual_formats(raw, expected):
    assert last10(raw) == expected


def test_promised_and_closed_are_reported_separately():
    ledger = [LedgerRow("a", 0, "outbound", AGENT_LINE, CUSTOMER),
              LedgerRow("b", 1, "outbound", AGENT_LINE, "+12065550151"),
              LedgerRow("f", D, "inbound", CUSTOMER, "+12065550100"),
              LedgerRow("end", 10 * D, "inbound", "2065550199", "+12065550100")]
    rows = check_outcomes([call("a", "outbound"), call("b", "outbound", started_at=1)],
                          [verdict("a", outcome="follow_up"), verdict("b", outcome="booked")], ledger)
    s = summarize(rows)
    assert (s["promised"], s["promised_kept"], s["closed"], s["closed_seen_again"]) == (1, 1, 1, 0)
    assert "a deal closed on the call needs no callback" in render_markdown(rows)


def test_a_counterparty_that_is_one_of_our_lines_is_refused():
    """One of our own numbers would match every call we ever made."""
    ledger = [LedgerRow("s1", 0, "outbound", "+12065550102", AGENT_LINE),
              *busy_agent_line(start=3600),
              LedgerRow("end", 10 * D, "inbound", "2065550199", "+12065550100")]
    out = run(ledger, [call("s1", "outbound")], [verdict("s1")])
    assert out["s1"].verdict == UNVERIFIABLE
