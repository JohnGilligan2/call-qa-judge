"""The one check that leaves the transcript.

Everything else in this package grades what was SAID. A judge reading a
transcript can only ever confirm that an agent said the right thing. The
failure that costs money is the agent who says it beautifully and nothing
follows, and no rubric wording can see that, because the transcript is word
perfect.

So this asks the phone system's own call ledger a question the transcript
cannot answer. For every scored call where the judge credited
`commitment_or_booking_attempted`, was that number on a call with us again
inside the window?

  CONFIRMED      the agent asked, and a later call with that number exists.
  NOT_OBSERVED   the agent asked and no later call exists in the window.
                 Read it as "no follow-up contact on this phone system", not
                 "the customer was lost": they may have emailed.
  NOT_CLAIMED    the judge did not credit a commitment. Nothing to verify.
  UNVERIFIABLE   no usable counterparty number (withheld, anonymous, no
                 ledger row, or one of our own lines).
  NO_RUNWAY      scored less than a day before the ledger ends, so a
                 follow-up could never have been observed. Counting it as
                 NOT_OBSERVED would manufacture a failure out of missing data.

A follow-up call is evidence of CONTACT. It is not evidence of a booking:
nothing here reads a calendar or a CRM.

The bug this module is shaped around: its first version in production
reported 100 percent confirmation, which was the tell. On an outbound call
the originating number is OUR agent's line, so "was this number on a later
call" was really asking "did this agent make any other call", and the answer
is always yes. The counterparty therefore comes from the ledger and the call
direction, and any number that originates many calls (our own lines) is
refused as a counterparty.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .judge import CallVerdict
from .transcript import Call

CONFIRMED = "CONFIRMED"
NOT_OBSERVED = "NOT_OBSERVED"
NOT_CLAIMED = "NOT_CLAIMED"
UNVERIFIABLE = "UNVERIFIABLE"
NO_RUNWAY = "NO_RUNWAY"

COMMITMENT_KEY = "commitment_or_booking_attempted"
DEFAULT_CLAIM_THRESHOLD = 6
OWN_LINE_MIN_CALLS = 5

_NON_DIGITS = re.compile(r"\D+")


def last10(value: str) -> str:
    """The last ten digits: what two spellings of one number share.

    A SIP URI with a country code, an E.164 string and a bare national number
    all reduce to the same ten digits. Anything shorter is returned as is,
    because an extension is not a phone number and must not match one.
    """
    digits = _NON_DIGITS.sub("", str(value or "").split("@")[0])
    return digits[-10:] if len(digits) >= 10 else digits


@dataclass(frozen=True)
class LedgerRow:
    call_id: str
    started_at: int
    direction: str
    from_number: str
    to_number: str


def load_ledger(path: str | Path) -> list[LedgerRow]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return [LedgerRow(str(r["call_id"]), int(r["started_at"]),
                      str(r.get("direction", "")).lower(),
                      str(r.get("from", "")), str(r.get("to", "")))
            for r in doc.get("calls", [])]


def own_numbers(ledger: Sequence[LedgerRow], min_calls: int = OWN_LINE_MIN_CALLS) -> set[str]:
    """Numbers that originate many calls: our own lines, read off the ledger."""
    counts = Counter(last10(r.from_number) for r in ledger)
    return {n for n, c in counts.items() if c >= min_calls and len(n) >= 10}


def counterparty(row: LedgerRow | None) -> str:
    """The OTHER party, by direction. Never the originator of an outbound call."""
    if row is None:
        return ""
    if row.direction == "inbound":
        return last10(row.from_number)
    return last10(row.to_number)


@dataclass
class OutcomeRow:
    call_id: str
    agent: str
    commitment: int | None
    judge_outcome: str
    verdict: str
    follow_ups: int
    first_gap_hours: float | None
    runway_days: float


def check_outcomes(calls: Sequence[Call], verdicts: Sequence[CallVerdict],
                   ledger: Sequence[LedgerRow], *, days: int = 7,
                   threshold: int = DEFAULT_CLAIM_THRESHOLD) -> list[OutcomeRow]:
    rows_by_id = {r.call_id: r for r in ledger}
    v_by_id = {v.call_id: v for v in verdicts}
    ours = own_numbers(ledger)
    ledger_end = max((r.started_at for r in ledger), default=0)
    window = days * 86400
    out: list[OutcomeRow] = []
    for call in calls:
        v = v_by_id.get(call.call_id)
        if v is None:
            continue
        score = v.scores.get(COMMITMENT_KEY)
        number = counterparty(rows_by_id.get(call.call_id))
        if number in ours:
            number = ""
        runway = max(0.0, (ledger_end - call.started_at) / 86400.0)
        hits: list[int] = []
        if score is None or score < threshold:
            verdict = NOT_CLAIMED
        elif len(number) < 10:
            verdict = UNVERIFIABLE
        else:
            hits = sorted(r.started_at for r in ledger
                          if number in (last10(r.from_number), last10(r.to_number))
                          and call.started_at < r.started_at <= call.started_at + window)
            if hits:
                verdict = CONFIRMED
            elif runway < 1.0:
                verdict = NO_RUNWAY
            else:
                verdict = NOT_OBSERVED
        gap = (hits[0] - call.started_at) / 3600.0 if hits else None
        out.append(OutcomeRow(call.call_id, call.agent, score, v.outcome, verdict,
                              len(hits), gap, round(runway, 1)))
    return out


def summarize(rows: Sequence[OutcomeRow]) -> dict[str, object]:
    counts = Counter(r.verdict for r in rows)
    claimed = [r for r in rows if r.verdict in (CONFIRMED, NOT_OBSERVED)]
    promised = [r for r in claimed if r.judge_outcome == "follow_up"]
    closed = [r for r in claimed if r.judge_outcome == "booked"]
    return {
        "counts": dict(counts),
        "claimed": len(claimed),
        "confirmed": counts.get(CONFIRMED, 0),
        "promised": len(promised),
        "promised_kept": sum(1 for r in promised if r.verdict == CONFIRMED),
        "closed": len(closed),
        "closed_seen_again": sum(1 for r in closed if r.verdict == CONFIRMED),
    }


def render_markdown(rows: Sequence[OutcomeRow], days: int = 7) -> str:
    s = summarize(rows)
    c: Mapping[str, int] = s["counts"]  # type: ignore[assignment]
    L = [f"## Outcome check ({days}-day window)", ""]
    L.append(f"Of {s['claimed']} calls where the judge credited a commitment and a "
             f"follow-up window existed, {s['confirmed']} show a later call with "
             f"the same number.")
    if s["promised"]:
        L.append("")
        L.append(f"Split by what the judge said the call was: of {s['promised']} "
                 f"calls flagged follow_up (a promise of further contact), "
                 f"{s['promised_kept']} show that contact. That is the number that "
                 f"means something.")
    if s["closed"]:
        L.append(f"Of {s['closed']} calls flagged booked, {s['closed_seen_again']} "
                 f"show a later call. A low number here is not a defect: a deal "
                 f"closed on the call needs no callback.")
    L += ["", "| verdict | n |", "|---|---|"]
    for name in (CONFIRMED, NOT_OBSERVED, NOT_CLAIMED, UNVERIFIABLE, NO_RUNWAY):
        L.append(f"| {name} | {c.get(name, 0)} |")
    L += ["", "| call | agent | commitment | judge outcome | verdict | follow-ups | first gap |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.verdict == NOT_CLAIMED:
            continue
        gap = f"{r.first_gap_hours:.1f}h" if r.first_gap_hours is not None else "-"
        L.append(f"| {r.call_id} | {r.agent or '-'} | {r.commitment} | {r.judge_outcome} "
                 f"| {r.verdict} | {r.follow_ups} | {gap} |")
    return "\n".join(L) + "\n"
