"""The questions that need counting, not judging.

Some rubric criteria are not judgments at all. "Did anyone say the call is
recorded" is a lookup. Paying a model to answer it costs money, adds latency,
and in production it got the answer wrong in the one direction a dashboard
cannot show: it credited the far end's phone menu with our agent's
compliance disclosure.

Every check returns one of three values: pass, fail or unknown. Unknown means
the transcript cannot answer the question (no speaker labels, nobody on the
line, a disclosure that may or may not be ours). `passed` is False for both
fail and unknown on purpose. Anything consuming a check as a boolean is
deciding whether to count a call as good, and "we could not tell" is not good.

The traps below came out of a real corpus, which is why they are here:

  * "monitor" and "record" are not disclosure words on their own. People say
    "the monitor on the wall is off" and voicemail greetings say "record your
    message at the tone". A disclosure needs a recording word AND a purpose
    word near it.
  * On an OUTBOUND call, a system voice is the other company's phone menu.
    Its "this call may be recorded" is their disclosure, not ours.
  * "I'm going to", "I'm sorry" and "This is what we need" are not
    self-introductions. The name after the phrase has to be capitalized.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .transcript import AGENT, CALLER, SYSTEM, Call

PASS = "pass"
FAIL = "fail"
UNKNOWN = "unknown"

Check = Callable[[Call], "TriResult"]


@dataclass(frozen=True)
class TriResult:
    name: str
    outcome: str
    detail: str

    @property
    def passed(self) -> bool:
        return self.outcome == PASS

    @property
    def is_unknown(self) -> bool:
        return self.outcome == UNKNOWN


def run_checks(checks: Iterable[Check], call: Call) -> list[TriResult]:
    return [c(call) for c in checks]


def all_passed(results: Iterable[TriResult]) -> bool:
    return all(r.passed for r in results)


# --- vocabulary --------------------------------------------------------------

_DISCLOSURE = re.compile(
    r"\b(record(?:ed|ing)?|monitor(?:ed|ing)?)\b[^.!?]{0,80}?"
    r"\b(quality|training|security|privacy|assurance|verification|"
    r"compliance)\b"
    r"|\b(quality|training|security|assurance)\b[^.!?]{0,60}?"
    r"\b(record(?:ed|ing)?|monitor(?:ed|ing)?)\b",
    re.IGNORECASE,
)

# Voicemail and call-screening phrases that contain "record" and are not
# disclosures. A match here vetoes the sentence.
_NOT_A_DISCLOSURE = re.compile(
    r"record your (name|message|reason)"
    r"|at the tone"
    r"|leave (a|your) message"
    r"|when you(?:'ve| have) finished recording"
    r"|press the pound key",
    re.IGNORECASE,
)

# The phrase is case-insensitive at its first letter (it can open a sentence)
# and the name after it is case-SENSITIVE: the capital is the discriminator.
_SELF_ID = re.compile(
    r"\b[Mm]y name(?:'s| is)\s+([A-Z][a-z]+)"
    r"|\b[Tt]his is\s+([A-Z][a-z]+)"
    r"|\b[Ii]t'?s\s+([A-Z][a-z]+)"
    r"|\b[Yy]ou(?:'re| are) speaking (?:with|to)\s+([A-Z][a-z]+)"
    r"|\b([A-Z][a-z]+)\s+speaking\b",
)

_NOT_A_NAME = frozenset({
    "The", "A", "An", "What", "Why", "How", "When", "Where", "Just", "Going",
    "Really", "Actually", "Probably", "Still", "Only", "Our", "Your", "My",
    "His", "Her", "Their", "It", "That", "This", "There", "Because", "So",
})

_LEGAL_SUFFIXES = frozenset({
    "inc", "incorporated", "llc", "ltd", "limited", "co", "corp",
    "corporation", "company", "lp", "llp", "plc",
})
_MIN_CONTAINMENT_CHARS = 6


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def is_disclosure(text: str) -> bool:
    return any(_DISCLOSURE.search(s) and not _NOT_A_DISCLOSURE.search(s)
               for s in _sentences(text))


def self_introduced_name(text: str) -> str | None:
    for m in _SELF_ID.finditer(text or ""):
        candidate = next(g for g in m.groups() if g)
        if candidate not in _NOT_A_NAME:
            return candidate
    return None


def normalize_org_name(value: str) -> str:
    """Case, punctuation, spacing and legal form removed.

    Deliberately not fuzzy. Speech recognition mangles company names, and the
    tempting fix is edit distance, but a false match here credits an agent
    with a greeting they never gave. A garbled spelling that keeps turning up
    goes on the configured list instead; it is one line.
    """
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    words = [w for w in text.split() if w not in _LEGAL_SUFFIXES]
    return "".join(words)


def org_name_matches(value: str, names: Sequence[str]) -> bool:
    target = normalize_org_name(value)
    if not target:
        return False
    for name in names:
        theirs = normalize_org_name(name)
        if not theirs:
            continue
        if theirs == target:
            return True
        shorter, longer = sorted((theirs, target), key=len)
        if len(shorter) >= _MIN_CONTAINMENT_CHARS and shorter in longer:
            return True
    return False


# --- the checks --------------------------------------------------------------

def gradeable_conversation(min_agent: int = 2, min_caller: int = 2,
                           name: str = "gradeable_conversation") -> Check:
    """Is there a two-sided conversation here at all?

    A precondition on the whole score. In production one call was five and a
    half minutes of a bank's phone menu and hold music: the agent dialed out
    and waited, and nobody ever answered. The judge scored every criterion 0
    and the call became the worst in that customer's corpus, top of a
    supervisor's review list, blaming an agent who did nothing wrong. The
    honest answer is "not gradeable", and a judge with no way to say that
    will say zero.

    No speaker labels at all returns UNKNOWN, not FAIL. Absent labels are
    not evidence of an absent conversation.
    """
    def check(call: Call) -> TriResult:
        if not call.labeled:
            return TriResult(name, UNKNOWN, "no speaker labels on this transcript")
        agent = len(call.by(AGENT))
        caller = len(call.by(CALLER))
        if agent >= min_agent and caller >= min_caller:
            return TriResult(name, PASS, f"agent {agent} turns, caller {caller} turns")
        return TriResult(name, FAIL,
                         f"agent {agent} turns, caller {caller} turns; needs "
                         f"{min_agent}/{min_caller}. Not a gradeable "
                         f"conversation: scoring it produces a made-up verdict, "
                         f"not a low one")
    check.check_name = name  # type: ignore[attr-defined]
    return check


def recording_disclosure(name: str = "compliance_recording_disclosure") -> Check:
    """Did OUR side disclose that the call is recorded?

    Attribution, in order:
      * an agent turn carrying a disclosure: ours. PASS.
      * a system turn on an INBOUND call: our own announcement, playing to
        someone who called us. PASS.
      * a system turn on an OUTBOUND call: the far end's phone menu. UNKNOWN,
        with the text quoted, because our announcement may simply not be in
        the recording.
      * nothing anywhere in a labeled transcript: FAIL.
      * no speaker labels: UNKNOWN. Guessing whose voice it was is the defect
        this check exists to remove.
    """
    def check(call: Call) -> TriResult:
        if not call.labeled:
            if is_disclosure(call.text):
                return TriResult(name, UNKNOWN, "disclosure language present but "
                                 "no speaker labels; cannot tell if it was ours")
            return TriResult(name, UNKNOWN, "no speaker labels")
        for text in call.by(AGENT):
            if is_disclosure(text):
                return TriResult(name, PASS, f"agent disclosed: {text[:70]!r}")
        hits = [t for t in call.by(SYSTEM) if is_disclosure(t)]
        if hits:
            if call.direction == "inbound":
                return TriResult(name, PASS,
                                 f"our announcement on an inbound call: {hits[0][:70]!r}")
            return TriResult(name, UNKNOWN,
                             f"disclosure is a system turn on an "
                             f"{call.direction or 'unknown-direction'} call, so it "
                             f"is most likely the far end's phone menu: "
                             f"{hits[0][:70]!r}")
        return TriResult(name, FAIL, "no recording disclosure by our side")
    check.check_name = name  # type: ignore[attr-defined]
    return check


def agent_identified_self(name: str = "agent_identified_self") -> Check:
    """Did the agent give their own name?

    Covers the factual half of the criterion. "Unprompted, early in the
    call" stays with the judge: "early" is a threshold nobody has agreed.
    """
    def check(call: Call) -> TriResult:
        turns = call.by(AGENT)
        if not turns:
            return TriResult(name, UNKNOWN, "no agent turns labeled on this transcript")
        for text in turns:
            found = self_introduced_name(text)
            if found:
                return TriResult(name, PASS, f"introduced as {found!r}")
        return TriResult(name, FAIL, "no self-introduction in any agent turn")
    check.check_name = name  # type: ignore[attr-defined]
    return check


def greeting_names_company(org_names: Sequence[str], opening_turns: int = 4,
                           name: str = "greeting_names_company") -> Check:
    """Was our company named in the opening?

    The objective half of `professional_greeting`. "Clear and courteous"
    stays with the judge. With no company names configured this returns
    UNKNOWN: nobody has given grounds to fail an agent on it.
    """
    names = [n for n in org_names if str(n).strip()]

    def check(call: Call) -> TriResult:
        if not names:
            return TriResult(name, UNKNOWN, "no company names configured; "
                             "nothing to match against")
        if not call.labeled:
            return TriResult(name, UNKNOWN, "no speaker labels")
        opening = [t for t in call.turns if t.speaker in (AGENT, SYSTEM)][:opening_turns]
        for t in opening:
            for phrase in re.split(r"[,.!?;:]", t.text):
                if org_name_matches(phrase, names):
                    return TriResult(name, PASS,
                                     f"named in opening: {phrase.strip()[:50]!r}")
        return TriResult(name, FAIL, f"company not named in the first "
                         f"{opening_turns} agent/system turns")
    check.check_name = name  # type: ignore[attr-defined]
    return check


# Which criterion each check stands in for. `gradeable_conversation` is not
# here: it is a precondition on the whole score and always runs.
CHECK_FOR_CRITERION = {
    "compliance_recording_disclosure": "recording_disclosure",
    "agent_identified_self": "agent_identified_self",
    "professional_greeting": "greeting_names_company",
}

# Criteria the checks take away from the judge entirely, as opposed to
# running alongside it.
FULLY_DEMOTED = ("compliance_recording_disclosure",)


def checks_for_rubric(criteria: Sequence[str], org_names: Sequence[str] = ()
                      ) -> list[Check]:
    """The checks a rubric actually asks for.

    A supervisor can turn a criterion off. Gating their numbers on a check
    for a criterion they disabled is not a stricter measurement, it is a
    wrong one: in production that exact mistake drove a whole pass^k table
    to zero.
    """
    wanted = set(criteria)
    suite: list[Check] = [gradeable_conversation()]
    if "compliance_recording_disclosure" in wanted:
        suite.append(recording_disclosure())
    if "agent_identified_self" in wanted:
        suite.append(agent_identified_self())
    if "professional_greeting" in wanted:
        suite.append(greeting_names_company(org_names))
    return suite
