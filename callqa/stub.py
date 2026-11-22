"""A rule-based stand-in for the model, so everything runs offline.

`rule_based_model()` returns a `complete(system, user, schema)` function, the
same shape `ModelJudge` takes for a real model. It reads the transcript and
the rubric guidance out of the prompt it is handed, so the whole prompt, schema
and parse path is exercised exactly as it would be live.

It is a toy grader, keyword rules and nothing more. What makes it useful is
that it can reproduce, on demand, the two judge defects the harness caught in
production:

  no_decline       the judge has no way to say "not gradeable", so a call
                   with nobody on the line gets every criterion scored 0.
  credit_far_end   the judge credits ANY recording announcement, including
                   the other company's phone menu on an outbound call.

And it reads the guidance for `caller_name_used` the way a model does. If the
guidance says "score low if the name was given and never used again", it
sets the bar at repeated use; reword the guidance and the bar moves. That is
what lets the example show a guidance edit fixing a harsh criterion without a
model in the loop.

`noise_seed` adds a reproducible +/-1 jitter, so replaying a call several
times behaves like a real, slightly noisy judge.
"""
from __future__ import annotations

import json
import random
import re
from typing import Iterable, Sequence

from .checks import is_disclosure, org_name_matches, self_introduced_name
from .rubric import MIN_GUIDANCE_CHARS

_TURN = re.compile(r"^(AGENT|CALLER|SYSTEM):\s?(.*)$")
_GUIDE = re.compile(r"^- ([a-z0-9_]+) \([^)]*\): ?(.*)$")
_DIRECTION = re.compile(r"direction (\w+)")

_OPEN_Q = re.compile(r"\b(what|when|how|which|where|why|tell me)\b", re.I)
_EMPATHY = ("sorry", "understand", "appreciate", "thank you", "thanks",
            "no problem", "happy to")
_OPTIONS = ("option", "either", "or we can", "$", "price", "cost", "quote")
_OBJECTION = ("expensive", "too much", "not sure", "cheaper", "other company",
              "think about it")
_ANSWER = ("because", "what i can do", "includes", "that covers", "warranty",
           "worth")
_ASK = ("book", "schedule", "deposit", "lock in", "set you up", "put you down",
        "reserve", "follow up", "call you back")
_CLOSE = ("tomorrow", "confirmation", "will call", "call you", "see you",
          "text you", "between", "all set", "monday", "tuesday", "wednesday",
          "thursday", "friday")
_BOOKED = ("you're all set", "you are all set", "booked", "confirmation number")
_FOLLOW = ("call you back", "call you tomorrow", "follow up", "get back to you")


def _parse(system: str, user: str):
    guidance = {}
    for line in system.splitlines():
        m = _GUIDE.match(line)
        if m:
            guidance[m.group(1)] = m.group(2)
    turns = []
    for line in user.splitlines():
        m = _TURN.match(line)
        if m:
            turns.append((m.group(1).lower(), m.group(2)))
    d = _DIRECTION.search(user.splitlines()[0] if user else "")
    return guidance, turns, (d.group(1) if d else "")


def _has(texts: Iterable[str], words: Sequence[str]) -> bool:
    blob = " ".join(texts).lower()
    return any(w in blob for w in words)


def rule_based_model(org_names: Sequence[str] = (), *, no_decline: bool = True,
                     credit_far_end: bool = True, noise_seed: int | None = None):
    rng = random.Random(noise_seed) if noise_seed is not None else None

    def complete(system: str, user: str, schema: dict) -> str:
        guidance, turns, direction = _parse(system, user)
        keys = list(schema["properties"]["criteria"]["required"])
        agent = [t for s, t in turns if s == "agent"]
        caller = [t for s, t in turns if s == "caller"]
        system_turns = [t for s, t in turns if s == "system"]
        two_sided = len(agent) >= 2 and len(caller) >= 2

        def item(score, evidence, gradeable=True):
            if rng is not None and gradeable and rng.random() < 0.15:
                score += rng.choice((-1, 1))
            return {"gradeable": gradeable, "score": max(0, min(10, score)),
                    "evidence": evidence}

        def disclosure_item():
            ours = any(is_disclosure(t) for t in agent) or (
                direction == "inbound" and any(is_disclosure(t) for t in system_turns))
            far = any(is_disclosure(t) for t in system_turns)
            if ours or (credit_far_end and far):
                return item(10, "recording announced")
            return item(0, "no recording announcement")

        out = {}
        if not two_sided:
            for k in keys:
                if k == "compliance_recording_disclosure" and no_decline:
                    out[k] = disclosure_item()
                elif no_decline:
                    out[k] = item(0, "nothing to credit")
                else:
                    out[k] = item(0, "no two-sided conversation", gradeable=False)
            return json.dumps({"criteria": out, "outcome": "unknown"})

        caller_name = next((n for n in (self_introduced_name(t) for t in caller) if n), None)
        name_uses = 0
        if caller_name:
            name_uses = sum(t.count(caller_name) for t in agent)

        for k in keys:
            g = guidance.get(k, "")
            if len(g.strip()) < MIN_GUIDANCE_CHARS:
                out[k] = item(0, "no usable guidance for this criterion")
                continue
            if k == "professional_greeting":
                first = agent[0] if agent else ""
                greet = bool(re.search(r"\b(hi|hello|good (morning|afternoon)|thanks for calling|thank you for calling)\b", first, re.I))
                named = any(org_name_matches(p, org_names) for p in re.split(r"[,.!?]", first))
                out[k] = item(9 if greet and named else 4 if greet else 2, first[:60])
            elif k == "agent_identified_self":
                found = any(self_introduced_name(t) for t in agent[:2])
                out[k] = item(9 if found else 1, agent[0][:60] if agent else "")
            elif k == "caller_name_used":
                repeated_bar = "never used again" in g.lower()
                if not caller_name:
                    out[k] = item(5, "caller never gave a name")
                elif repeated_bar:
                    score = 9 if name_uses >= 2 else 3 if name_uses == 1 else 1
                    out[k] = item(score, f"name used {name_uses} time(s)")
                else:
                    score = 9 if name_uses >= 2 else 8 if name_uses == 1 else 2
                    out[k] = item(score, f"name used {name_uses} time(s)")
            elif k == "needs_discovery":
                qs = sum(1 for t in agent if "?" in t and _OPEN_Q.search(t))
                out[k] = item(2 if qs == 0 else 5 if qs == 1 else 8, f"{qs} open question(s)")
            elif k == "empathy_tone":
                n = sum(1 for t in agent if _has([t], _EMPATHY))
                out[k] = item(min(9, 4 + 2 * n), f"{n} warm turn(s)")
            elif k == "clear_options_offered":
                out[k] = item(8 if _has(agent, _OPTIONS) else 4, "options")
            elif k == "objection_handling":
                if _has(caller, _OBJECTION):
                    out[k] = item(8 if _has(agent, _ANSWER) else 3, "objection raised")
                else:
                    out[k] = item(6, "no objection raised")
            elif k == "commitment_or_booking_attempted":
                if _has(agent, _ASK):
                    out[k] = item(8, "asked for the booking")
                elif _has(agent, ("let us know",)):
                    out[k] = item(2, "ended on 'let us know'")
                else:
                    out[k] = item(3, "no ask")
            elif k == "accurate_information":
                hedges = _has(agent, ("i think maybe", "not sure but", "probably",
                                      "depends", "could be a lot"))
                out[k] = item(5 if hedges else 7, "consistency")
            elif k == "call_control_pacing":
                out[k] = item(7 if 3 <= len(agent) <= 15 else 5, f"{len(agent)} agent turns")
            elif k == "proper_close_next_steps":
                out[k] = item(8 if _has(agent[-2:], _CLOSE) else 3, "close")
            elif k == "compliance_recording_disclosure":
                out[k] = disclosure_item()
            else:
                out[k] = item(5, "criterion not known to the stub")

        outcome = ("booked" if _has(agent, _BOOKED)
                   else "follow_up" if _has(agent, _FOLLOW)
                   else "no_commitment")
        return json.dumps({"criteria": out, "outcome": outcome})

    return complete
