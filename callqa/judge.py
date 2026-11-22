"""The judge: one verdict per criterion, with a legal way to decline.

The design choice that matters most is the decline. A judge that must score
every criterion on every call will confidently score calls nobody could
grade, and in production that put a phone-menu hold loop at the top of a
supervisor's worst-calls list. Here each criterion comes back as
`{"gradeable": bool, "score": int, "evidence": str}`, and `gradeable: false`
is recorded as None, which every downstream number treats as absent rather
than zero.

The model call is injected. `ModelJudge` takes any function
`complete(system, user, schema) -> str` that returns JSON text, so tests and
the example run offline with a stub, and production can wrap whatever client
it uses. `AnthropicJudge` is that wrapper for the Anthropic Messages API with
a JSON-schema constrained response; it imports the SDK only when built.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .rubric import Rubric, clamp_score, weighted_overall
from .transcript import Call

OUTCOMES = ("booked", "follow_up", "no_commitment", "unknown")

Complete = Callable[[str, str, dict], str]


@dataclass
class CallVerdict:
    call_id: str
    rubric_version: str
    judge: str
    scores: dict[str, int | None] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)
    outcome: str = "unknown"
    problems: list[str] = field(default_factory=list)

    def overall(self, weights: Mapping[str, float]) -> float | None:
        return weighted_overall(self.scores, weights)

    @property
    def declined(self) -> list[str]:
        return [k for k, v in self.scores.items() if v is None]


class Judge(Protocol):
    name: str

    def score(self, call: Call, rubric: Rubric) -> CallVerdict: ...


def verdict_schema(rubric: Rubric) -> dict[str, Any]:
    """The schema the model is constrained to.

    Every property is required and `additionalProperties` is false: a
    partially filled verdict is worse than a failed one, because it averages
    silently into a leaderboard. No numeric bounds, because constrained
    decoding does not support them; scores are clamped on the way in.
    """
    per = {
        "type": "object",
        "properties": {
            "gradeable": {"type": "boolean"},
            "score": {"type": "integer"},
            "evidence": {"type": "string"},
        },
        "required": ["gradeable", "score", "evidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "object",
                "properties": {k: per for k in rubric.keys},
                "required": list(rubric.keys),
                "additionalProperties": False,
            },
            "outcome": {"type": "string", "enum": list(OUTCOMES)},
        },
        "required": ["criteria", "outcome"],
        "additionalProperties": False,
    }


def build_prompt(call: Call, rubric: Rubric) -> tuple[str, str]:
    lines = [
        "You grade one recorded customer call against a rubric.",
        "Score each criterion from 0 to 10 using only what the transcript shows.",
        "If a criterion cannot be judged from this call (nobody answered, the",
        "call is a phone menu or voicemail, the topic never came up), set",
        "gradeable to false. Declining is a correct answer, not a failure;",
        "a score you cannot support is the failure.",
        "Quote the transcript in evidence.",
        "",
        f"Rubric version: {rubric.version}",
    ]
    for c in rubric.criteria:
        lines.append(f"- {c.key} ({c.label}): {c.guidance}")
    lines += ["", "outcome: booked if the business was won on the call, "
              "follow_up if further contact was promised, no_commitment "
              "otherwise, unknown if you cannot tell."]
    user = (f"Call {call.call_id}, direction {call.direction or 'unknown'}.\n\n"
            f"{call.full_text}")
    return "\n".join(lines), user


def parse_verdict(raw: str, call: Call, rubric: Rubric, judge: str) -> CallVerdict:
    """Turn the model's JSON into a verdict, loudly.

    A criterion missing from the response is recorded as declined AND listed
    in `problems`. It never becomes a zero.
    """
    v = CallVerdict(call_id=call.call_id, rubric_version=rubric.version, judge=judge)
    try:
        doc = json.loads(raw)
    except (TypeError, ValueError):
        v.scores = {k: None for k in rubric.keys}
        v.problems.append("response was not JSON; every criterion declined")
        return v
    crits = doc.get("criteria") or {}
    for key in rubric.keys:
        item = crits.get(key)
        if not isinstance(item, Mapping):
            v.scores[key] = None
            v.problems.append(f"{key}: missing from response")
            continue
        if item.get("gradeable") is False:
            v.scores[key] = None
        else:
            score = clamp_score(item.get("score"))
            if score is None:
                v.problems.append(f"{key}: unreadable score")
            v.scores[key] = score
        v.evidence[key] = str(item.get("evidence", ""))
    outcome = str(doc.get("outcome", "unknown"))
    v.outcome = outcome if outcome in OUTCOMES else "unknown"
    return v


class ModelJudge:
    """A judge backed by any text-in, JSON-out model call."""

    def __init__(self, complete: Complete, name: str = "model-judge") -> None:
        self._complete = complete
        self.name = name

    def score(self, call: Call, rubric: Rubric) -> CallVerdict:
        system, user = build_prompt(call, rubric)
        raw = self._complete(system, user, verdict_schema(rubric))
        return parse_verdict(raw, call, rubric, self.name)


class AnthropicJudge(ModelJudge):
    """ModelJudge over the Anthropic Messages API.

    Never constructed by the tests or the example. Pass `client=` to reuse
    one you already have; otherwise the SDK is imported here and reads
    ANTHROPIC_API_KEY from the environment.
    """

    def __init__(self, model: str, *, client: Any = None,
                 max_tokens: int = 4000) -> None:
        if client is None:
            import anthropic  # optional dependency
            client = anthropic.Anthropic()
        self._client = client
        self.model = model
        self.max_tokens = max_tokens
        super().__init__(self._call, name=f"anthropic:{model}")

    def _call(self, system: str, user: str, schema: dict) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": user}],
        )
        # A refusal comes back as HTTP 200 with no content. Check before
        # indexing, or the crash hides the reason.
        if getattr(resp, "stop_reason", None) == "refusal":
            return ""
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", "") == "text":
                return block.text
        return ""
