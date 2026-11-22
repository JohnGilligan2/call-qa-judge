"""A recorded call, as the judge and the checks see it.

Speaker labels matter more than anything else on this object. A transcript
with no labels cannot say whose voice disclosed the recording or whether
anyone was on the other end of the line, and every check in `checks.py`
answers UNKNOWN for it instead of guessing.

Speakers are `agent`, `caller` and `system` (an IVR, a hold message, a
voicemail greeting, a platform announcement).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

AGENT = "agent"
CALLER = "caller"
SYSTEM = "system"


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str


@dataclass(frozen=True)
class Call:
    call_id: str
    direction: str                 # "inbound" or "outbound"
    turns: tuple[Turn, ...] = ()
    started_at: int = 0            # unix seconds
    agent: str = ""
    text: str = ""                 # flat transcript, for unlabeled calls
    meta: Mapping[str, Any] = field(default_factory=dict)

    def by(self, speaker: str) -> list[str]:
        return [t.text for t in self.turns if t.speaker.lower() == speaker]

    @property
    def labeled(self) -> bool:
        return bool(self.turns)

    @property
    def full_text(self) -> str:
        if self.turns:
            return "\n".join(f"{t.speaker.upper()}: {t.text}" for t in self.turns)
        return self.text


def call_from_dict(doc: Mapping[str, Any]) -> Call:
    turns = tuple(Turn(str(t.get("speaker", "")).lower(), str(t.get("text", "")))
                  for t in doc.get("turns", []) or [])
    return Call(call_id=str(doc["call_id"]),
                direction=str(doc.get("direction", "")).lower(),
                turns=turns,
                started_at=int(doc.get("started_at", 0) or 0),
                agent=str(doc.get("agent", "")),
                text=str(doc.get("text", "")),
                meta=dict(doc.get("meta", {}) or {}))


def load_calls(path: str | Path) -> list[Call]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return [call_from_dict(c) for c in doc.get("calls", [])]
