from __future__ import annotations

from callqa.transcript import Call, Turn


def call(call_id="t1", direction="inbound", turns=(), started_at=0, agent="", text=""):
    return Call(call_id=call_id, direction=direction,
                turns=tuple(Turn(s, t) for s, t in turns),
                started_at=started_at, agent=agent, text=text)


TWO_SIDED = [
    ("agent", "Hi, thanks for calling Bayside Garage Doors, this is Dana."),
    ("caller", "Hi, this is Erin. My door is stuck."),
    ("agent", "Sorry to hear that, Erin. What happens when you press the button?"),
    ("caller", "It hums and stops."),
]
