"""No transcript text in any artifact this package writes.

Calibration needs scores, not conversations. The trap is a field that looks
like metadata and is not: a judge's `evidence` is a verbatim quote from the
call, often with a customer's name or account detail in it. In production it
was the field that would actually have leaked, so it is on the list.

`assert_no_transcript` is a positive check on the way out, not a habit of
remembering. It fails the write.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator, Mapping

FORBIDDEN_KEYS = frozenset({
    "transcript", "turns", "utterances", "text", "evidence",
    "call_summary", "coaching_note",
})

# Seven or more consecutive digits reads as a phone or account number. A
# masked value ("*****0142") breaks the run and passes.
_LONG_DIGIT_RUN = re.compile(r"\d{7,}")


class LeakError(RuntimeError):
    pass


def _walk(node: Any, path: str = "$") -> Iterator[tuple[str, Any]]:
    yield path, node
    if isinstance(node, Mapping):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def mask_number(value: str) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    return "*" * max(0, len(digits) - 4) + digits[-4:] if digits else ""


def assert_no_transcript(payload: Any) -> None:
    for path, node in _walk(payload):
        if isinstance(node, Mapping):
            for key in node:
                if str(key).lower() in FORBIDDEN_KEYS:
                    raise LeakError(f"{path}.{key}: {key!r} carries transcript "
                                    f"text and must not be written to an artifact")
        if isinstance(node, str):
            m = _LONG_DIGIT_RUN.search(node)
            if m:
                raise LeakError(f"{path}: {m.group(0)[:3]}... looks like an "
                                f"unmasked number; mask it first")


def write_json(path: str | Path, payload: Any) -> Path:
    assert_no_transcript(payload)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
