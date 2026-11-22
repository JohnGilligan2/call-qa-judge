from __future__ import annotations

import contextlib
import io
import runpy
from pathlib import Path

import pytest

from callqa.leak import LeakError, assert_no_transcript, mask_number, write_json
from callqa.passk import PassKRow, explain_gap, passk_table, passk_table_with_checks, render_markdown

ROOT = Path(__file__).resolve().parents[1]


def test_pass_pow_k_is_the_documented_decay():
    r = PassKRow(threshold=6, n=10, passes=9)
    assert r.pass_pow_k(5) == pytest.approx(0.59, abs=0.001)
    assert r.pass_pow_k(10) == pytest.approx(0.349, abs=0.001)
    assert r.pass_at_k(5) == pytest.approx(0.99999)


def test_passk_table_sweeps_thresholds():
    rows = passk_table([3, 5, 7, 9], [5, 8])
    assert [(r.threshold, r.passes) for r in rows] == [(5.0, 3), (8.0, 1)]


def test_a_failed_or_unknown_check_is_not_a_success():
    items = ["a", "b"]
    rows = passk_table_with_checks(items, [5], {"a": 9, "b": 9}.get, lambda i: i == "a")
    assert rows[0].passes == 1
    rows = passk_table_with_checks(["c"], [5], {"c": None}.get, lambda i: True)
    assert rows[0].passes == 0     # an ungraded call is never a pass


def test_render_bolds_pass_pow_k():
    md = render_markdown(passk_table([7, 8], [6]))
    assert "**100.0%**" in md and "pass^10" in md


def test_explain_gap():
    assert explain_gap(0.9, 10) == ("At 90% per call, a supervisor reviewing 10 scored "
                                    "calls has a 65% chance of seeing at least one they "
                                    "would overturn.")


def test_evidence_is_refused_on_the_way_out(tmp_path):
    with pytest.raises(LeakError):
        write_json(tmp_path / "x.json", {"calls": [{"id": "c1", "evidence": "Hi Tom"}]})
    assert not (tmp_path / "x.json").exists()


def test_unmasked_numbers_are_refused_and_masked_ones_pass():
    with pytest.raises(LeakError):
        assert_no_transcript({"caller": "2065550150"})
    assert_no_transcript({"caller": mask_number("2065550150")})
    assert mask_number("2065550150") == "******0150"


def test_example_runs_end_to_end():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_path(str(ROOT / "example" / "demo.py"), run_name="__main__")
    out = buf.getvalue()
    assert exit_info.value.code == 0
    assert "suspect the rubric" in out
    assert "PROMOTE" in out
    assert "self-test PASSED" in out
    assert "one-sided criteria before: ['caller_name_used', 'compliance_recording_disclosure']  after: []" in out
