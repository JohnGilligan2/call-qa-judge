"""The whole method on synthetic calls, offline, in a few seconds.

    python example/demo.py

Every call, name and number here is made up. The judge is the rule-based
stand-in from callqa/stub.py, which can reproduce the two judge defects the
harness caught in production. Nothing calls a model or the network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from callqa import (ModelJudge, calibrate, check_outcomes, checks_for_rubric,  # noqa: E402
                    evaluate, gate, golden_from_human, load_calls, load_human,
                    load_ledger, load_rubric, rule_based_model, run_checks,
                    self_test)
from callqa.calibrate import render_markdown  # noqa: E402
from callqa.checks import PASS  # noqa: E402
from callqa.kappa import format_kappa  # noqa: E402
from callqa.outcome import render_markdown as render_outcome  # noqa: E402
from callqa.passk import explain_gap, passk_table_with_checks  # noqa: E402
from callqa.passk import render_markdown as render_passk  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
NEW_NAME_GUIDANCE = ("Did the agent learn the other party's name and use it at "
                     "least once? Using it once is enough; repeating it is not "
                     "required and on a short call can sound forced.")


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def summary_line(cal) -> str:
    return (f"call-level {cal.call_agreements}/{len(cal.calls)}, per-criterion "
            f"{cal.crit_agreements}/{cal.crit_n}, kappa {format_kappa(cal.pooled_kappa)} "
            f"over {cal.pooled_n} pairs, overreach {len(cal.overreach)}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    org_names = json.loads((DATA / "calls.json").read_text(encoding="utf-8"))["org_names"]
    calls = load_calls(DATA / "calls.json")
    human = load_human(DATA / "human_verdicts.json")
    ledger = load_ledger(DATA / "ledger.json")
    v1 = load_rubric()

    section("1. The judge as found: scores every call, credits any recording message")
    prod = ModelJudge(rule_based_model(org_names), name="judge-as-found")
    prod_verdicts = [prod.score(c, v1) for c in calls]
    ranked = sorted(prod_verdicts, key=lambda v: v.overall(v1.weights) or 0.0)
    print("Worst five calls by the judge's overall score:")
    for v in ranked[:5]:
        call = next(c for c in calls if c.call_id == v.call_id)
        gc = run_checks(checks_for_rubric(v1.keys, org_names), call)[0]
        print(f"  {v.call_id}  overall {v.overall(v1.weights):5.2f}  "
              f"disclosure {v.scores['compliance_recording_disclosure']:>2}  "
              f"gradeable: {gc.outcome}")

    section("2. Deterministic checks (pass / fail / unknown)")
    checks = checks_for_rubric(v1.keys, org_names)
    for c in calls:
        res = run_checks(checks, c)
        print(f"  {c.call_id} {c.direction:8} " + "  ".join(f"{r.name}={r.outcome}" for r in res))

    section("3. Calibrate against a blind human pass")
    cal1 = calibrate(prod_verdicts, human, v1)
    print(render_markdown(cal1, v1))

    section("4. Fix: let the judge decline, demote disclosure to the check, edit one guidance")
    v2 = v1.with_guidance("caller_name_used", NEW_NAME_GUIDANCE, "call-qa-v2")
    print(f"caller_name_used guidance, {v1.version}:\n  {v1.guidance('caller_name_used')}")
    print(f"caller_name_used guidance, {v2.version}:\n  {v2.guidance('caller_name_used')}")
    fixed = ModelJudge(rule_based_model(org_names, no_decline=False, credit_far_end=False),
                       name="judge-fixed")
    cal2 = calibrate([fixed.score(c, v2) for c in calls], human, v2, reuse_human_pass=True)
    print()
    print(f"before: {summary_line(cal1)}")
    print(f"after:  {summary_line(cal2)}")
    print(f"one-sided criteria before: {[r.key for r in cal1.suspect]}  after: {[r.key for r in cal2.suspect]}")

    section("5. Gate the rubric change: frozen golden bands, 5 replays, noisy judge")
    golden = golden_from_human(calls, human)

    def noisy():
        return ModelJudge(rule_based_model(org_names, no_decline=False,
                                           credit_far_end=False, noise_seed=7),
                          name="judge-noisy")
    inc = evaluate("incumbent", golden, noisy(), v1, replays=5)
    cand = evaluate("candidate", golden, noisy(), v2, replays=5)
    print(gate(inc, cand, k=5).to_text())
    ok, broken = self_test(golden, noisy(), v1, replays=5)
    print()
    print("Self-test with a deliberately broken rubric (two guidance strings emptied):")
    print(broken.to_text())
    print(f"  self-test {'PASSED' if ok else 'FAILED'}: rejected and scored materially worse")

    section("6. pass^k on the fixed judge, with and without the checks")
    v2_verdicts = {c.call_id: fixed.score(c, v2) for c in calls}
    check_ok = {c.call_id: all(r.outcome == PASS for r in run_checks(checks_for_rubric(v2.keys, org_names), c))
                for c in calls}
    ids = [c.call_id for c in calls]
    score_of = lambda cid: v2_verdicts[cid].overall(v2.weights)  # noqa: E731
    plain = passk_table_with_checks(ids, [6, 7], score_of, lambda cid: True)
    checked = passk_table_with_checks(ids, [6, 7], score_of, lambda cid: check_ok[cid])
    print(render_passk(plain, title="judge score only"))
    print(render_passk(checked, title="judge score AND every deterministic check"))
    print(explain_gap(plain[0].per_trial))

    section("7. Outcome check: did anything follow the commitments the judge credited?")
    rows = check_outcomes(calls, list(v2_verdicts.values()), ledger, days=7)
    print(render_outcome(rows, days=7))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
