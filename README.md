# call-qa-judge

A small harness for scoring recorded calls with an LLM judge, and for finding
out whether the judge deserves to be trusted before anyone coaches an agent
off its scores.

Stdlib only. Tests and the example run offline in about a second against a
rule-based stand-in for the model. An Anthropic-backed judge is included; it
is stub-tested only, since the tests and the example never call the API.

## The problem

Call-QA products score every call against a rubric and put the result on a
leaderboard. Supervisors coach from it. Nobody checks whether the judge agrees
with a person, so the leaderboard is a confident number pointing in an
unverified direction.

I built this for a call center customer whose calls were already being scored
by a judge I'd written. The problem showed up while I was building the
harness, before any human had graded a single call. There were 115 scored
transcripts with speaker labels. In nine of them nobody was on the other end
of the line, and the judge had scored every one. The worst was 334 seconds of
a bank's phone menu and hold music. The agent dialed out and waited. The
judge gave every criterion a 0, so the call came out at 0.50 overall, the
worst call in that customer's corpus and the first thing a supervisor would
see. It also got 10 out of 10 on recording disclosure, because the bank's menu
said the call may be recorded.

Both fixes are in here. The judge can now decline a criterion when the call
gives it nothing to grade, which is a legal answer and gets recorded as
absent. And recording disclosure moved out of the judge entirely, into a
check that knows whose voice said it.

## What's in it

| file | what it does |
|---|---|
| `callqa/rubrics/default.json` | The rubric as data. Twelve criteria, each a label, a relative weight and the guidance text the judge reads. A supervisor can edit it without touching code. |
| `callqa/judge.py` | `ModelJudge` takes any `complete(system, user, schema)` function, so the model call is injected. Each criterion comes back with `gradeable: true/false`, and a decline is recorded as absent, never as zero. `AnthropicJudge` wraps the Messages API with a JSON-schema constrained response. |
| `callqa/checks.py` | Deterministic checks that return pass, fail or unknown: is this a two-sided conversation, did our side disclose recording, did the agent give a name, was the company named in the greeting. Unknown never counts as a pass. |
| `callqa/calibrate.py` | Judge against a blind human pass, per call and per criterion, with the direction of every miss. Three or more misses on one criterion all leaning one way flags the rubric and quotes the guidance to edit. Human "ungradable" verdicts the judge scored anyway are counted as overreach, separately. |
| `callqa/kappa.py` | Quadratic-weighted Cohen's kappa beside the raw agreement rate. |
| `callqa/passk.py` | pass@k and pass^k over scored calls, with and without the checks. |
| `callqa/gate.py` | A rubric change only ships if it beats the incumbent on mean agreement with frozen human bands and on pass^k, over repeated noisy replays. Ties lose. A deliberately broken rubric must be rejected and score materially worse, or the gate's own self-test fails. |
| `callqa/outcome.py` | The one check that leaves the transcript. For calls where the judge credited "asked for the commitment", it asks the phone system's call ledger whether that number was on a call with us again inside the window. |
| `callqa/leak.py` | Refuses to write transcript text or an unmasked phone number into any artifact. |
| `docs/PREREGISTRATION.md` | How to seal judge hypotheses before the human pass, plus the four that were sealed in production. |

## Run it

```
python -m pytest
python example/demo.py
```

The demo runs on eleven made-up calls for a made-up garage door company. The
stand-in judge starts with both production defects switched on, so a
phone-menu call and a voicemail land at the bottom of the worst-calls list.
The calibration report then flags `caller_name_used` as one-sided harsh and
recording disclosure as one-sided lenient. After the fixes, nothing is
one-sided. Then the gate promotes a guidance edit and rejects a broken
rubric, and the outcome check runs against a fake call ledger.

## Using it on real calls

```python
from callqa import AnthropicJudge, calibrate, load_calls, load_human, load_rubric
from callqa.calibrate import render_markdown

rubric = load_rubric("my_rubric.json")
judge = AnthropicJudge(model="<model id>")
calls = load_calls("calls.json")            # speaker-labeled turns
verdicts = [judge.score(c, rubric) for c in calls]

human = load_human("human_verdicts.json")   # graded blind, before reading anything
print(render_markdown(calibrate(verdicts, human, rubric), rubric))
```

A few things that cost me time and might save you some.

Grade blind. The human must not see the judge's scores, and should not read
the preregistered hypotheses until their verdicts are saved. A verdict given
with the judge's number on screen is an upper bound, and the report labels it
that way.

Read the gap before the kappa. A judge that's a constant one point harsh scores
kappa 0.86, three points harsh scores 0.40, and both are one guidance edit
away from fixed. Kappa measures distance, so it punishes an easy fix almost as
hard as noise.

Tolerances get fixed before the human pass: 2 points per criterion on the
0-10 scale and 1.5 on the overall. Choosing them after you've seen the
results defeats the purpose.

The pass threshold is yours to pick. Where a good call sits on 0-10 is a
question for whoever runs the floor, so `passk_table` sweeps a range and
prints what each bar implies.

On the outcome check, the first version in production reported 100 percent
confirmation. On an outbound call the originating number is our own agent's
line, so it was really asking "did this agent make another call." The
counterparty now comes from the ledger and the call direction, and any number
that originates a lot of calls is treated as ours and refused.

## Related

[grade-the-grader](https://github.com/johngilligan2/grade-the-grader) is the
general method this started from, written up for a helpdesk triage judge.
This repo is the call-center version. What's new here is the editable
rubric with per-criterion kappa, and the parts that look past the judge:
checks that know who was speaking and the outcome check against the call
ledger.

## License

MIT.
