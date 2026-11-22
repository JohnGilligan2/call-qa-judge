# Preregistering judge hypotheses before the human pass

The calibration method here turns on one claim: the direction of the judge's
misses points at a defect in the rubric. That claim has a known failure mode,
which is hindsight. Once you can see which way the misses lean, it is easy to
write a plausible story for any guidance string, and the story feels like a
finding because it fits.

So the predictions go on the record first. Write them down, date them, and
seal them before a single human verdict exists. A prediction that survives a
blind human pass is evidence. One that does not gets deleted, not reworded.

## The rules

1. Seal the file before the human pass starts, and put the date at the top.
   The grader must not read it until their blind verdicts are saved. Reading
   it first makes the grading worse and the result worth less.
2. Derive hypotheses from score distributions only: per-criterion means,
   spreads, how a criterion moves (or does not move) across call types.
   If you read a transcript to form one, say which and why.
3. Every hypothesis has four parts: the prediction, the evidence that
   produced it, why it matters if true, and a falsifier. The falsifier is the
   human result that would prove you wrong. No falsifier, no hypothesis.
4. Amendments are allowed before the human pass, as new dated sections.
   Never fold a late idea into an earlier hypothesis. The timestamp is the
   only thing that makes a preregistration worth anything.
5. Keep the hypotheses the data killed before the human pass. Write down why.
   Otherwise nothing stops one coming back later dressed as a finding.
6. Do not predict the overall agreement rate. Stating a number in advance
   invites grading toward it.

## Template

```
## H<n>. <one-line claim>

Prediction. What the human pass will show, specific enough to fail.
Evidence. The distribution, count or call that produced it, with the date.
Why it matters. What goes wrong downstream if it is true.
Falsifier. The human result that kills it.
```

## The four sealed in production

These were written for a call center customer's corpus before any human
grading, from score distributions plus the one transcript named in H1. Names,
numbers and call identifiers are left out here. They are the reason the
checks and the decline path in this repo exist.

### H1. The judge scores ungradable calls zero instead of declining

Prediction: at least one sampled call has no two-sided conversation in it,
and the judge scored every criterion 0 instead of declining.

Evidence: one call was 334 seconds of a bank's phone menu and hold loop. The
agent dialed out and nobody ever came on the line. Every substantive criterion
scored 0, overall 0.50, the worst call in that customer's corpus.

Why it matters: the worst score sorts to the top of a supervisor's review
list, drags the team average down, and blames an agent who did nothing wrong.
Zero claims the agent performed terribly. The true answer is that the call
cannot be graded, and the judge had no way to say so.

Falsifier: the human scores the call low on the merits without reaching for
"ungradable".

### H2. "Used the caller's name" is systematically harsh

Prediction: this criterion shows the largest one-directional gap of any
criterion, judge below human, and the misses do not scatter.

Evidence: its mean was the lowest of any criterion in both scored cohorts and
nearly identical across customer, vendor and internal calls. A criterion that
scores the same whatever kind of call it is looking at is not measuring the
agent.

Suspected cause: the guidance said "score low if the name was given and never
used again", which sets the bar at repeated use. On a short or transactional
call, using a name once is normal and using it over and over is odd.

Falsifier: the human also scores it low. Then agents really are not using
names, and the rubric is reporting a real coaching point.

### H3. Call-role mismatch is NOT the main defect

Prediction: vendor and internal calls do not show a materially larger
judge-harsh gap than customer calls.

Evidence: this began as the leading hypothesis, since a sales-and-service
rubric was scoring support and vendor calls. The distributions did not back
it. Vendor calls scored slightly higher on average than customer calls.

Why it is written down: a hypothesis the data killed before the human pass is
worth as much as one it survived, and recording it stops it being quietly
reintroduced later.

Falsifier: vendor and internal calls diverge from customer calls by more than
the per-criterion tolerance.

### H4. Recording disclosure is systematically LENIENT, and credits the far end

Added as a dated amendment while building the deterministic checks, still
before any human verdict.

Prediction: where judge and human disagree on this criterion, the judge is
the lenient one, the opposite direction to H2, and the calls are outbound.

Evidence: disclosure language was spoken by a system voice on outbound calls
17 times against 3 on inbound. Outbound means our agent dialed someone, so
that system voice is the other company's phone menu. The H1 call scored 10
out of 10 on disclosure because the bank's menu said the call may be
recorded.

Why it matters: a false pass on the one criterion with legal weight, and an
invisible one. It makes the dashboard greener, so nobody goes looking.

Falsifier: the human also credits these calls, on the grounds that a
disclosure both parties heard counts however it got there.

Two traps rule out the naive regex fix: "monitor" is not a disclosure word
when someone is talking about a screen, and voicemail greetings say "record
your message". Both are handled in `callqa/checks.py` and both have tests.

## Where each hypothesis lands in this code

| hypothesis | code |
|---|---|
| H1 | `judge.py` lets each criterion decline; `checks.gradeable_conversation`; `calibrate` reports overreach separately |
| H2 | `calibrate` flags one-sided criteria and quotes the guidance; `gate` tests a rewording against frozen human bands |
| H3 | nothing to build, which is the point of writing it down |
| H4 | `checks.recording_disclosure` attributes by speaker and direction, and returns unknown for a far-end menu |
