# Critical Review — judging our own submission

Written adversarially: what would a judge attack? Scored against the published
criteria in [[Docs]]. Evidence in [[Run-Log]].

---

## 🔴 SEVERE — We handed the agent its only winning idea

**The finding.** `agent/findings.py`, in the list I wrote and injected into the
agent's context:

> *"No run has tried averaging the SAME model across several random seeds and
> rank-averaging the results, which reduces variance rather than bias."*

The agent's next run implemented seed ensembling. **The one mechanism out of eleven
that worked was suggested to it by us.**

This directly damages two criteria at once:
- **Innovation (20%)** is judged on *"what the agent identified as worth trying and why"*. It didn't identify this. We did.
- **Autonomy (20%)** — steering the agent to the answer is intervention, whatever the per-run counter says.

**The honest mitigation** (not a defence, a nuance): the agent chose this from four
open questions, and it *deviated* from our suggestion — we proposed rank-averaging,
it implemented logit-averaging. That deviation turned out to be correct: when a later
iteration tried rank-normalising before averaging it scored **−0.0182**, the worst
result of the project. So the selection and the implementation were the agent's; the
direction was ours.

**Fix before submission:** remove the ensembling hint from `findings.py` and do a
clean run. If the agent finds it unaided, the claim is real. If it doesn't, we report
that honestly and the result stands as a human-suggested, agent-implemented finding.

## 🔴 SEVERE — Our "0 interventions" claim is framed favourably

We report 0 manual interventions, which is true *within* a run. Across the project we
modified core agent logic at least eight times between runs:

| # | modification | run |
|---|---|---|
| 1 | removed prescriptive hints from `diagnose.py` | after 1 |
| 2 | added anti-hyperparameter-tuning prior | after 1 |
| 3 | rewrote the convergence rule (per-iteration → window) | after 3 |
| 4 | replaced the 2σ keep-threshold with seed confirmation | after 4 |
| 5 | added cross-run `findings.py` | after 4 |
| 6 | added auxiliary signals to the data cache | after 5 |
| 7 | added model rotation, then key rotation | after 5, 6 |
| 8 | added behaviour-history features | after 6 |

The spec's own definition: *"Only modifications to the core agent logic count as
manual intervention."* By that definition our count is ~8, not 0. The spec does allow
*"a handful of interventions"* as realistic — eight is arguably beyond a handful.

**Fix:** designate ONE run as the official run, execute it start to finish with zero
modifications, and report that. Present the earlier runs as development, which is what
they were.

## 🟠 SIGNIFICANT — The agent uses 8% of its iteration budget

Every run converged at **3–4 iterations** out of 50 allowed. A judge will reasonably
ask why an autonomous research agent runs out of ideas after four attempts.

Two causes, and only one is the benchmark's fault:
- The convergence rule is genuinely tight (>0.002 per 3 iterations against ~0.001 available).
- **But our agent is a greedy single-chain searcher.** It always improves the current
  best and never explores alternatives in parallel.

## 🟠 SIGNIFICANT — We identified the fix and did not build it

We surveyed AIDE (cited in the challenge's own references) and extracted its policy:
draft several *diverse* solutions first, then improve the best. Measured effect on
MLE-bench: **8.7% vs 4.4%** medal rate against linear agents — and the paper names our
exact symptom, that linear agents *"terminate early."*

We wrote this up and then didn't implement it. That is the single highest-value
unbuilt improvement, and it is squarely within scope (public prior work is explicitly
permitted).

## 🟡 MODERATE — The submitted score is a fortunate draw

The seed sweep measured the expected gain from ensembling at **+0.0010** (asymptote,
20 models). Our submission measured **+0.0012 validation / +0.0021 test**.

The submission is legitimate — real mechanism, correctly measured, honestly validated.
But the effect size we should *report* is +0.0010; +0.0021 is the upper end of the
distribution, not the expectation. Claiming +0.0021 as the method's effect would be
overclaiming.

## 🟡 MODERATE — Best-of-seven run selection

We ran seven times and are submitting the best. Selecting across runs on validation is
a mild form of human-guided search, and it inflates the apparent result the same way
single-seed measurement did. Disclose it.

---

## What holds up well

- **Robustness is genuinely strong.** Eight-class failure taxonomy with recovery
  ladders, subprocess isolation, five deliberately injected faults all correctly
  classified, plus real unplanned recoveries (a sampling bug repaired mid-run that
  then produced a kept improvement). Zero crashes across seven runs.
- **Measurement discipline caught our own false positives.** Runs 1–3 each reported
  ~+0.002; multi-seed replication reduced the same code to +0.0005. Without that check
  we would have submitted a phantom improvement. Very few teams will do this, and
  fewer will report it.
- **Eleven mechanisms measured with mechanistic explanations**, not just a results
  table — including a control experiment the agent ran unknowingly (per-user offsets →
  −0.0001, exactly the zero the invariance property predicts).
- **Anti-self-deception is structural, not aspirational.** `evaluate.py` is
  SHA-256 checksummed each iteration; the test split is physically absent from the
  data cache rather than merely forbidden.
- **External validation.** Published work (Crocodile) reports GAUC 0.662 on KuaiRand;
  the provided baseline is 0.661. The baseline is already at the level of published
  research, which supports the near-ceiling conclusion rather than leaving it as an
  excuse.
- **Cheap.** 46K tokens, 4 LLM calls, 5 minutes per converged run — comfortably the
  low-consumption tier, and it only counts because we cleared the baseline gate.

## Ranked fixes, by score impact per hour of work

| # | fix | criterion | effort |
|---|---|---|---|
| 1 | One clean untouched run, reported as THE run | Autonomy 20% | 10 min |
| 2 | Remove the ensembling hint, see if the agent finds it unaided | Innovation 20% | 10 min |
| 3 | AIDE-style drafting phase (draft K diverse solutions, then improve best) | Technical 35% | ~1 hour |
| 4 | Report +0.0010 as the effect size, disclose best-of-7 selection | credibility | 5 min |

**1, 2 and 4 are nearly free and fix the two most damaging criticisms.** 3 is the real
engineering work and the only one likely to move the score itself.
