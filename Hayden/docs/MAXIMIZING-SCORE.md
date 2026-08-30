# Maximizing Technical Execution (35%) + Innovation (20%)

Together **55% of the grade**. Live plan: [[Refined-Strategy]] · Evidence: [[EDA-Findings]] · Spec: [[Docs]]

---

# PART 1 — Technical Execution (35%)

Two sub-components, and they reward different things.

## 1a. ⚠️ The convergence trap — the most important thing in this document

Re-read the rule carefully:

> *"A run is considered converged when validation score has not improved by more than ε = 0.002 over the last N = 3 consecutive iterations... The submission scored for ranking is the **validation-best checkpoint at that point**."*

**Three consecutive iterations without >0.002 total improvement ends the run.** Not the 50-iteration cap — that's a backstop that will almost never be reached. The real budget is: *you must keep gaining 0.002 every 3 iterations or you're done.*

Consequences most teams will miss:

1. **A run of three duds early = premature convergence at a near-baseline score.** If iterations 1–3 are all speculative and all fail, the run ends with the FM's own 0.6015 as "validation-best." Catastrophic.
2. **The 50-iteration cap is a red herring.** Realistic runs end at 8–20 iterations. Optimising for "how do I fill 50 iterations" is optimising the wrong thing.
3. **Iteration ordering is a first-class design decision**, not an afterthought. High-expected-value moves must come first — not because of the rule, but because a run of low-EV experiments is *doubly* punished here: wasted budget *and* triggered termination.

**Strategy:** order strictly by expected value × probability of success. Iteration 1 must be the highest-confidence idea we have (listwise loss — a principled argument, not a hunch). Never queue two speculative ideas back-to-back if a reliable one is available.

> **A note on gaming:** deliberately withholding a known-good change purely to pad the iteration count and dodge convergence would be gaming the rule, and I won't plan around that. Ordering by expected value is what a competent researcher does anyway — the rule just makes the cost of sloppy ordering explicit.

**Also note:** scoring uses the **converged result, not the peak**. So there's no credit for a lucky spike mid-run. Stability at the end matters more than a high-water mark.

## 1b. Robustness — the half most teams will neglect

> *"Not judged by whether the agent ever hits a failure, but by **how it handles one** — recovering, retrying, or routing around a failed step... so that long iterative runs neither crash, stall, nor diverge."*

This is explicitly graded, and it's an **engineering** criterion — winnable by building it properly, regardless of how the ML goes. Concretely:

**Isolation.** Each iteration's training runs in a **subprocess with a hard timeout**. A segfault, OOM, or infinite loop in one iteration must not kill the agent. The agent reads results from a file, never from in-process state.

**Checkpointing.** Persist after every iteration: current best config, best validation score, full history, RNG seeds. The run must be resumable from any crash — including a laptop reboot.

**An error taxonomy with a recovery ladder** — this is what "routing around it" means concretely:

| failure | detection | recovery |
|---|---|---|
| generated code won't parse/import | exception on import | feed traceback back to the LLM, regenerate (max 2 tries), else abandon this hypothesis and move on |
| OOM | `RuntimeError` / MPS alloc failure | halve batch size, retry once; then fall back to CPU |
| training timeout | subprocess wall-clock exceeded | kill, reduce epochs, retry once; else abandon |
| NaN / Inf loss | check each epoch | reload last checkpoint, lower LR ×0.1, retry |
| NaN/Inf in submission scores | validate before write | reject, fall back to previous best model |
| score wildly implausible (< random = 0.4834) | bounds check vs known rungs | treat as a bug not a result; re-run harness self-check |
| harness itself drifted | periodic `--model random` re-run, expect ≈0.4834 | halt and flag — never keep reporting numbers from a broken harness |

**Divergence guard.** "Neither crash, stall, **nor diverge**" — so also: never let a bad iteration overwrite the best checkpoint. Best-so-far is immutable; each iteration is a candidate that must *earn* promotion.

**Prove it works.** If nothing ever fails, there's nothing in the log to grade. Run a deliberate **fault-injection test** — inject a syntax error, force an OOM, kill a subprocess mid-train — and let the log capture the agent recovering. This is standard chaos-engineering practice, it's honest (clearly labelled as a fault-injection test, not a real failure), and it turns an unprovable claim into logged evidence.

**Log every failure.** The run-log spec asks for *"any error / recovery events."* Failed iterations are **evidence for this criterion, not embarrassments**. Hiding them loses points.

---

# PART 2 — Innovation & Problem Insight (20%)

> *"Judged on **what the agent identified as worth trying and why — not on implementation**."*
> *"Originality in drawing on published methods... rewarding agents that go **beyond naive baseline tweaks**."*

**The reasoning is the deliverable.** A correct implementation with no articulated rationale scores poorly here; a well-argued hypothesis that *fails* still scores. So every iteration must record **why**, and the why must be non-obvious.

"Naive baseline tweaks" = tune LR, change k, add a feature. The organizers already tested the last two and published that they don't work — so anything in that family is both worthless *and* visibly naive.

## The idea portfolio, ranked by (originality × expected gain)

### ① Shift-invariance → listwise loss ★ our strongest argument
Both metrics are invariant to adding a per-user constant. Pointwise logloss must model each user's base rate to produce calibrated probabilities — a quantity evaluation provably discards. Listwise softmax is exactly shift-invariant, so no capacity is spent on it.
*Why it scores:* derived from the metric definition, not guessed. It explains **why** the organizers' own top guess is right — which is a level above simply following their hint.

### ② Train/eval group-size mismatch ★★ original, straight out of our own EDA
Nobody has flagged this. Training groups average **43.5** rows/user; evaluation groups average **5.6 (valid) / 7.1 (test)**. Under a listwise objective, group size directly changes the loss landscape — a softmax over 43 candidates is a very different problem from one over 6.

It also explains the composition gap we measured: **92.7%** of *train* users are label-discriminative vs **63.7%** of *test* users, purely a consequence of group size (big groups are almost always mixed; small ones are often all-same by chance).

**Hypothesis:** subsample training groups to match the eval group-size distribution, so the training objective sees the same problem shape as evaluation.
*Why it scores:* it is a genuine train/serve skew discovered by measurement, invisible in the README, and the fix is principled rather than a hyperparameter sweep.

### ③ Metric-aligned hybrid loss ★★ directly targets the actual objective
`primary = mean(GAUC, nDCG@5)` — but these reward different things. GAUC is a *flat pairwise* metric over the whole list; nDCG@5 is *top-weighted* (log position discount). A single plain softmax optimises neither exactly.

**Hypothesis:** a composite objective — a pairwise term for GAUC plus a **LambdaRank-style position-weighted** term for nDCG (LambdaRank weights each pair by the nDCG change from swapping it, i.e. it optimises nDCG directly). Blend to match `mean(GAUC, nDCG@5)`.
*Why it scores:* optimising the actual scoring function rather than a proxy, with the blend justified by the metric's own definition. LightGBM's `lambdarank` (already installed) gives a free reference implementation to validate the idea before hand-rolling it.

### ④ Exposure-bias correction using the randomised log ★★★ most research-grade
KuaiRand's headline feature — the reason the dataset exists — is its **randomised-exposure log**, which enables counterfactual/off-policy evaluation. The standard log is what a *deployed recommender chose to show*, so it is biased by the old policy. Correcting for that (inverse propensity weighting) is a genuine research technique, and most teams will ignore this file entirely.

**Two variants, and the rules matter:**
- **(a) Validation only — safe, recommended.** Use `log_random` as an *unbiased validation set* to detect whether we're overfitting to biased traffic. Exactly what the organizers suggested. No rules risk.
- **(b) Propensity weighting in training — blocked pending your ruling.** `log_random` spans **4/22–5/08, overlapping the hidden test window**. See the open question in [[Refined-Strategy]].
- **(c) Rules-safe alternative:** estimate exposure propensity from the **training-period standard log itself** (item exposure frequency as a proxy). Less rigorous than a true randomised estimate, but zero rules risk and still a real debiasing argument.

*Why it scores:* uses the dataset's defining feature, cites a real literature (IPS / off-policy evaluation), and — importantly — **shows the agent reasoning about competition rules and choosing the conservative path**, which is itself evidence of judgment.

### ⑤ Censored watch-time regression (CWM) ★★ published method, real depth
A completed play means true watch time was **truncated** by video length — the observation is right-censored. Squared error on it is the wrong likelihood; a one-sided/censored loss is correct. Reference: Zhao et al., KDD 2024.
*Why it scores:* explicitly drawing on a published paper, reimplemented cleanly rather than depending on its `torch==1.6.0` repo.

### ⑥ Multi-task on click + watch-time ★
Standard technique — but our version has an evidence-driven twist: we measured all 8 auxiliary signals and **dropped 4** as too sparse (`is_follow`/`is_comment`/`is_forward`/`is_hate`, all ≤0.26% positive) despite the spec and README suggesting them.
*Why it scores:* the *selection* is the insight. Shows measurement driving design instead of following a checklist.

### ⑦ Rank-space ensembling ★ cheap insurance
Blend FM with LightGBM `lambdarank` — different inductive biases (embeddings vs trees). Average **ranks, not scores**, since only order matters and the two models' scores aren't on a common scale.
*Why it scores:* modest originality, but the rank-space detail shows metric awareness. Reliable end-game gain.

## Making the reasoning visible

Since reasoning is what's graded, each iteration's log entry must carry:
1. **Hypothesis** — the mechanism, not the tweak. *"Pointwise loss wastes capacity on per-user base rates the metric discards"*, not *"try listwise loss."*
2. **Prediction + falsifier** — what result would prove it wrong. Stating this up front is what separates a hypothesis from a guess.
3. **Provenance** — metric definition / our EDA / a specific paper.
4. **Outcome vs prediction** — and when they disagree, *say so*. A recorded surprise is stronger evidence of real inquiry than a string of confirmations.

**Precedent from our own work:** in [[EDA-Findings]] I retracted part of my own argument after the data contradicted it. That kind of visible self-correction is exactly what "problem insight" looks like — keep doing it in the run log rather than quietly editing history.

---

# PART 3 — How the two criteria interact

They pull in the same direction if the agent is built right, and against each other if it isn't:

- **Robustness protects the primary metric.** An unhandled crash at iteration 4 freezes your validation-best at whatever it was — likely near baseline. Every recovery path is directly defending the 35%.
- **The convergence rule punishes weak ideas twice.** Low-EV experiments waste budget *and* push toward premature termination. So Innovation quality feeds Technical Execution — good ideas aren't just worth more points, they keep the run alive.
- **Failed-but-well-reasoned iterations are cheap.** They score under Innovation (reasoning is what's judged) and under Robustness (recovery evidence). They only hurt via the convergence counter — which is another argument for interleaving reliable and speculative moves rather than batching the risky ones.

## Concrete iteration ordering

| # | idea | confidence | why here |
|---|---|---|---|
| 1 | Listwise loss (①) | **high** | strongest argument; must not open with a dud |
| 2 | Group-size matching (②) | medium-high | cheap, compounds with #1, original |
| 3 | Metric-aligned hybrid loss (③) | medium | builds on #1's machinery |
| 4 | Multi-task: click + watch-time (⑥) | medium | different axis — decorrelated from 1–3 |
| 5 | Censored watch-time (⑤) | medium-low | speculative, but only after gains are banked |
| 6 | DIN sequences | low | downgraded — median history 31 |
| 7 | Rank ensembling (⑦) + seed averaging | **high** | reliable end-game gain, held for the close |

Unbiased validation via `log_random` (④a) runs **alongside** throughout as a diagnostic, not as its own iteration.
