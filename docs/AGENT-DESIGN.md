# Agent Design — proposal

Plan: [[Refined-Strategy]] · Scoring: [[Maximizing-Score]] · Evidence: [[EDA-Findings]]
Status: **proposal, not built.**

---

## The core loop

```python
state = load_checkpoint() or init()

while not converged(state) and within_budget(state):
    context   = build_context(state)        # priors + memory + last diagnostics
    proposal  = ask_llm(context)            # hypothesis + reasoning + new code
    result    = run_isolated(proposal)      # subprocess + timeout + recovery
    diag      = diagnose(result)            # rich feedback  <-- key differentiator
    state     = decide(state, result, diag) # keep/revert on VALID only
    log(proposal, result, diag, state)      # emits the graded run-log fields
    save_checkpoint(state)                  # resumable after any crash

finalize(state)   # validation-best -> submission -> submit.py --check
```

## Files

| file | job |
|---|---|
| `agent.py` | the loop above |
| `context.py` | assembles what the LLM sees each turn |
| `runner.py` | isolated execution + the recovery ladder |
| `diagnose.py` | turns a run into rich feedback |
| `decide.py` | keep/revert, noise threshold, convergence |
| `guards.py` | anti-self-deception checks |
| `state.json` / `history.jsonl` | checkpoint + append-only memory |

---

## 1. What the LLM sees each turn (`context.py`)

Six blocks, in this order:

1. **Task** — within-user ranking, `long_view`, GAUC + nDCG@5, primary = mean. Splits. Crucially: **what the metric is invariant to** (adding any per-user constant changes nothing).
2. **Reference points** — random 0.4834 · pop 0.5807 · FM baseline **0.6015** · oracle **0.8484** (all valid). Per-seed σ = 0.0008.
3. **Priors — what's already ruled out** (so it doesn't burn iterations rediscovering):
   - larger embedding dim k (organizers: 8/16/32 → flat)
   - more static features (organizers: 5→13 fields → flat/slightly worse)
   - `(user,date)` grouping (our EDA: 28% singletons in train, median size 1 in eval)
   - `is_follow`/`is_comment`/`is_forward`/`is_hate` as aux tasks (≤0.26% positive)
   - SIM long-sequence retrieval (median history is 31, not hundreds)
4. **Current best** — the winning code, its score, and *why it won*.
5. **Memory** — every prior attempt: hypothesis → result → why we think it failed.
6. **Rules** — never modify `evaluate.py`; never look at test; one change per iteration; state a falsifier.

> **Design principle:** give it *facts and constraints*, never the answer. If it derives listwise loss from the invariance fact, that's the agent's insight and scores under Innovation. If we hardcode "use listwise loss," it's ours and the autonomy claim is fiction.

## 2. Rich feedback (`diagnose.py`) — the cheapest big win

Most teams will return one scalar. Return this instead:

**Headline** — valid GAUC / nDCG@5 / primary, delta vs current best, delta vs baseline.

**Training dynamics** — best epoch vs total epochs, train loss trend, gap between train and valid.
→ auto-hint: *"valid peaked at epoch 3/40 while train loss kept falling → overfitting; consider regularisation or fewer epochs."*

**Segment breakdown** — score by user history length (short/medium/long), by group size, cold vs warm users, by `tab`.
→ auto-hint: *"worst segment is cold users (3.3% of rows) — no personalisation signal available there."*

**Prediction health** — score distribution, variance, fraction of tied scores, any degenerate constant output.
→ catches the classic "model collapsed to a constant but the metric looks okay-ish" failure.

**Regression analysis vs previous best** — which user segments improved, which got worse.
→ *"gained on long-history users, lost on short — the change trades one population for another."*

Same LLM, far better hypotheses. This is where most of the Innovation score is actually made.

## 3. Robustness (`runner.py`) — worth a chunk of the 35%

Every run is a **subprocess with a hard timeout**. The agent reads a result file; it never shares memory with training code.

| failure | recovery |
|---|---|
| code won't parse / import | feed traceback back, regenerate (max 2), else abandon hypothesis |
| OOM | halve batch, retry once → fall back to CPU |
| timeout | kill, reduce epochs, retry once → abandon |
| NaN / Inf loss | reload checkpoint, LR × 0.1, retry |
| NaN / Inf in submission scores | reject before writing, keep previous best |
| score below random floor (0.4834) | treat as a **bug**, not a result — rerun harness self-check |

**Divergence guard:** best-so-far is immutable. Each iteration is a *candidate* that must earn promotion.

**Fault-injection test:** deliberately inject a syntax error, force an OOM, kill a subprocess mid-train — so the log contains real recovery evidence. Clearly labelled as a test.

## 4. Anti-self-deception (`guards.py`)

The signature failure of autonomous ML agents is optimising hard against their own bug.

- **Checksum `evaluate.py` every iteration** — if the hash changes, halt. The agent must never "improve" the scorer.
- **Test split is never loaded** during the run. Not read, not scored, not shown.
- **Implausible score → bug hypothesis first.** Any jump > ~0.03 triggers a re-run and a leakage check before it's believed.
- **Periodic harness self-check** — re-run `--model random`, expect ≈0.4834.
- **Multi-seed confirmation** before believing a delta under ~0.002 (σ = 0.0008).

## 5. Decision policy (`decide.py`)

- Compare on **valid only**.
- Improvement must clear noise; if marginal, re-run across seeds before promoting.
- Track the convergence counter (**ε = 0.002, N = 3**) — and log a warning as it approaches 2, since that's a run-ending condition.
- Budget: 50 iterations / 6h.

## 6. Built-in accounting (don't retrofit)

The Feasibility criterion (15%) **requires reporting** total input+output tokens, wall-clock, and iterations used. Meter these from iteration 1 — reconstructing them afterwards is painful and error-prone.

Also auto-count **manual interventions** (any human input mid-run), since that's the headline Autonomy number.

## 7. Search strategy

**Start greedy-with-backtrack:** expand from current best; on failure, revert and try a different direction.

**Upgrade if time allows:** keep the top-K solutions and occasionally expand from the runner-up when the best stalls — cheap tree-search flavour, closer to AIDE (which the spec cites), more robust against the convergence trap, and stronger on Innovation.

---

## Build order

1. `runner.py` + `decide.py` + checkpointing → a loop that survives crashes *(robustness = 35%)*
2. `diagnose.py` → rich feedback *(cheapest quality multiplier)*
3. `context.py` → priors + memory
4. `guards.py` → anti-self-deception
5. Accounting + logging in the graded format
6. Fault-injection test
7. *(optional)* top-K search

**1–3 are the minimum viable agent.** 4–6 are what turn a working agent into a well-scoring one.

## Open decision

Still needs Hayden: build this as a program (recommended), or keep iterating interactively and accept the autonomy hit? See the fork in [[Maximizing-Score]] discussion.
