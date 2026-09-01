# Run Log

Per-iteration log — this is a **graded deliverable** (hypothesis / code diff / metrics / errors & recovery).
Plan: [[Refined-Strategy]] · Spec: [[Docs]]

---

## Iteration 0 — Environment setup & baseline reproduction

**Date:** 2026-08-29
**Hypothesis:** none — this is verification, not a modeling change. Goal: prove the harness is trustworthy before any later number can be believed.

### Environment
| | |
|---|---|
| Python | 3.12.14 (Homebrew), venv at `/Users/hayden/TechJamHackathon/.venv` |
| numpy | 2.5.2 |
| torch | 2.13.0 |
| MPS (Apple Silicon GPU) | **available** — verified with a matmul on `mps:0` |
| Data | `KuaiRand-Pure.tar.gz` (45M) from Zenodo, extracted to `./KuaiRand-Pure/data` (matches the kit's default `--data_dir`) |

System Python was 3.9.6 with no numpy; deliberately left untouched (macOS system Python is externally managed) in favour of a Homebrew 3.12 venv.

### Split sizes — match spec exactly
`train 1,141,112 / valid 124,909 / test 170,588` ✓ (spec: 1,141,112 / 124,909 / 170,588)

### Baseline reproduction

| model | valid primary | published valid | test primary | published test | verdict |
|---|---|---|---|---|---|
| random (seed 0) | 0.4827 | 0.4834 | **0.4757** | 0.4753 | ✓ within ±0.001 |
| item popularity | 0.5807 | 0.5807 | **0.5715** | 0.5715 | ✓ **exact** |
| **FM (target to beat)** | 0.6015 | 0.6016 | **0.5953** | 0.5946 (σ=0.0008) | ✓ within 1σ |

Full FM (seed 0): valid GAUC 0.6671 / nDCG@5 0.5358 · test GAUC 0.6621 / nDCG@5 0.5286.

**Harness self-check passed** — the README's stated gate is `--model random` → primary ≈ 0.475 ±0.001; we got 0.4757. Scoring code is trustworthy.

### Observations
- **FM runs in ~15s**, not the ~40s quoted (Apple Silicon, single core). Iteration cost is therefore very low — the 6h wall-clock ceiling and 50-iteration cap are unlikely to bind for FM-scale models. Compute is not the constraint; **idea quality is**.
- FM early-stops at **epoch 11** (patience 4, best at epoch 7, valid primary 0.6015). Validation primary peaks around epoch 7 then declines — the model starts overfitting quickly. Only ~7 useful epochs on 1.14M rows.
- `pop` reproducing *exactly* confirms deterministic data loading and ordering — important, since submission alignment depends on stable row order.

### Errors / recovery
- `wget` not present on the machine → used `curl -L` instead. No other failures.

---

## ⚠️ Methodological discipline going forward (important)

`baseline.py` prints **test** metrics on every run, and the starter kit ships test labels locally. But the challenge states the agent *"develops using only the training split and the public validation feedback — it never has access to the hidden test set,"* and the real ranking is scored once on the organizers' copy.

If we select changes by watching the local test number, we are effectively fitting to it, and the real hidden-test delta will come in **worse than measured** — plus it violates the stated development protocol.

**Rule adopted: all keep/revert decisions are made on `valid` primary only.** Local test is recorded for the record but never used as a selection signal, and is not to be consulted while choosing between candidate ideas.

---

---

## Iteration 0b — EDA

**Date:** 2026-08-29 · **Artifact:** `kuairand-starter-kit/eda.py` (read-only, ~6s runtime)
**Hypothesis:** none — measurement, to validate assumptions before committing to a phase order.
**Full writeup:** [[EDA-Findings]]

**Correctness check:** oracle ceiling reproduced exactly (valid 0.8484 / test 0.8645) as did the published test user composition (27.1/9.2/63.7%) — confirms `evaluate.py` is being driven correctly.

**Findings that changed the plan:**
1. README's "hundreds to thousands of interactions per user" is **wrong** — actual median 31, mean 43.5. → sequence/DIN phase downgraded; SIM dropped.
2. Group granularity **resolved without an ablation**: group by `(user)`. `(user,date)` is 28% singletons in train, median size 1 in eval. → one planned iteration saved.
3. **Retracted one of my own claims** — the "listwise auto-focuses on discriminative users" argument is negligible in training (92.7% of train users are already discriminative). Shift-invariance remains the real argument.
4. Multi-task narrowed from 6 auxiliary signals to **2** (`is_click` 46.3%, `play_time_ms` dense). `is_follow`/`is_comment`/`is_forward`/`is_hate` are 0.04–0.26% positive — too sparse to carry gradient.
5. Duration engineering **demoted** — coupling is non-monotonic and shallow (0.273–0.376).
6. Label rate drifts down train→valid→test (0.337→0.313→0.314), tab mix shifts too. → additional support for a shift-invariant objective.
7. Cold users: 3.3% of test users unseen in train (videos ~fully covered).

**Errors / recovery:** none.

**Revised phase order:** listwise loss → multi-task (click + watch-time) → DIN (modest) → censored watch-time → architecture swap.

---

---

## Iteration 0c — PyTorch port, parity PASSED

**Date:** 2026-08-30 · **Artifact:** `kuairand-starter-kit/fm_torch.py`
**Hypothesis:** none — infrastructure. A torch FM that does not reproduce the numpy baseline exactly would invalidate every later measurement.

Parity was achieved by matching the numpy version's details rather than approximating them:
- same init RNG (`np.random.default_rng(seed)`) → identical starting weights
- same shuffle RNG → identical batch composition and order
- **the bias uses plain SGD, not Adam** (`baseline.py:69` does `b -= lr * g.sum()`) — easy to miss, and it would have shown up as a small unexplained drift
- L2 added to the gradient before the Adam moments (== torch `weight_decay`)

**Result — exact to 4 decimals on all 6 metrics:**

| | torch | numpy | delta |
|---|---|---|---|
| valid primary | 0.6015 | 0.6015 | +0.0000 |
| test primary | 0.5953 | 0.5953 | +0.0000 |

Also **1.6× faster** (7.9s vs 12.9s). PARITY PASSED — safe to build on.

---

## Iteration 0d — Agent built

**Date:** 2026-08-30 · **Artifacts:** `agent/`

| file | role |
|---|---|
| `llm.py` | Gemini client, retries, **token metering from call 1** (Feasibility deliverable) |
| `prep.py` | data cache — **train+valid only, test physically excluded** |
| `worker.py` | subprocess entry point; validates score alignment and finiteness |
| `runner.py` | isolated execution + failure taxonomy + recovery ladder |
| `diagnose.py` | rich feedback: metrics, overfitting, segments, health, regressions |
| `context.py` | priors and memory → prompt |
| `guards.py` | `evaluate.py` checksum + test-leak + sanity bounds |
| `agent.py` | the loop |
| `seed_solution.py` | official FM in the agent's contract (verified 0.6015) |

**LLM:** the provided key is **Google Gemini**, not Anthropic (the `AQ.` prefix gave it away; Anthropic keys are `sk-ant-`). Pro tier returns `429 RESOURCE_EXHAUSTED` — quota-limited. **`gemini-3.6-flash` works** (~2s, a thinking model).

**Two real bugs found and fixed during setup** (both logged as robustness evidence):
1. *Retrying empty responses.* Gemini 3.x are thinking models; with a small `max_output_tokens` the thinking budget consumes the whole allowance and `resp.text` is empty. The client treated that as transient and retried 4× with backoff — a 6-minute hang across a model sweep. Empty-response is now a distinct non-retryable error that tells the caller to raise the token budget.
2. *No HTTP timeout.* A quota-blocked model hung indefinitely instead of erroring. The client now sets a hard request timeout.

**Fault-injection test — all five failure classes correctly caught and routed:**

| injected fault | classified as | recovery offered |
|---|---|---|
| syntax error | `syntax` | regenerate the module |
| wrong output length | `alignment` | scores must be 1:1 with valid rows |
| NaN in scores | `numerical` | lower LR, clamp logits |
| infinite loop | `timeout` | reduce epochs — baseline is 8s |
| bad import | `import` | only installed libs available |

**Seed verified end-to-end through the isolated runner:** 7.5s, 11 epochs, valid primary **0.6015** — matches the official baseline exactly.

**Diagnostics sanity check:** segment analysis returns `all_neg: 0.2500` and `all_pos: 0.7500` — exactly the theoretical values (all-negative users score GAUC 0.5 by convention and nDCG 0 → 0.25; all-positive → 0.75). Confirms the segmentation is correct.

**Design commitment — priors, not answers.** `context.py` gives the agent the *fact* that both metrics are invariant to adding a per-user constant, plus every ruled-out idea and every EDA measurement. It deliberately does **not** say "use a listwise loss." If the agent derives that itself, it is the agent's insight and counts under Innovation; hardcoding it would make the autonomy claim fiction.

**Errors / recovery:** two client bugs (above), both fixed. Bash quoting issues while probing models — resolved by writing the Python client instead of fighting `curl`.

---

## RUN 1 — first autonomous agent run (3 iterations)

**Date:** 2026-08-30 · **Artifacts:** `agent/state_run1/`
**Result: baseline beaten with zero human intervention.**

| | |
|---|---|
| iterations | 3 |
| best valid primary | **0.6035** (baseline 0.6015, **+0.0020**) |
| **manual interventions** | **0** |
| wall clock | 2.7 min |
| LLM tokens | 29,019 across 3 calls (~10K/iteration) |
| accepted changes | 1 / 3 |
| converged | no — hit the iteration cap at stall 2/3 |

### Iteration 1 — ✅ KEPT, 0.6015 → 0.60346 (+0.0020)

**The agent derived the core insight unprompted.** Its own words:

> *"Both GAUC and nDCG@5 depend exclusively on the relative ordering of items within each individual user's group, meaning adding any per-user constant c_u leaves validation scores unchanged. Pointwise BCE penalizes global log-odds miscalibrations across users, wasting capacity on fitting inter-user variations."*

`context.py` states only the mathematical *fact* (both metrics are invariant to per-user constants). It does not mention loss functions, pointwise, pairwise, or listwise. The conclusion is the agent's. It chose **pairwise BPR**; our own analysis had favoured listwise softmax first.

**Prediction vs outcome — the interesting part.** It predicted 0.615–0.625 and wrote its own falsifier: *"if primary is below 0.6035, the hypothesis is falsified."* It scored **0.60346** — fractionally under its own threshold, and ~8× below its predicted gain.

So the mechanism is real (the gain exceeds 2σ) but far weaker than the clean theoretical argument implies. A genuine result: objective/metric mismatch matters here, but it is not where the headroom lives.

### Iterations 2 & 3 — ❌ both reverted, both hyperparameter tweaks
- iter 2: weight decay 1e-6 → 1e-3 → **0.5990 (−0.0044)**
- iter 3: `samples_per_pos` 2 → 8 → **0.6023 (−0.0012)**

### Root-cause analysis (our design failure, not the model's)

After one genuinely novel change, the agent **collapsed into tuning constants of its own idea**. Two causes, both ours:

1. **The diagnostics handed out prescriptions.** `diagnose.py` emitted *"Consider stronger regularisation, fewer epochs, or more data per step."* The agent read that and turned exactly the named knob. We built `context.py` on "facts, never conclusions" and then violated that principle in the feedback layer.
2. **Nothing marked hyperparameter tuning as low-value.** The ruled-out list covered the organizers' dead ends, not "don't fiddle with what you just wrote."

**Fixes applied before run 2:**
- `diagnose.py` now reports the observation only (*"validation peaked at epoch 7 of 11 while training loss continued to fall"*) with no suggested remedy.
- `context.py` carries the lesson forward, citing run 1's own data: *"Constants are not mechanisms... propose a DIFFERENT MECHANISM instead."* — cross-run learning, not just within-run memory.

### Scorecard against the judging criteria

| criterion | evidence |
|---|---|
| Autonomy (20%) | **0 interventions** — fully autonomous |
| Technical — primary metric | beat baseline (+0.0020), though only 0.8% of headroom |
| Technical — robustness | 5/5 injected faults classified and recovered; 0 crashes |
| Innovation (20%) | insight independently derived; falsifier stated in advance and honoured |
| Feasibility (15%) | 29K tokens, 2.7 min — comfortably the "low consumption" tier |

**Weakest area: the score delta.** +0.0020 is real but small. Run 2 (12 iterations, fixed priors) targets exactly that.

---

## RUNS 2 & 3 — and two harness bugs that were suppressing the score

All runs: **0 manual interventions**, 0 crashes.

| run | winning mechanism | best valid | delta | iters used | why it ended |
|---|---|---|---|---|---|
| 1 | intra-user pairwise BPR | 0.6035 | +0.0020 | 3 / 3 | iteration cap |
| 2 | intra-user pairwise BPR | 0.6033 | +0.0018 | **3 / 12** | ❌ buggy convergence |
| 3 | **DeepFM** (MLP branch) | 0.6032 | +0.0017 | **4 / 14** | ✅ correct convergence |

### 🐞 Bug 1 — convergence rule implemented per-iteration instead of per-window

The spec says converged when the score *"has not improved by more than ε = 0.002 **over the last N = 3 consecutive iterations**"* (Chinese README: 连续 3 轮迭代…提升不超过 0.002). That is a **cumulative window**. Our code tested each iteration individually, which is strictly harsher — three gains of 0.001 total 0.003 and are *not* converged, but the old code declared them so.

Cost: run 2 ended at iteration **3 of 12**. Found by re-reading the spec, not by anything failing loudly — the kind of bug that quietly costs most of the Technical Execution score. Fixed and unit-tested against five cases.

### 🐞 Bug 2 — a hard 2σ keep-threshold discarding real gains

The identical BPR mechanism measured **+0.0020 / +0.0018 / +0.0016** across three runs. The keep bar sat at exactly 2σ = 0.0016, *inside* that noise band — so the same mechanism was accepted or rejected by luck. Worse, a rejection was logged as a failed idea, steering the agent away from the one direction that reproducibly works. In run 3 this cost two iterations re-deriving a gain it had already found in iteration 1.

Fixed: a marginal positive delta now triggers two extra seeds (~10s each) and is kept on the **mean**.

### 🔬 An independent confirmation of the core theory

Run 3, iteration 4: the agent added **per-user empirical log-odds as fixed logit offsets**. Result: **−0.0001** — exactly nothing. That is precisely what shift-invariance predicts (a per-user constant cannot change within-user ordering). The agent unknowingly ran a clean control experiment for its own iteration-1 hypothesis.

### 🧠 Structural problem identified: gains must stack, and nothing was stacking

The convergence rule demands **>0.002 cumulative gain per 3 iterations**. Every single mechanism found so far is worth **~0.002 alone**. So one idea per run is never enough to survive — improvements must accumulate.

But state reset between runs, so verified discoveries died with each run. Run 1 found BPR. Run 3 found DeepFM. **Nobody ever tried them together.**

Fixed via `agent/findings.py`: measured results and open questions now persist across runs. The combination is presented as *unmeasured*, not as an instruction — the agent is told the arithmetic ("every mechanism is worth ~0.002, the threshold is 0.002") and left to draw the conclusion.

---

## Next

RUN 4 in progress — first run with seed confirmation and cross-run findings.
