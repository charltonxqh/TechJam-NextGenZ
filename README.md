# TechJam-NextGenZ

An autonomous ML research agent for the TikTok TechJam 2026 recommendation
challenge, built on the KuaiRand-Pure benchmark.

## Project overview

The agent is handed the official FM baseline and nothing else. From there it
runs on its own: it questions the data, reads papers, writes a complete
solution, trains it, scores it on validation, and decides whether to keep it.
Then it repeats, building on whichever attempt looks most promising, and stops
when it stops improving.

The target is `long_view`, scored by ranking quality inside each user's own
impressions: **GAUC**, **nDCG@5**, and **Primary = mean(GAUC, nDCG@5)**.

Two things distinguish it:

**It searches a tree, not a line.** Most agent loops keep one best solution and
edit it. That cannot work here, and we measured why: the FM baseline is already
a tuned optimum for the five pre-encoded fields it is given, and on those same
fields nothing stronger beats it — CatBoost 0.5956, LightGBM lambdarank 0.5994,
all below the FM's 0.6015. Only changing features *and* model together clears it.
An agent restricted to one edit at a time is stuck by construction, and ours was:
twelve iterations, zero accepted. The fix was to open with three independent
drafts and keep them all alive in a tree, selecting the next node to expand by
UCB. The same budget then produced accepted solutions and moved validation from
0.6015 to 0.6051.

**It cannot cheat, structurally.** Every read of the raw logs passes through one
function that drops test dates, so the agent cannot load the test set even if it
asks. The scorer is checksummed at startup and the run refuses to begin if it
has changed. Post-exposure columns from the same impression — `is_click`,
`is_like`, watch time and the rest — are marked as outcomes and blocked as model
inputs; using them is easy and yields GAUC near 0.87, which is exactly why the
block is enforced in code rather than left to the model's judgment.

```text
src/engine/     the agent: loop, solution tree, tools, guards, memory
src/            a second, linear loop (orchestrator, research intelligence)
kuairand-starter-kit/       starter kit and KuaiRand-Pure data, unmodified
deliverables/submitted-run/ artefacts for the submitted run
```

## Setup and installation

**1. Virtual environment**

```bash
# macOS / Linux
python3 -m venv venv && source venv/bin/activate

# Windows PowerShell
python -m venv venv; .\venv\Scripts\Activate.ps1
```

**2. Dependencies**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**3. API key** — create `.env` in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

**4. Data** — the KuaiRand-Pure CSVs belong in
`kuairand-starter-kit/KuaiRand-Pure/data`, which is where extracting the kit
puts them and is also the default, so a standard checkout needs no
configuration. Expected files:

```text
log_standard_4_08_to_4_21_pure.csv
log_standard_4_22_to_5_08_pure.csv
user_features_pure.csv
video_features_basic_pure.csv
video_features_statistic_pure.csv
```

Set `DATA_DIR` in `.env` only if they live elsewhere.

## Steps to reproduce the results

**1. Run the agent**

```bash
python src/engine/agent.py
```

It declares its stopping rule, reproduces the FM baseline, then begins:

```text
convergence_rule_declared  epsilon=0.002  N=3  min_iterations=12
seed solution: valid primary 0.6015
DRAFT 1/3 — independent solution, features and model chosen together
```

Do not pick experiments for it mid-run. It decides what becomes the new best
from validation alone, and repairs its own crashes inside the same iteration.
The final model is the **validation-best checkpoint**, not the last iteration.
State is written to `src/engine/state/` as it goes, so a killed run resumes.

**2. Evaluate on test, once, after the loop stops**

```bash
python src/engine/finalize.py
```

**3. Check the submission**

```bash
python kuairand-starter-kit/submit.py --check deliverables/submitted-run/submission_draft.csv
```

### Results

KuaiRand-Pure. Artefacts in `deliverables/submitted-run/`.

| | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| FM baseline — validation<sup>†</sup> | 0.6674 | 0.5357 | 0.6016 |
| **Validation-best** | **0.6723** | **0.5379** | **0.6051** |
| FM baseline — test<sup>†</sup> | 0.6610 | 0.5282 | 0.5946 |
| **Final test (one-time)** | **0.6657** | **0.5304** | **0.5981** |

<sup>†</sup> Quoted from `kuairand-starter-kit/baseline_scores.json`, not
re-measured by us.

Absolute delta over the baseline, `½[(GAUC_agent − GAUC_base) + (nDCG_agent − nDCG_base)]`
on test:

```text
½[(0.665747 − 0.6610) + (0.530438 − 0.5282)]  =  +0.003493
```

GAUC **+0.0047**, nDCG@5 **+0.0022**. The baseline's own standard deviation over
five seeds is 0.0008 per test metric, so this is roughly four to six sigma. Note
that a perfect model scores 0.8645 on test, not 1.0 — nDCG@5 cannot exceed
0.7289 because 27.1% of test users have no positive impression at all.

Bonus benchmarks (KuaiRand-1k, KuaiRand-27k) were **not attempted**.

### Resource usage

| | |
|---|---:|
| Total tokens (input + output) | **418,825** |
| — input / output | 402,598 / 16,227 |
| LLM calls | 67 |
| Wall-clock | **120.6 min** |
| Iterations used (cap 50) | **12** |
| GPU-hours | **0** — CPU only |
| Manual interventions | 0 |

The stopping rule was declared before the run started — epsilon 0.002, N 3,
minimum 12 iterations, self-imposed caps of 40 iterations and 300 minutes, both
inside the 50-iteration and 6-hour limits. It is the first line of
`run_log.jsonl`. Crashed iterations count toward the caps but do not advance the
convergence window, so a run cannot stop early by failing.

## Limitations and what we would improve

**The feature representation is the ceiling, and we only partly moved it.** The
diagnostic that shaped this design showed no model beats the FM on the five
given fields. Three drafts found better feature sets, but the search over
feature space stayed shallow — with more time this is where we would spend it,
since it is the one axis measurement says is binding.

**One seed.** The reported run is a single seed. At 0.0008 noise the gain is
clear of it, but a multi-seed ensemble would be both stronger and better
evidenced.

**The agent is optimistic about its own ideas.** It commits to a numeric
prediction before each experiment, which let us score it: mean absolute error
0.0015, but six of seven predictions were overestimates. Useful for ranking
candidates, not trustworthy as a substitute for running them — which is why
accept/reject stays on measured validation.

**Crashes cost five of twelve iterations.** Each got three attempts with
shrinking timeouts, but a subtle bug in generated code usually just burns the
iteration. Stronger schema-aware constraints and generated unit tests would
recover most of that budget.

**Research and search are not yet joined up.** The engine can query arXiv, and
did so once in the submitted run, but the richer research stack under
`src/research_intelligence/` — live web retrieval, on-demand skills — is not
wired into the tree search. Folding the two together is the clearest next step,
and the two loops already share a starter kit, data path and metrics.

**Free-tier quota shapes the run.** The client rotates across five Gemini models
because each has a small daily limit; runs can end on quota rather than on
convergence. A practical constraint, not a scientific one.

## Team member contributions

**Hayden** — the research engine in `src/engine/`: tool layer, solution tree and
UCB selection, drafting phase, leakage guards, cross-run memory, diagnostics,
and the measurements the design rests on. Produced the submitted result.

**Charlton** — agent architecture and orchestration, researcher/coder
separation, online research integration, multi-hypothesis generation and
ranking, research-integrity controls, system integration.

**David** — co-developed the orchestration and research tooling, the autonomous
research workflow and experiment-loop design; integration and debugging.

**Esther** — the memory component: experiment records, compressed context for
later researcher calls, and persistence of hypotheses, code changes, metrics,
diagnostics and recovery information.
