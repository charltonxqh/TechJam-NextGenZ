# TechJam-NextGenZ

An autonomous ML research agent for the TikTok TechJam 2026 recommendation
challenge, built on the KuaiRand-Pure benchmark.

## What it does

The agent is handed the official FM baseline and nothing else. From there it
runs on its own: it questions the data, reads papers, writes a full solution,
trains it, scores it on validation, and decides whether to keep it. Then it does
it again, building on whichever attempt looks most promising. It stops when it
stops improving.

The target is `long_view`. Ranking quality is measured inside each user's own
impressions:

- **GAUC**
- **nDCG@5**
- **Primary = mean(GAUC, nDCG@5)**

## The loop

```text
                    ┌──────────────────────────────┐
                    │  Draft phase (iterations 1-3)│
                    │  three independent solutions │
                    │  from scratch, not edits     │
                    └───────────────┬──────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────┐
       │  Pick a node from the solution tree (UCB)           │
       │  not always the best one — sometimes a near-miss    │
       └───────────────────────┬────────────────────────────┘
                               │
                               ▼
       ┌────────────────────────────────────────────────────┐
       │  Investigate: query the data, search arXiv,         │
       │  recall what past runs already measured             │
       └───────────────────────┬────────────────────────────┘
                               │
                               ▼
       ┌────────────────────────────────────────────────────┐
       │  Propose a hypothesis and write the code for it     │
       └───────────────────────┬────────────────────────────┘
                               │
                               ▼
       ┌────────────────────────────────────────────────────┐
       │  Run it in a subprocess, score on validation        │
       │  crash → diagnose → repair, up to 3 tries           │
       └───────────────────────┬────────────────────────────┘
                               │
                               ▼
       ┌────────────────────────────────────────────────────┐
       │  Keep or reject. Either way the node joins the tree │
       │  and the result goes into memory                    │
       └───────────────────────┬────────────────────────────┘
                               │
                    improving? │
              ┌────────────────┴─────────────────┐
              │ yes                           no │
              ▼                                  ▼
        back to the tree              one-time test evaluation
```

### Why a tree

Most agent loops keep one current-best solution and edit it. That does not work
here, and we measured why.

The FM baseline is already a tuned optimum for the five pre-encoded fields it is
given. On that same feature set, nothing stronger beats it — CatBoost QueryRMSE
scored 0.5956, CatBoost YetiRank 0.5944, LightGBM lambdarank 0.5994, all under
the FM's 0.6015. The only thing that clears it is changing the features *and*
the model at once (0.6039).

So there is no single small edit that improves anything. An agent that only ever
applies one change to its incumbent is stuck by construction, and ours was:
twelve iterations, zero accepted.

The fix was to open with three independent drafts, each choosing features and
model together, and keep them all alive in a tree. The next node to expand is
picked by UCB, so a slightly worse branch still gets explored instead of being
discarded. After that change the same budget produced accepted solutions and
moved validation from 0.6015 to 0.6051.

### What the agent can actually do

Eight tools. It chooses which to call and when.

| Tool | Purpose |
|---|---|
| `list_columns` | Every column in every CSV, flagged for whether it is legal as a model input |
| `inspect` | Ask questions of the data — group sizes, label rates, cold-start overlap, cardinality |
| `build_features` | Build a design matrix from any columns it picks, plus derived fields, video metadata and a dense block over the 51 video-statistics columns |
| `search_papers` | Search arXiv before proposing a mechanism |
| `train_model` | Fit a gradient-boosting model, score it, cache the predictions |
| `blend` | Rank-average cached predictions |
| `list_predictions` | Everything trained so far this run, with scores |
| `recall` | Long-term memory: what previous runs already established |

### Guards

An autonomous agent's worst failure is optimising against its own bug and
reporting a beautiful number that means nothing. Two things make that
structurally impossible rather than merely discouraged:

- **The test set is unreachable.** Every read of the raw logs goes through one
  function, and that function drops test dates. The agent cannot load them even
  if it asks. Test data is materialised once, after the run is over, by a
  separate script.
- **The scorer is checksummed.** `evaluate.py` is hashed at startup and compared
  against a recorded value. If the file has changed, the run refuses to start.

Post-exposure columns from the same impression — `is_click`, `is_like`,
`is_comment`, `is_follow`, `is_forward`, `is_hate`, watch time — are marked as
outcomes and blocked as model inputs. Using them is trivially easy and produces
GAUC around 0.87 on validation, which is why the block is enforced in code
rather than left to the model's judgment.

### Memory

Results persist across runs, not just within one. Each attempt records its
hypothesis, the code, the score, and whether it was accepted. The `recall` tool
searches that history, so the agent does not spend an iteration re-measuring
something a previous run already settled.

---

## Repository Layout

```text
main.py                     alternative linear loop (see below)
src/
├── engine/                 the agent described above
│   ├── agent.py            the loop, drafting phase, convergence rule
│   ├── tree.py             solution tree, UCB selection
│   ├── tools.py            the eight tools; the single choke point that hides test
│   ├── context.py          prompt construction
│   ├── models.py           CatBoost / LightGBM / XGBoost wrappers
│   ├── guards.py           checksum and sanity bounds
│   ├── memory.py           cross-run memory
│   ├── diagnose.py         per-segment error analysis fed back to the agent
│   ├── runner.py           subprocess isolation
│   ├── llm.py              Gemini client with model rotation
│   ├── finalize.py         one-time test evaluation
│   └── evidence/           scripts reproducing the measurements quoted here
├── agents/                 orchestrator, researcher, policy
├── research_intelligence/  web + arXiv research, EDA, on-demand skills
├── memory/                 experiment memory and compressed context
├── tools/                  runner, validators, metrics
├── config.py  schemas.py
kuairand-starter-kit/       the starter kit and KuaiRand-Pure data, unmodified
deliverables/submitted-run/ run log, usage, tree and scores for the submitted run
```

`src/engine/` produced the results reported below.

The repo also contains a second, linear loop under `src/agents/` and
`src/research_intelligence/`, reachable via `main.py`. It adds capabilities the
engine does not have — live web research and on-demand skill loading — and is
the natural place to fold the tree search into next.

---

## Setup

### 1. Create a virtual environment

```bash
# macOS / Linux
python3 -m venv venv && source venv/bin/activate

# Windows PowerShell
python -m venv venv; .\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Gemini access, structured validation, web retrieval, HTML parsing and numerical
ML: Google GenAI, Pydantic, DDGS, Requests, BeautifulSoup, NumPy, python-dotenv.

### 3. Add your API key

Create `.env` in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Never commit `.env`.

### 4. Data

The KuaiRand-Pure CSVs belong in `kuairand-starter-kit/KuaiRand-Pure/data`,
which is where extracting the kit puts them. That is also the default, so a
standard checkout needs no configuration. Expected files:

```text
log_standard_4_08_to_4_21_pure.csv
log_standard_4_22_to_5_08_pure.csv
user_features_pure.csv
video_features_basic_pure.csv
video_features_statistic_pure.csv
```

Set `DATA_DIR` in `.env` only if they live somewhere else.

---

## Running it

```bash
python src/engine/agent.py
```

The run starts by declaring its stopping rule, then reproduces the FM baseline:

```text
convergence_rule_declared  epsilon=0.002  N=3  min_iterations=12
seed solution: valid primary 0.6015
DRAFT 1/3 — independent solution, features and model chosen together
```

Do not pick experiments for it mid-run. The agent uses validation to decide what
becomes the new best, and repairs its own crashes inside the same iteration.
The final model is the **validation-best checkpoint**, not the last iteration.

Test evaluation happens once, after the loop stops:

```bash
python src/engine/finalize.py
```

### Run artefacts

State is written to `src/engine/state/` as the run proceeds, so a killed run
resumes:

- `run_log.jsonl` — the stopping rule, then one record per iteration
- `state.json` — full loop state, including every attempt
- `tree.json` — the solution tree
- `memory.json`, `findings.json` — what carries into later runs
- `usage.json` — tokens, calls, wall-clock, iterations
- `best_solution.py` — the winning code

`deliverables/submitted-run/` holds these files for the submitted run.

---

## Results

KuaiRand-Pure. Raw artefacts in `deliverables/submitted-run/`.

| | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| FM baseline — validation<sup>†</sup> | 0.6674 | 0.5357 | 0.6016 |
| **Validation-best** | **0.6723** | **0.5379** | **0.6051** |
| FM baseline — test<sup>†</sup> | 0.6610 | 0.5282 | 0.5946 |
| **Final test (one-time)** | **0.6657** | **0.5304** | **0.5981** |

<sup>†</sup> Baseline figures quoted from
`kuairand-starter-kit/baseline_scores.json`, not re-measured by us.

**Absolute delta over the baseline**, scoring formula
`½[(GAUC_agent − GAUC_base) + (nDCG_agent − nDCG_base)]` on test:

```text
½[(0.665747 − 0.6610) + (0.530438 − 0.5282)]  =  +0.003493
```

GAUC **+0.0047**, nDCG@5 **+0.0022**.

For scale: the baseline's own standard deviation over five seeds is 0.0008 on
each test metric, so this is roughly four to six sigma. A perfect model scores
0.8645 on test, not 1.0 — nDCG@5 cannot exceed 0.7289 because 27.1% of test
users have no positive impression at all and score zero no matter what.

Submission file: `deliverables/submitted-run/submission_draft.csv`, 170,588 rows,
`row_id,user_id,video_id,score`. Check it with:

```bash
python kuairand-starter-kit/submit.py --check deliverables/submitted-run/submission_draft.csv
```

**Bonus benchmarks** (KuaiRand-1k, KuaiRand-27k) were **not attempted**.

### Resource usage

From the agent's own accounting in `deliverables/submitted-run/usage.json`.

| | |
|---|---:|
| Total tokens (input + output) | **418,825** |
| — input | 402,598 |
| — output | 16,227 |
| — served from cache | 106,251 |
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

---

## Limitations

**The feature representation is the ceiling, and we only partly moved it.** The
diagnostic that shaped this design showed no model beats the FM on the five
given fields. Three drafts found better feature sets, but the search over
feature space is still shallow — it is the highest-value direction left.

**One seed.** The reported run is a single seed. With noise at 0.0008 per metric
the gain is well clear of it, but a multi-seed ensemble would be both stronger
and better evidenced.

**No live web research in the loop that produced these numbers.** The engine can
search arXiv, and did so once during the submitted run. The richer research
stack under `src/research_intelligence/` is not wired into it yet.

**Free-tier quota shapes the run.** The LLM rotates across five Gemini models
because each has a small daily limit. Runs can end on quota rather than on
convergence, which is a practical constraint, not a scientific one.

**Repair is shallow.** Crashes get three attempts with shrinking timeouts. A
genuinely subtle bug in generated code will usually just cost the iteration.

---

## Benchmark integrity

The target is `long_view`, scored by ranking within each user's impressions.

Legal:

```text
training-time is_click  →  auxiliary loss  →  shared representation
                                                     ↓
                            validation/test prediction from legal inputs only
```

Not legal:

```text
validation/test is_click  →  input feature  →  predict long_view, same impression
```

The distinction is enforced in `src/engine/tools.py`, not left to judgment. The
same rule governs history features: an aggregate over a user's *earlier* rows is
fine, but it must not be built from labels in the evaluation period.

The goal is a score that comes from a reproducible, honest search — not just a
high number.

---

## Team

**Hayden** — the research engine in `src/engine/`: tool layer, solution tree and
UCB selection, drafting phase, leakage guards, cross-run memory, diagnostics,
and the measurements the design is based on. Produced the submitted result.

**Charlton** — agent architecture and orchestration, researcher/coder
separation, online research integration, multi-hypothesis generation and
ranking, research-integrity controls, system integration.

**David** — co-developed the orchestration and research tooling, the autonomous
research workflow and experiment-loop design; integration and debugging.

**Esther** — the memory component: experiment records, compressed context for
later researcher calls, persistence of hypotheses, code changes, metrics,
diagnostics and recovery information.

---

## Acknowledgements

Built for TikTok TechJam 2026 on the provided KuaiRand starter kit and the
KuaiRand-Pure benchmark. The drafting policy follows AIDE (arXiv:2502.13138),
whose stated approach is to "first explore a set of diverse initial solutions
and continuously improve the best one" — we had implemented only the second
half, and measuring that gap is what produced the result above.
