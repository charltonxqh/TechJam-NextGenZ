# TechJam-NextGenZ

Autonomous ML Research Agent for the TikTok TechJam 2026 recommendation-system challenge.

## Project Overview

TechJam-NextGenZ is an autonomous machine learning research agent designed to improve a recommender-system pipeline on the KuaiRand-Pure benchmark.

The challenge requires the agent to reproduce the official baseline, autonomously investigate promising improvements, modify the ML pipeline, run controlled experiments, evaluate validation performance, and iterate until convergence. The benchmark uses `long_view` as the relevance target and evaluates ranking quality within each user's logged impressions using:

- **GAUC**
- **nDCG@5**
- **Primary = mean(GAUC, nDCG@5)**

Our system is designed around an evidence-driven research loop rather than simple hyperparameter search.

```text
Official baseline
      ↓
Research / targeted EDA / on-demand skills
      ↓
Research knowledge + experiment memory
      ↓
Generate multiple candidate hypotheses
      ↓
Rank hypotheses
      ↓
Select one controlled experiment
      ↓
Coder implements the experiment
      ↓
Candidate validation
      ↓
Research-integrity / leakage validation
      ↓
Implementation-fidelity verification
      ↓
Run validation experiment
      ↓
Keep / reject
      ↓
Update memory and research direction
      ↓
Repeat until convergence
      ↓
One-time final test evaluation
```

### Key Components

#### Autonomous Researcher
The Researcher decides whether the next useful action should be online research, targeted EDA, loading an on-demand skill, or running an experiment. Research is evidence-driven: additional searches must address a concrete unresolved knowledge gap rather than simply consuming a research budget.

Before implementation, the Researcher generates multiple falsifiable hypotheses and ranks them using evidence support, metric alignment, dataset fit, information gain, feasibility, novelty, leakage safety, and compute efficiency.

#### Online Research Intelligence
The agent can search the public web and arXiv. Retrieved sources are filtered for relevance before structured evidence is extracted and stored in research memory.

#### Autonomous EDA
EDA is performed by deterministic tools. Completed analyses are tracked so the agent does not repeatedly run the same EDA without gaining new information.

#### On-Demand Skills
The agent first sees lightweight skill metadata. Full skill content is loaded only when useful for the current reasoning step, reducing unnecessary prompt size and token usage.

#### Researcher–Coder Separation
The Researcher decides **what should be tested and why**. A separate Coder translates the selected experiment specification into runnable code.

#### Research Integrity
The system includes deterministic checks to prevent prediction-time leakage. Same-impression post-exposure behavior such as `is_click`, `is_like`, `is_comment`, `is_follow`, `is_forward`, `is_hate`, and watch-time outcomes must not be used as validation/test input features for predicting `long_view` on the same impression.

Training-only auxiliary objectives and causally prior historical aggregates may still be valid.

#### Experiment Memory
Each experiment stores its hypothesis, rationale, research direction, code diff, metrics, decision, diagnostics, recovery events, and relevant resource usage so future research steps can learn from earlier experiments.

---

## Repository Layout

```text
main.py                     entry point for the autonomous research loop
src/
├── agents/                 orchestrator, researcher, policy, decision
├── research_intelligence/  online research, EDA, on-demand skills, knowledge store
├── tools/                  experiment runner, candidate + integrity validators, metrics
├── memory/                 experiment memory and compressed context
├── engine/                 tree-search research engine (second track, see below)
│   ├── evidence/           standalone scripts reproducing the reported measurements
│   └── state/              live run state (git-ignored)
├── config.py               paths, loop settings, model selection
└── schemas.py              typed records exchanged between components
kuairand-starter-kit/       organiser-provided kit and KuaiRand-Pure data (unmodified)
deliverables/submitted-run/ run log, usage, tree and scores for the submitted run
```

### Two research engines

The repository contains two implementations of the research loop, developed in
parallel during the hackathon.

`src/` is the system this README describes: a linear, evidence-driven loop with
online research, on-demand skills, hypothesis ranking and integrity validation.

`src/engine/` is a second engine built around a **solution tree** — it keeps
several promising branches alive at once and selects the next node to expand
with UCB, rather than advancing a single line of work. It is the engine that
produced the numbers recorded in `deliverables/submitted-run/`. Run it with:

```bash
python src/engine/agent.py
```

The two share the same starter kit, the same data directory, and the same
`long_view` target and metrics, so their results are directly comparable. The
README's *Limitations* section notes that the `src/` search policy "selects one
hypothesis at a time" and that a stronger version "could maintain a true
research tree" — `src/engine/` is an implementation of exactly that idea, and
folding it into the main loop is the clearest next step for this codebase.

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd TechJam-NextGenZ
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

The project uses dependencies for Gemini access, structured validation, web retrieval, HTML parsing, and numerical ML experimentation, including Google GenAI, Pydantic, DDGS, Requests, BeautifulSoup, NumPy, and python-dotenv.

### 5. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Do not commit `.env` or API credentials to GitHub.

### 6. Prepare KuaiRand-Pure

Place the required KuaiRand-Pure files under the directory configured by `DATA_DIR` in `src/config.py`.

Expected files include:

```text
log_standard_4_08_to_4_21_pure.csv
log_standard_4_22_to_5_08_pure.csv
user_features_pure.csv
video_features_basic_pure.csv
video_features_statistic_pure.csv
```

Make sure `DATA_DIR` points to the directory containing these files.

`DATA_DIR` defaults to `kuairand-starter-kit/KuaiRand-Pure/data`, which is where
the extracted kit puts them, so no configuration is needed for a standard
checkout. Set `DATA_DIR` in `.env` only if the data lives elsewhere.

---

## Steps to Reproduce the Results

### 1. Activate the environment

```bash
# Windows
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 2. Run the autonomous research agent

From the repository root:

```bash
python main.py
```

A successful run begins by reproducing the official FM baseline:

```text
Starting autonomous ML research session...
Baseline established: Validation Primary = 0.6015
```

The agent then autonomously performs research, EDA, skill loading, hypothesis generation/ranking, coding, validation, training, evaluation, and memory updates.

### 3. Allow the loop to converge

Do not manually select experiments during the run. The agent uses validation results to decide whether a candidate becomes the new best solution. Technical failures may be repaired automatically within the same scientific iteration.

The final model is the **validation-best checkpoint**, not necessarily the final iteration.

### 4. Final test evaluation

After convergence, the system performs a one-time final test evaluation using the validation-best experiment.

Typical output:

```text
Research session finished.
Best experiment: <experiment_id>
Best Validation Primary: <score>

Running final one-time test evaluation...
Final Test GAUC: <score>
Final Test nDCG@5: <score>
Final Test Primary: <score>
```

### 5. Inspect run artifacts

Each run produces research and experiment logs under `runs/`.

Typical artifacts include:

```text
runs/
├── run_<timestamp>/
│   ├── run_log.jsonl
│   ├── run_log_full.json
│   ├── run_log.md
│   ├── research_knowledge.jsonl
│   ├── research_trace.jsonl
│   ├── skill_trace.jsonl
│   └── debug/
```

Key files:

- `run_log.jsonl` — append-only per-iteration experiment records.
- `run_log_full.json` — full exported run history.
- `run_log.md` — human-readable run summary.
- `research_knowledge.jsonl` — EDA findings and external research evidence.
- `research_trace.jsonl` — online-search, source-selection, extraction, and storage trace.
- `skill_trace.jsonl` — records which skills were loaded and injected.
- `debug/` — generated candidates, validation failures, verification reports, and repair artifacts.

### Result Reporting

Results below are from the submitted autonomous run, not an intermediate
development experiment. Raw artefacts are in `deliverables/submitted-run/`.

**KuaiRand-Pure — required benchmark**

| | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Official FM baseline — validation<sup>†</sup> | 0.6674 | 0.5357 | 0.6016 |
| **Validation-best autonomous result** | **0.6723** | **0.5379** | **0.6051** |
| Official FM baseline — test<sup>†</sup> | 0.6610 | 0.5282 | 0.5946 |
| **Final one-time test result** | **0.6657** | **0.5304** | **0.5981** |

<sup>†</sup> Baseline figures are the organisers' own, quoted from
`kuairand-starter-kit/baseline_scores.json`, not re-measured by us.

**Absolute delta over the official baseline** (Judging Criteria formula,
`½[(GAUC_agent − GAUC_base) + (nDCG_agent − nDCG_base)]`, on the test split):

```text
½[(0.665747 − 0.6610) + (0.530438 − 0.5282)]  =  +0.003493
```

Per-metric: GAUC **+0.0047**, nDCG@5 **+0.0022**. For scale, the organisers
report a 5-seed baseline standard deviation of 0.0008 on each test metric, and
an oracle ceiling of 0.8645 test primary — nDCG@5 cannot exceed 0.7289 because
27.1% of test users are all-negative.

The submission file is `deliverables/submitted-run/submission_draft.csv`
(170,588 rows, `row_id,user_id,video_id,score`). Verify it with:

```bash
python kuairand-starter-kit/submit.py --check deliverables/submitted-run/submission_draft.csv
```

**Bonus benchmarks** (KuaiRand-1k, KuaiRand-27k) were **not attempted**; no
outputs are submitted for them.

### Resource Usage

Measured for the converged run, from the agent's own accounting in
`deliverables/submitted-run/usage.json` and `run_log.jsonl`.

| | |
|---|---:|
| Total token consumption (input + output) | **418,825** |
| — input / prompt tokens | 402,598 |
| — output tokens | 16,227 |
| — of which served from cache | 106,251 |
| LLM calls | 67 |
| Total agent wall-clock | **120.6 min** (≈2.0 h) |
| Iterations used (cap 50) | **12** |
| GPU-hours | **0** — CPU only, no GPU used |
| Manual interventions | 0 |

The run declared its convergence rule before starting (`epsilon` 0.002, `N` 3,
minimum 12 iterations, self-imposed caps of 40 iterations / 300 min, all within
the organisers' 50-iteration / 6-hour limits). The declaration is the first
record in `run_log.jsonl`.

---

## Limitations and Future Improvements

The current system demonstrates an end-to-end autonomous ML research loop, but several limitations remain.

**Retrieval quality is imperfect.** Search engines may return irrelevant or inaccessible sources. Relevance filtering helps, but future work could improve query reformulation, domain-aware retrieval, GitHub/code search, PDF parsing, and source-quality scoring.

**Hypothesis ranking remains partly LLM-dependent.** Although candidates are scored on explicit criteria, the criterion values are still model judgments. With more time, we would calibrate ranking using historical experiment outcomes.

**EDA coverage can be expanded.** Additional tools for feature-target relationships, hard-negative structure, temporal drift, popularity bias, and historical-signal construction could make hypotheses more dataset-specific.

**The search policy is still relatively shallow.** The current loop selects one hypothesis at a time. A stronger version could maintain a true research tree, preserve multiple promising branches, and allocate experiment budget using a more advanced explore/exploit strategy.

**Cross-session memory is still early-stage.** Future versions could better separate general research knowledge from benchmark-specific findings, detect contradictions, consolidate duplicate sources, and learn from experiment outcomes across runs.

**Generated code can still fail.** LLM-generated candidates may contain incorrect assumptions about fields, APIs, or the existing baseline implementation. More unit-test generation, schema-aware constraints, and stronger sandbox checks would improve reliability.

**Model/API availability affects runtime.** Rate limits, model capacity, and network latency can slow autonomous runs. Retry logic, fallback models, checkpoint/resume, and lightweight models for extraction/verification are natural improvements.

**Compute-aware research can be improved.** A future version could allocate an explicit token/compute budget across research, coding, and experiments and optimize expected score improvement per unit cost.

---

## Team Member Contributions

### Charlton
- Agent architecture and orchestration.
- Autonomous research workflow and Researcher–Coder separation.
- Online research integration and research-action routing.
- Multi-hypothesis generation and ranking workflow.
- Research-integrity / leakage controls.
- Integration of team components into the end-to-end system.

### David
- Co-developed the agent/orchestration and research-tool components.
- Contributed to the autonomous research workflow and experiment-loop design.
- Supported integration and debugging of the end-to-end system.

### Hayden
- Developed early MLE / research-intelligence work that informed the current system.
- Contributed procedural skills, EDA/recommender-system intelligence, and supporting tools.
- Contributed to implementation and evaluation utilities during development.

### Esther
- Developed the memory component for persistent experiment history.
- Designed experiment records and compressed context for later Researcher calls.
- Supported persistence of hypotheses, code changes, metrics, diagnostics, and recovery information.

---

## Benchmark Integrity

The target is `long_view`, and evaluation is based on ranking within each user's logged impressions.

A valid auxiliary-training setup may look like:

```text
training is_click
    ↓
auxiliary training loss
    ↓
shared representation
    ↓
validation/test prediction using legal prediction-time inputs
```

An invalid setup is:

```text
validation/test is_click
    ↓
input feature
    ↓
predict long_view for the same impression
```

The objective is not only to obtain a high score, but to obtain it through a reproducible and scientifically valid autonomous research process.

---

## Acknowledgements

This project was developed for the TikTok TechJam 2026 Autonomous ML Research Agent challenge and uses the organizer-provided KuaiRand starter kit and KuaiRand-Pure benchmark.

The design is informed by broader autonomous-ML research ideas such as evidence-driven experimentation, experiment memory, code optimization, and search over research directions.
