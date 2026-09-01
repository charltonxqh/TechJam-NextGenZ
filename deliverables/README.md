# Deliverables

Everything here comes from one autonomous run of `src/engine/agent.py` on
KuaiRand-Pure, finished 2026-08-31. No file was edited by hand after the run.

## What to look at

| File | Deliverable |
|---|---|
| `submitted-run/submission_draft.csv` | **The model output.** 170,588 scored test rows in the Starter Kit schema `row_id,user_id,video_id,score`. Validate with `python kuairand-starter-kit/submit.py --check`. |
| `ITERATION-LOG.md` | **The per-iteration run log.** All twelve iterations: hypothesis and why, the code diff applied, resulting GAUC / nDCG@5, and every error and recovery event. Manual interventions: 0. |
| `../README.md` | Results table, absolute delta over the baseline, and resource usage. |

## Supporting evidence

These back the numbers above; they are not deliverables in themselves.

| File | What it shows |
|---|---|
| `submitted-run/run_log.jsonl` | The raw log the agent wrote as it ran. First line is the stopping rule, declared before iteration 1. |
| `submitted-run/final_results.json` | Validation and test scores, and the scored delta. |
| `submitted-run/usage.json` | Tokens, LLM calls, wall-clock, iterations used, manual interventions. |
| `submitted-run/tree.json` | The search tree — nine nodes with parent links, so every diff in the iteration log is reproducible. |
| `submitted-run/best_solution.py` | The code that produced the submission (iteration 8). |
| `submitted-run/state.json` | Full loop state, including every attempt the agent made. |
| `submitted-run/findings.json`, `memory.json` | What the run learned and carried forward. |
| `submitted-run/evaluate.sha256` | Checksum of the scorer, verified at startup. The run refuses to begin if `evaluate.py` has changed. |

## Headline numbers

| | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Validation-best | 0.6723 | 0.5379 | 0.6051 |
| Final test (one-time) | 0.6657 | 0.5304 | 0.5981 |

Absolute delta over the baseline on test: **+0.003493**.
12 iterations, 120.6 minutes, 418,825 tokens, 0 GPU-hours, 0 manual interventions.

Bonus benchmarks (KuaiRand-1k, KuaiRand-27k) were not attempted.

## Regenerating the iteration log

```bash
python src/engine/make_iteration_log.py
```

It reads `run_log.jsonl` and `tree.json` and rewrites `ITERATION-LOG.md`. The
diffs are computed from the code each iteration recorded, not written by hand.
