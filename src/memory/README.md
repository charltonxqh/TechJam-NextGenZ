# Memory Module (Esther)

## What this is
A standalone memory system for the autonomous ML agent. It does NOT depend
on how the agent loop, LLM calls, or code execution are built - it only
ever sends/receives plain data. This means it can be built and tested now,
before the agent architecture exists, and dropped in later with zero rework.

## Files
- `records.py` - data structures: `IterationRecord`, `Metrics`, `ResourceUsage`, `FailureType`
- `memory_store.py` - the main class: `MemoryStore`. This is the ONLY class teammates need to import.
- `llm_client.py` - calls Google Gemini to analyze WHY a code change caused a score change. Falls back to a rule-based summary if no key/network - never crashes the run.
- `mock_loop_demo.py` - a fake agent loop (random scores instead of a real LLM/training run) that proves the whole thing works end-to-end. Run it directly: `python3 mock_loop_demo.py`
- `test_llm_summary_demo.py` - realistic fake iterations (real code snippets + hand-picked scores telling an "improve, overfit, recover" story) run through the LLM analysis, so you can inspect exactly what feedback it gives.

## Setup for the LLM analysis part (llm_client.py)

1. Get a free-tier key from https://aistudio.google.com/apikey
2. Put it in the project's `.env` file (recommended) or set it as an
   environment variable. The client loads `GOOGLE_API_KEY` from `.env`
   automatically, and never writes or prints the key:
   ```bash
   # .env (in this project folder)
   GOOGLE_API_KEY="your-key-here"

   # Or set it in your terminal environment:
   export GOOGLE_API_KEY="your-key-here"       # Mac/Linux
   setx GOOGLE_API_KEY "your-key-here"         # Windows (restart terminal after)
   ```
3. Install the one dependency:
   ```bash
   pip install requests
   ```
4. Run the test:
   ```bash
   python3 test_llm_summary_demo.py
   ```

If the key is missing, `requests` isn't installed, or the network call
fails for any reason, the script automatically falls back to a
rule-based summary instead of crashing - this is intentional (see
"Robust operation" in the challenge's Task Requirements) and you'll see
a `NOTE: No GOOGLE_API_KEY found...` message when it happens.

## Try it yourself
```bash
python3 mock_loop_demo.py
```
This simulates 20 iterations with fake scores, shows what gets fed into
the LLM each round, detects convergence automatically, and produces:
- `demo_run_log.jsonl` (raw log, one line per iteration)
- `demo_run_log.md` (human-readable report - matches the deliverable format)
- `demo_run_log_full.json` (full structured export)

## How the REAL agent loop will use this (integration contract)

Whoever builds the actual orchestrator (Charlton/David) needs to do exactly
four things with this module, in this order, each iteration:

```python
from memory_store import MemoryStore
from records import IterationRecord, Metrics, ResourceUsage, FailureType

memory = MemoryStore(log_path="run_log.jsonl")

# 1. BEFORE calling the LLM: get compressed history to put in the prompt
context = memory.get_prompt_context()
prompt = f"...your prompt template...\n\nPrior progress:\n{context}"

# 2. Call your LLM + run the training code + call evaluate.py as normal
#    (this part is NOT this module's job)
hypothesis, stage, code_diff = call_llm(prompt)
metrics, failure, error_msg = run_and_evaluate(code_diff)

# 3. AFTER each iteration: save the result
record = IterationRecord(
    iteration=i,
    hypothesis=hypothesis,
    stage=stage,             # "data" | "features" | "model" | "training" | "eval"
    code_diff=code_diff,
    metrics=Metrics(gauc=..., ndcg5=..., primary=...),
    failure=failure,         # FailureType.NONE if it succeeded
    error_message=error_msg,
    manual_intervention=False,  # set True only if a human had to step in
    resource_usage=ResourceUsage(input_tokens=..., output_tokens=..., wall_clock_sec=...),
)
memory.add(record)

# 4. Check whether to stop
if memory.check_convergence(epsilon=0.002, n=3):
    break
```

At the end of the run, for the deliverables:
```python
memory.export_run_log_markdown("run_log.md")   # for the report
memory.export_run_log_json("run_log_full.json")
print(memory.best())                            # best checkpoint to submit
print(memory.manual_intervention_count())        # for Autonomy scoring
print(memory.total_resource_usage())             # for Feasibility scoring
```

## Using the LLM analysis in the real loop

```python
from memory_store import MemoryStore
from llm_client import LLMClient
from records import IterationRecord, Metrics, ResourceUsage, FailureType

memory = MemoryStore(log_path="run_log.jsonl")
llm = LLMClient()  # reads GOOGLE_API_KEY from environment

# ... after running the agent's generated code and scoring it ...
record = IterationRecord(
    iteration=i, hypothesis=hypothesis, stage=stage, code_diff=code_diff,
    metrics=Metrics(gauc=..., ndcg5=..., primary=...),
)

# This one call analyzes the diff AND stores the record.
memory.add_with_analysis(record, llm)
```

`add_with_analysis()` replaces the plain `add()` call whenever you want
the LLM's summary/reason attached. Use plain `add()` if you ever want to
skip analysis for a given iteration (e.g. to save cost on a run where
you already know the answer, like a pure hyperparameter sweep).

## Design notes / things to discuss with the team

1. **`stage` field**: I used five buckets (`data`, `features`, `model`,
   `training`, `eval`) matching the Figure 1 loop. If the architecture
   team wants different categories, this is a one-line enum change - flag it early.

2. **Convergence check lives here, not in the harness.** Since it reads
   directly from score history, it made sense to keep it next to the
   data. If "harness engineering" (limits/goals) wants to own this logic
   instead, `check_convergence()` can be copied out - it has no other
   dependencies.

3. **Token budget for `get_prompt_context()`**: currently capped at
   ~2000 characters. If real prompts need more/less room, this is a
   parameter (`max_chars`), not a rewrite.

4. **What "manual_intervention" means**: needs a team-wide definition
   (e.g., does restarting a crashed process count? Editing the harness
   config? Suggest agreeing on this before the real run starts, since
   it directly affects the Autonomy score.)

5. **Persistence**: writes to disk after every iteration (JSONL,
   append-only), so a crash mid-run doesn't lose history and the run
   can resume by re-pointing to the same `log_path`.
