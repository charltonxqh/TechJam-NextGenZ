"""
memory_store.py
----------------
The main interface for the memory component. This is the ONE class your
teammates (agent architecture, harness engineering) need to import.

Design goals:
  1. No dependency on how the agent loop, LLM calls, or code execution
     are implemented - it only ever receives/returns plain data.
  2. Persists to disk (JSONL) as it goes, so a crashed run doesn't lose
     history - this also directly satisfies the "run log" deliverable.
  3. Provides a compressed, token-budgeted summary for LLM prompts, so
     the context doesn't grow unboundedly over 50 iterations.
  4. Provides the convergence check as a single method, based on the
     official rule: eps=0.002, N=3 consecutive iterations.
  5. Converts the agent's FULL generated code each iteration into a
     compact form for storage: the very first iteration keeps the full
     code (nothing to compare against yet), every iteration after that
     stores only a diff against the previous iteration's code. This
     keeps the run log and the LLM prompt context small even though the
     agent hands over a complete script every time.
"""

import json
import os
import difflib
from typing import List, Optional, Dict, Any
try:
    from .records import IterationRecord, Metrics, FailureType
except ImportError:
    from records import IterationRecord, Metrics, FailureType


class MemoryStore:
    def __init__(self, log_path: str = "run_log.jsonl"):
        self.log_path = log_path
        self.history: List[IterationRecord] = []

        # Tracks the last FULL code seen, so compute_code_diff() can diff
        # against it next time instead of storing full code every round.
        # Persisted to a small side-file so a crash/restart doesn't lose
        # it (otherwise, after a restart, the next iteration would look
        # like a "first iteration" and store full code again).
        self._code_cache_path = log_path + ".last_code_cache.txt"
        self._last_full_code: Optional[str] = None
        if os.path.exists(self._code_cache_path):
            with open(self._code_cache_path, "r") as f:
                self._last_full_code = f.read()

        # Distilled "lessons learned" notes, built up periodically by the
        # LLM (see consolidate_if_needed) rather than by a fixed rule.
        # Persisted separately so a restart doesn't lose accumulated notes.
        self._notes_path = log_path + ".distilled_notes.txt"
        self._last_consolidated_at = 0
        self.distilled_notes = ""
        if os.path.exists(self._notes_path):
            with open(self._notes_path, "r") as f:
                content = f.read()
                if "\n---iter---\n" in content:
                    notes, marker = content.rsplit("\n---iter---\n", 1)
                    self.distilled_notes = notes
                    self._last_consolidated_at = int(marker.strip())

        # if a log already exists (e.g. resuming a crashed run), load it
        if os.path.exists(log_path):
            self._load()

    # ------------------------------------------------------------------
    # 0. CODE COMPRESSION
    # ------------------------------------------------------------------
    def compute_code_diff(self, current_full_code: str) -> str:
        """
        Call this with the agent's FULL generated code for the current
        iteration, BEFORE constructing the IterationRecord. Returns what
        should actually go into IterationRecord.code_diff:

          - First call ever (no prior code cached): returns the full
            code as-is, since there's nothing to diff against.
          - Every call after that: returns a compact unified diff
            against the previous iteration's full code.

        Usage:
            code_diff = memory.compute_code_diff(full_code_from_agent)
            record = IterationRecord(..., code_diff=code_diff, ...)
            memory.add_with_analysis(record, llm)
        """
        if self._last_full_code is None:
            result = current_full_code
        else:
            diff_lines = difflib.unified_diff(
                self._last_full_code.splitlines(keepends=True),
                current_full_code.splitlines(keepends=True),
                fromfile="previous_iteration",
                tofile="this_iteration",
            )
            result = "".join(diff_lines)
            if not result.strip():
                result = "# no code changes detected vs previous iteration"

        # update cache for next time, persist so restarts don't lose it
        self._last_full_code = current_full_code
        with open(self._code_cache_path, "w") as f:
            f.write(current_full_code)

        return result

    # ------------------------------------------------------------------
    # 1. RECORD
    # ------------------------------------------------------------------
    def add(self, record: IterationRecord) -> None:
        """Append one iteration's record, persist immediately to disk."""
        self.history.append(record)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record.as_dict()) + "\n")

    def add_with_analysis(self, record: IterationRecord, llm_client) -> None:
        """
        Like `add()`, but first calls the LLM client to fill in
        `code_summary` and `likely_reason` on the record, based on the
        code diff (already compacted via compute_code_diff) and the
        before/after score.

        `llm_client` must have an `.analyze_change(hypothesis, code_diff,
        primary_before, primary_after)` method returning a CodeAnalysis
        (see llm_client.py). Passed in rather than imported directly, so
        this module has no hard dependency on any specific LLM provider.

        Never raises - if the LLM call fails, llm_client itself is
        responsible for returning a rule-based fallback (see llm_client.py).
        """
        prev_best = self.best()
        primary_before = prev_best.metrics.primary if prev_best else None
        primary_after = record.metrics.primary

        analysis = llm_client.analyze_change(
            hypothesis=record.hypothesis,
            code_diff=record.code_diff,
            primary_before=primary_before,
            primary_after=primary_after,
        )
        record.code_summary = analysis.summary
        record.likely_reason = analysis.likely_reason
        if analysis.used_fallback:
            record.notes = (record.notes + " | " if record.notes else "") + \
                "LLM analysis fell back to rule-based summary"

        self.add(record)

    def consolidate_if_needed(self, llm_client, every_n: int = 10) -> bool:
        """
        Call this after add()/add_with_analysis() each iteration. Every
        `every_n` iterations, asks the LLM to compress the batch of
        iterations since the last consolidation into a short "lessons
        learned" note, merged with whatever was distilled before.

        This is what keeps get_prompt_context() bounded even after 50+
        iterations - instead of a fixed "keep last N" rule silently
        dropping mid-run discoveries, the LLM decides what's still worth
        keeping each time it re-writes the notes.

        Returns True if consolidation ran this call, False otherwise.
        """
        since_last = len(self.history) - self._last_consolidated_at
        if since_last < every_n:
            return False

        batch = self.history[self._last_consolidated_at:]
        summaries = []
        for r in batch:
            score_str = f"{r.metrics.primary:.4f}" if r.metrics.primary is not None else "failed"
            detail = r.code_summary or r.hypothesis
            summaries.append(f"[{r.stage}] {detail} -> {score_str} ({r.likely_reason})")

        self.distilled_notes = llm_client.consolidate_learnings(summaries, self.distilled_notes)
        self._last_consolidated_at = len(self.history)

        with open(self._notes_path, "w") as f:
            f.write(self.distilled_notes + "\n---iter---\n" + str(self._last_consolidated_at))

        return True

    def _load(self) -> None:
        with open(self.log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.history.append(IterationRecord.from_dict(json.loads(line)))

    # ------------------------------------------------------------------
    # 2. COMPRESS -> prompt-ready context for the LLM
    # ------------------------------------------------------------------
    def get_prompt_context(self, max_chars: int = 2000, recent_full: int = 3) -> str:
        """
        Returns a compact text block summarizing everything tried so far.
        Feed this directly into the "propose next step" LLM prompt.

        Strategy:
          - Always show current best score + what produced it.
          - Always show the trend of the last few scores (for convergence
            awareness and to help the LLM notice plateaus).
          - Show FAILED attempts as one-liners, grouped, so the agent
            doesn't retry the same dead end.
          - Show the most recent `recent_full` iterations in a bit more
            detail (hypothesis + outcome), since they're most relevant.
          - Hard-capped at max_chars so token cost stays predictable.
        """
        if not self.history:
            return "No prior iterations. This is the first attempt."

        lines = []

        if self.distilled_notes:
            lines.append("LESSONS LEARNED SO FAR (from earlier iterations):")
            lines.append(self.distilled_notes)
            lines.append("")

        best = self.best()
        if best:
            lines.append(
                f"BEST SO FAR: iteration {best.iteration}, "
                f"primary={best.metrics.primary:.4f} "
                f"(stage={best.stage}: {best.hypothesis})"
            )

        trend = [
            f"{r.metrics.primary:.4f}" if r.metrics.primary is not None else "n/a"
            for r in self.history[-5:]
        ]
        lines.append("RECENT SCORE TREND: " + " -> ".join(trend))

        failed = [r for r in self.history if r.failure != FailureType.NONE]
        if failed:
            lines.append("TRIED AND DID NOT WORK:")
            for r in failed[-8:]:  # cap how many we list
                reason = r.notes or r.error_message or r.failure.value
                lines.append(f"  - [{r.stage}] {r.hypothesis} -> {reason}")

        lines.append(f"MOST RECENT {min(recent_full, len(self.history))} ITERATIONS:")
        for r in self.history[-recent_full:]:
            score_str = f"{r.metrics.primary:.4f}" if r.metrics.primary is not None else "failed"
            # Prefer the LLM's code_summary/likely_reason (short) over
            # dumping the raw code_diff (long) - keeps prompt cost bounded.
            detail = r.code_summary or r.hypothesis
            lines.append(f"  - Iter {r.iteration} [{r.stage}]: {detail} -> {score_str}")
            if r.likely_reason:
                lines.append(f"      reason: {r.likely_reason}")

        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...(truncated)"
        return text

    # ------------------------------------------------------------------
    # 3. DECIDE
    # ------------------------------------------------------------------
    def best(self) -> Optional[IterationRecord]:
        """Return the iteration with the highest validation primary score."""
        scored = [r for r in self.history if r.metrics.primary is not None]
        if not scored:
            return None
        return max(scored, key=lambda r: r.metrics.primary if r.metrics.primary is not None else float("-inf"))

    def check_convergence(self, epsilon: float = 0.002, n: int = 3) -> bool:
        """
        Official convergence rule: converged when the validation primary
        score has not improved by more than `epsilon` over the last `n`
        consecutive iterations.
        """
        scored = [r.metrics.primary for r in self.history if r.metrics.primary is not None]
        if len(scored) < n + 1:
            return False
        window = scored[-(n + 1):]
        baseline = window[0]
        improvements = [max(0.0, s - baseline) for s in window[1:]]
        return max(improvements) <= epsilon

    def manual_intervention_count(self) -> int:
        """Used for the Autonomy scoring criterion."""
        return sum(1 for r in self.history if r.manual_intervention)

    def total_resource_usage(self) -> Dict[str, float]:
        """Used for the Feasibility & Practicality deliverable."""
        totals = {"input_tokens": 0, "output_tokens": 0, "wall_clock_sec": 0.0, "gpu_hours": 0.0}
        for r in self.history:
            totals["input_tokens"] += r.resource_usage.input_tokens
            totals["output_tokens"] += r.resource_usage.output_tokens
            totals["wall_clock_sec"] += r.resource_usage.wall_clock_sec
            totals["gpu_hours"] += r.resource_usage.gpu_hours
        return totals

    # ------------------------------------------------------------------
    # 4. EXPORT (deliverables)
    # ------------------------------------------------------------------
    def export_run_log_markdown(self, path: str) -> None:
        """Human-readable version of the run log, for the submission."""
        lines = ["# Run Log\n"]
        for r in self.history:
            lines.append(f"## Iteration {r.iteration} ({r.stage})")
            lines.append(f"**Hypothesis:** {r.hypothesis}\n")
            lines.append(f"**Metrics:** GAUC={r.metrics.gauc}, nDCG@5={r.metrics.ndcg5}, "
                         f"primary={r.metrics.primary}\n")
            if r.failure != FailureType.NONE:
                lines.append(f"**Failure:** {r.failure.value} - {r.error_message}\n")
            if r.manual_intervention:
                lines.append("**Manual intervention:** yes\n")
            lines.append(f"**Code diff:**\n```\n{r.code_diff}\n```\n")
            lines.append("---\n")

        summary = self.total_resource_usage()
        lines.append("# Summary")
        lines.append(f"- Total iterations: {len(self.history)}")
        lines.append(f"- Manual interventions: {self.manual_intervention_count()}")
        lines.append(f"- Total tokens: {summary['input_tokens'] + summary['output_tokens']}")
        lines.append(f"- Total wall-clock (sec): {summary['wall_clock_sec']:.1f}")
        best = self.best()
        if best:
            lines.append(f"- Best primary score: {best.metrics.primary:.4f} (iteration {best.iteration})")

        with open(path, "w") as f:
            f.write("\n".join(lines))

    def export_run_log_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump([r.as_dict() for r in self.history], f, indent=2)
