"""
memory_store.py
----------------
The main interface for the memory component.

Design goals:
  1. No dependency on how the agent loop, LLM calls, or code execution
     are implemented - it only ever receives/returns plain data.
  2. Persists to disk as it goes, so a crashed run does not lose history.
  3. Provides compressed, token-budgeted context for future Researcher calls.
  4. Converts full generated candidate code into compact diffs for memory.
  5. Keeps the baseline explicitly separate from autonomous hypotheses.
  6. Stores factual experiment diagnostics for use by later research turns.

Stopping and convergence remain the responsibility of the policy layer.
"""

import difflib
import json
import os

from typing import (
    List,
    Optional,
    Dict,
)

try:
    from .records import (
        IterationRecord,
        FailureType,
    )

except ImportError:
    from records import (
        IterationRecord,
        FailureType,
    )


class MemoryStore:

    def __init__(
        self,
        log_path: str = "run_log.jsonl",
    ):

        self.log_path = (
            log_path
        )

        self.history: List[
            IterationRecord
        ] = []

        self._code_cache_path = (
            log_path
            + ".last_code_cache.txt"
        )

        self._last_full_code: (
            Optional[str]
        ) = None

        if os.path.exists(
            self._code_cache_path
        ):

            with open(
                self._code_cache_path,
                "r",
                encoding="utf-8",
            ) as file:

                self._last_full_code = (
                    file.read()
                )

        self._notes_path = (
            log_path
            + ".distilled_notes.txt"
        )

        self._last_consolidated_at = 0

        self.distilled_notes = ""

        if os.path.exists(
            self._notes_path
        ):

            with open(
                self._notes_path,
                "r",
                encoding="utf-8",
            ) as file:

                content = (
                    file.read()
                )

                if (
                    "\n---iter---\n"
                    in content
                ):

                    notes, marker = (
                        content.rsplit(
                            "\n---iter---\n",
                            1,
                        )
                    )

                    self.distilled_notes = (
                        notes
                    )

                    self._last_consolidated_at = (
                        int(
                            marker.strip()
                        )
                    )

        if os.path.exists(
            log_path
        ):
            self._load()

    # ------------------------------------------------------------------
    # 0. CODE COMPRESSION
    # ------------------------------------------------------------------

    def compute_code_diff(
        self,
        current_full_code: str,
        reference_full_code: str | None = None,
    ) -> str:
        """
        Convert full candidate code into the representation stored in memory.

        If reference_full_code is supplied, the diff is explicitly computed
        against that code. The research loop uses the current validation-best
        code as this reference.

        If no reference is supplied, the previous cached full code is used.
        If no previous code exists, the complete code is returned.
        """

        reference = (
            reference_full_code
            if reference_full_code
            is not None
            else self._last_full_code
        )

        if reference is None:

            result = (
                current_full_code
            )

        else:

            diff_lines = (
                difflib.unified_diff(
                    reference.splitlines(
                        keepends=True
                    ),
                    current_full_code.splitlines(
                        keepends=True
                    ),
                    fromfile=(
                        "current_best"
                    ),
                    tofile=(
                        "candidate"
                    ),
                )
            )

            result = "".join(
                diff_lines
            )

            if not result.strip():

                result = (
                    "# no code changes detected "
                    "vs current best"
                )

        self._last_full_code = (
            current_full_code
        )

        with open(
            self._code_cache_path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                current_full_code
            )

        return result

    # ------------------------------------------------------------------
    # 1. RECORD
    # ------------------------------------------------------------------

    def add(
        self,
        record: IterationRecord,
    ) -> None:

        self.history.append(
            record
        )

        with open(
            self.log_path,
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    record.as_dict(),
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _load(
        self,
    ) -> None:

        with open(
            self.log_path,
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = (
                    line.strip()
                )

                if line:

                    self.history.append(
                        IterationRecord.from_dict(
                            json.loads(
                                line
                            )
                        )
                    )

    # ------------------------------------------------------------------
    # 2. COMPRESSED PROMPT CONTEXT
    # ------------------------------------------------------------------

    def get_prompt_context(
        self,
        max_chars: int = 5000,
        recent_full: int = 3,
    ) -> str:
        """
        Return compact factual research memory for the next Researcher call.

        The baseline is presented as a reference benchmark, never as a
        previous autonomous hypothesis.
        """

        if not self.history:

            return (
                "No baseline or previous "
                "research has been recorded."
            )

        lines = []

        baseline = next(
            (
                record
                for record
                in self.history
                if record.is_baseline
            ),
            None,
        )

        research_history = [
            record
            for record
            in self.history
            if not record.is_baseline
        ]

        # --------------------------------------------------------------
        # Baseline
        # --------------------------------------------------------------

        if baseline:

            lines.append(
                "BASELINE REFERENCE:"
            )

            lines.append(
                "This is the official starting benchmark "
                "that autonomous research is trying to improve."
            )

            lines.append(
                "It is NOT a previous research hypothesis."
            )

            if (
                baseline.metrics.primary
                is not None
            ):

                lines.append(
                    f"Validation Primary: "
                    f"{baseline.metrics.primary:.4f}"
                )

            if (
                baseline.metrics.gauc
                is not None
            ):

                lines.append(
                    f"GAUC: "
                    f"{baseline.metrics.gauc:.4f}"
                )

            if (
                baseline.metrics.ndcg5
                is not None
            ):

                lines.append(
                    f"nDCG@5: "
                    f"{baseline.metrics.ndcg5:.4f}"
                )

            lines.append("")

        # --------------------------------------------------------------
        # Distilled long-term memory
        # --------------------------------------------------------------

        if self.distilled_notes:

            lines.append(
                "LESSONS LEARNED SO FAR:"
            )

            lines.append(
                self.distilled_notes
            )

            lines.append("")

        # --------------------------------------------------------------
        # Current best
        # --------------------------------------------------------------

        best = self.best()

        if best:

            if best.is_baseline:

                lines.append(
                    "BEST SO FAR: "
                    "baseline reference remains best "
                    f"with Primary="
                    f"{best.metrics.primary:.4f}."
                )

            else:

                lines.append(
                    f"BEST SO FAR: "
                    f"iteration {best.iteration}, "
                    f"Primary="
                    f"{best.metrics.primary:.4f}, "
                    f"hypothesis="
                    f"{best.hypothesis}"
                )

        # --------------------------------------------------------------
        # Score trend
        # --------------------------------------------------------------

        trend = []

        for record in (
            self.history[-5:]
        ):

            label = (
                "baseline"
                if record.is_baseline
                else (
                    f"iter"
                    f"{record.iteration}"
                )
            )

            score = (
                f"{record.metrics.primary:.4f}"
                if (
                    record.metrics.primary
                    is not None
                )
                else "failed"
            )

            trend.append(
                f"{label}={score}"
            )

        lines.append(
            "RECENT SCORE TREND: "
            + " -> ".join(
                trend
            )
        )

        # --------------------------------------------------------------
        # Failed / rejected ideas
        # --------------------------------------------------------------

        failed = [
            record
            for record
            in research_history
            if (
                record.failure
                != FailureType.NONE
            )
        ]

        if failed:

            lines.append(
                "TRIED AND DID NOT WORK:"
            )

            for record in (
                failed[-8:]
            ):

                reason = (
                    record.error_message
                    or record.notes
                    or record.failure.value
                )

                lines.append(
                    f"  - Iter "
                    f"{record.iteration} "
                    f"[{record.stage}] "
                    f"{record.hypothesis} "
                    f"-> {reason}"
                )

        # --------------------------------------------------------------
        # Recent detailed attempts
        # --------------------------------------------------------------

        if not research_history:

            lines.append(
                "AUTONOMOUS RESEARCH HISTORY: "
                "No research experiments completed yet."
            )

        else:

            recent = (
                research_history[
                    -recent_full:
                ]
            )

            lines.append(
                f"MOST RECENT "
                f"{len(recent)} "
                f"RESEARCH ITERATIONS:"
            )

            for record in recent:

                score = (
                    f"{record.metrics.primary:.4f}"
                    if (
                        record.metrics.primary
                        is not None
                    )
                    else "failed"
                )

                lines.append(
                    f"  - Iter "
                    f"{record.iteration} "
                    f"[{record.stage}]"
                )

                lines.append(
                    f"      hypothesis: "
                    f"{record.hypothesis}"
                )

                lines.append(
                    f"      result: "
                    f"{score}"
                )

                if record.verdict:

                    lines.append(
                        f"      decision: "
                        f"{record.verdict}"
                    )

                if record.diagnostics:

                    diagnostics = (
                        record.diagnostics
                    )

                    lines.append(
                        "      diagnostics:"
                    )

                    lines.append(
                        f"        status="
                        f"{diagnostics.get('status')}"
                    )

                    if (
                        diagnostics.get(
                            "delta_vs_best"
                        )
                        is not None
                    ):

                        lines.append(
                            f"        delta_vs_best="
                            f"{diagnostics.get('delta_vs_best'):+.6f}"
                        )

                    if (
                        diagnostics.get(
                            "delta_vs_baseline"
                        )
                        is not None
                    ):

                        lines.append(
                            f"        delta_vs_baseline="
                            f"{diagnostics.get('delta_vs_baseline'):+.6f}"
                        )

                    if diagnostics.get(
                        "error"
                    ):

                        lines.append(
                            f"        error="
                            f"{diagnostics.get('error')}"
                        )

        text = "\n".join(
            lines
        )

        if len(text) > max_chars:

            text = (
                text[:max_chars]
                + "\n...(truncated)"
            )

        return text

    # ------------------------------------------------------------------
    # 3. QUERY
    # ------------------------------------------------------------------

    def best(
        self,
    ) -> Optional[
        IterationRecord
    ]:

        scored = [
            record
            for record
            in self.history
            if (
                record.metrics.primary
                is not None
            )
        ]

        if not scored:

            return None

        return max(
            scored,
            key=lambda record: (
                record.metrics.primary
                if (
                    record.metrics.primary
                    is not None
                )
                else float("-inf")
            ),
        )

    def manual_intervention_count(
        self,
    ) -> int:

        return sum(
            1
            for record
            in self.history
            if (
                not record.is_baseline
                and record.manual_intervention
            )
        )

    def total_resource_usage(
        self,
    ) -> Dict[str, float]:

        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "wall_clock_sec": 0.0,
            "gpu_hours": 0.0,
        }

        for record in self.history:

            totals[
                "input_tokens"
            ] += (
                record.resource_usage
                .input_tokens
            )

            totals[
                "output_tokens"
            ] += (
                record.resource_usage
                .output_tokens
            )

            totals[
                "wall_clock_sec"
            ] += (
                record.resource_usage
                .wall_clock_sec
            )

            totals[
                "gpu_hours"
            ] += (
                record.resource_usage
                .gpu_hours
            )

        return totals

    # ------------------------------------------------------------------
    # 4. EXPORT
    # ------------------------------------------------------------------

    def export_run_log_markdown(
        self,
        path: str,
    ) -> None:

        lines = [
            "# Run Log\n"
        ]

        baseline = next(
            (
                record
                for record
                in self.history
                if record.is_baseline
            ),
            None,
        )

        if baseline:

            lines.append(
                "## Baseline Reference"
            )

            lines.append(
                "Official starter-kit baseline used "
                "as the benchmark autonomous research "
                "attempts to improve.\n"
            )

            lines.append(
                f"**Metrics:** "
                f"GAUC={baseline.metrics.gauc}, "
                f"nDCG@5={baseline.metrics.ndcg5}, "
                f"primary={baseline.metrics.primary}\n"
            )

            lines.append(
                "---\n"
            )

        research_history = [
            record
            for record
            in self.history
            if not record.is_baseline
        ]

        for record in (
            research_history
        ):

            lines.append(
                f"## Iteration "
                f"{record.iteration} "
                f"({record.stage})"
            )

            lines.append(
                f"**Hypothesis:** "
                f"{record.hypothesis}\n"
            )

            if record.rationale:

                lines.append(
                    f"**Rationale:** "
                    f"{record.rationale}\n"
                )

            lines.append(
                f"**Metrics:** "
                f"GAUC={record.metrics.gauc}, "
                f"nDCG@5={record.metrics.ndcg5}, "
                f"primary={record.metrics.primary}\n"
            )

            if record.verdict:

                lines.append(
                    f"**Decision:** "
                    f"{record.verdict}\n"
                )

            if record.diagnostics:

                lines.append(
                    "**Diagnostics:**"
                )

                for (
                    key,
                    value,
                ) in (
                    record.diagnostics.items()
                ):

                    lines.append(
                        f"- {key}: {value}"
                    )

                lines.append("")

            if (
                record.failure
                != FailureType.NONE
            ):

                lines.append(
                    f"**Failure:** "
                    f"{record.failure.value} "
                    f"- "
                    f"{record.error_message}\n"
                )

            if record.recovery_events:

                lines.append(
                    "**Recovery events:**"
                )

                for event in (
                    record.recovery_events
                ):

                    lines.append(
                        f"- {event}"
                    )

                lines.append("")

            if (
                record.manual_intervention
            ):

                lines.append(
                    "**Manual intervention:** "
                    "yes\n"
                )

            lines.append(
                "**Code diff:**\n"
                "```diff\n"
                f"{record.code_diff}\n"
                "```\n"
            )

            lines.append(
                "---\n"
            )

        usage = (
            self.total_resource_usage()
        )

        lines.append(
            "# Summary"
        )

        lines.append(
            f"- Total iterations: "
            f"{len(research_history)}"
        )

        lines.append(
            f"- Manual interventions: "
            f"{self.manual_intervention_count()}"
        )

        lines.append(
            f"- Total tokens: "
            f"{usage['input_tokens'] + usage['output_tokens']}"
        )

        lines.append(
            f"- Total wall-clock (sec): "
            f"{usage['wall_clock_sec']:.1f}"
        )

        best = self.best()

        if best:

            best_label = (
                "baseline"
                if best.is_baseline
                else (
                    f"iteration "
                    f"{best.iteration}"
                )
            )

            lines.append(
                f"- Best validation Primary: "
                f"{best.metrics.primary:.4f} "
                f"({best_label})"
            )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                "\n".join(
                    lines
                )
            )

    def export_run_log_json(
        self,
        path: str,
    ) -> None:

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                [
                    record.as_dict()
                    for record
                    in self.history
                ],
                file,
                indent=2,
                ensure_ascii=False,
            )