"""
Description: Stores iteration logs, code diffs, recovery events, and run summaries for reproducibility and judging.
Owner: Esther
Input: IterationLog and RunSummary
Output: Persisted run logs and experiment history
"""

import json
from dataclasses import asdict
from pathlib import Path

from src.config import RUNS_DIR
from src.schemas import (
    IterationLog,
    RunSummary,
)


class ExperimentStore:

    def __init__(
        self,
        run_id: str,
    ) -> None:

        self.run_dir = (
            RUNS_DIR / run_id
        )

        self.iteration_dir = (
            self.run_dir / "iterations"
        )

        self.diff_dir = (
            self.run_dir / "diffs"
        )

        self.iteration_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.diff_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save_iteration(
        self,
        log: IterationLog,
    ) -> None:

        iteration_path = (
            self.iteration_dir
            / f"iteration_{log.iteration:03d}.json"
        )

        with iteration_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                asdict(log),
                file,
                indent=2,
                ensure_ascii=False,
            )

        # Save code diff separately as well
        diff_path = (
            self.diff_dir
            / f"{log.experiment_id}.diff"
        )

        diff_path.write_text(
            log.code_diff,
            encoding="utf-8",
        )

    def save_summary(
        self,
        summary: RunSummary,
    ) -> None:

        summary_path = (
            self.run_dir
            / "run_summary.json"
        )

        with summary_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                asdict(summary),
                file,
                indent=2,
                ensure_ascii=False,
            )

    def get_history(self) -> list[dict]:
        history = []

        for path in sorted(
            self.iteration_dir.glob("iteration_*.json")
        ):
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                history.append(
                    json.load(file)
                )

        return history