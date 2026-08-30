"""
Description: Runs the organizer-provided baseline to establish the starting benchmark for an autonomous research session.
Owner: Hayden
Input: Starter-kit configuration
Output: ExperimentResult
"""

from src.config import (
    STARTER_KIT_DIR,
    DATA_DIR,
    EXPERIMENT_TIMEOUT,
)

from src.schemas import ExperimentResult
from src.tools.subprocess_runner import run_command
from src.tools.metrics import parse_metrics


def run_starterkit_baseline() -> ExperimentResult:
    """
    Run the official FM baseline and return its validation metrics.
    """

    command = [
        "python",
        "baseline.py",
        "--data_dir",
        str(DATA_DIR),
        "--model",
        "fm",
    ]

    command_result = run_command(
        command=command,
        cwd=STARTER_KIT_DIR,
        timeout=EXPERIMENT_TIMEOUT,
    )

    # ---------------------------------------------
    # Command failed
    # ---------------------------------------------

    if command_result.return_code != 0:
        return ExperimentResult(
            experiment_id="baseline",
            status="failed",
            runtime_seconds=command_result.runtime_seconds,
            stdout=command_result.stdout,
            stderr=command_result.stderr,
            error=command_result.stderr,
        )

    # ---------------------------------------------
    # Parse validation metrics
    # ---------------------------------------------

    try:
        metrics = parse_metrics(
            command_result.stdout,
            split="valid",
        )

    except ValueError as error:
        return ExperimentResult(
            experiment_id="baseline",
            status="failed",
            runtime_seconds=command_result.runtime_seconds,
            stdout=command_result.stdout,
            stderr=command_result.stderr,
            error=str(error),
        )

    # ---------------------------------------------
    # Successful baseline
    # ---------------------------------------------

    return ExperimentResult(
        experiment_id="baseline",
        status="success",
        gauc=metrics.gauc,
        ndcg5=metrics.ndcg5,
        primary=metrics.primary,
        runtime_seconds=command_result.runtime_seconds,
        stdout=command_result.stdout,
        stderr=command_result.stderr,
    )