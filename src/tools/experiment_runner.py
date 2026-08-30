"""
Description: Executes an implemented ML experiment and returns its evaluation results.
Owner: Hayden
Input: ImplementedExperiment
Output: ExperimentResult
"""

from src.config import EXPERIMENT_TIMEOUT

from src.schemas import (
    ImplementedExperiment,
    ExperimentResult,
    RecoveryEvent,
)

from src.tools.subprocess_runner import run_command
from src.tools.metrics import parse_metrics


def run_experiment(
    experiment: ImplementedExperiment,
) -> ExperimentResult:

    # =============================================
    # 1. Check whether coding stage succeeded
    # =============================================

    if experiment.status != "success":
        return ExperimentResult(
            experiment_id=experiment.experiment_id,
            status="failed",
            error=(
                experiment.error
                or "Experiment implementation failed."
            ),
        )

    # =============================================
    # 2. Execute experiment
    # =============================================

    command_result = run_command(
        command=experiment.command,
        cwd=experiment.workspace_path,
        timeout=EXPERIMENT_TIMEOUT,
    )

    # =============================================
    # 3. Handle execution failure
    # =============================================

    if command_result.return_code != 0:

        recovery_event = RecoveryEvent(
            stage="experiment_execution",
            error=command_result.stderr,
            action=(
                "Experiment execution failed and "
                "the failure was returned to the "
                "research loop for reflection."
            ),
            success=False,
        )

        return ExperimentResult(
            experiment_id=experiment.experiment_id,
            status="failed",
            runtime_seconds=(
                command_result.runtime_seconds
            ),
            stdout=command_result.stdout,
            stderr=command_result.stderr,
            error=command_result.stderr,
            recovery_events=[
                recovery_event
            ],
        )

    # =============================================
    # 4. Parse evaluation metrics
    # =============================================

    try:
        metrics = parse_metrics(
            command_result.stdout
        )

    except ValueError as error:

        recovery_event = RecoveryEvent(
            stage="metric_parsing",
            error=str(error),
            action=(
                "Raw experiment output was preserved "
                "for diagnosis and reflection."
            ),
            success=False,
        )

        return ExperimentResult(
            experiment_id=experiment.experiment_id,
            status="failed",
            runtime_seconds=(
                command_result.runtime_seconds
            ),
            stdout=command_result.stdout,
            stderr=command_result.stderr,
            error=str(error),
            recovery_events=[
                recovery_event
            ],
        )

    # =============================================
    # 5. Successful experiment
    # =============================================

    return ExperimentResult(
        experiment_id=experiment.experiment_id,
        status="success",
        gauc=metrics.gauc,
        ndcg5=metrics.ndcg5,
        primary=metrics.primary,
        runtime_seconds=(
            command_result.runtime_seconds
        ),
        stdout=command_result.stdout,
        stderr=command_result.stderr,
    )