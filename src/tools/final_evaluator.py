"""
Description: Runs the validation-selected best experiment once on the official KuaiRand test split.
Owner: Charlton / David
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


def run_final_test(
    experiment: ImplementedExperiment,
) -> ExperimentResult:
    """
    Evaluate the selected best experiment once on test.

    This function must only be called after the autonomous
    validation-based research loop has finished.
    """

    if experiment.status != "success":
        return ExperimentResult(
            experiment_id=experiment.experiment_id,
            status="failed",
            error=(
                "Cannot run final test for an experiment "
                "whose implementation failed."
            ),
        )

    if not experiment.test_command:
        return ExperimentResult(
            experiment_id=experiment.experiment_id,
            status="failed",
            error=(
                "No test command was provided for the "
                "selected experiment."
            ),
        )

    command_result = run_command(
        command=experiment.test_command,
        cwd=experiment.workspace_path,
        timeout=EXPERIMENT_TIMEOUT,
    )

    if command_result.return_code != 0:

        error_message = (
            command_result.stderr
            or command_result.stdout
            or "Final test execution failed."
        )

        recovery_event = RecoveryEvent(
            stage="final_test",
            error=error_message,
            action=(
                "Final test failure was captured. "
                "No additional research iteration was "
                "performed using test information."
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
            error=error_message,
            recovery_events=[
                recovery_event
            ],
        )

    try:
        metrics = parse_metrics(
            command_result.stdout
        )

    except ValueError as error:

        recovery_event = RecoveryEvent(
            stage="final_test_metrics",
            error=str(error),
            action=(
                "The final test completed but its metrics "
                "could not be parsed. The error was recorded "
                "without feeding test information back into "
                "the research loop."
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