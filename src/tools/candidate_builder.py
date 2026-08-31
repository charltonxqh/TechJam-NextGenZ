"""
Description: Materializes Researcher-generated experiment code into an isolated runnable workspace.
Owner: Charlton / David
Input: Experiment ID and complete candidate Python code
Output: ImplementedExperiment
"""

import shutil
from pathlib import Path

from src.config import (
    DATA_DIR,
    STARTER_KIT_DIR,
    WORKSPACES_DIR,
)

from src.schemas import (
    ImplementedExperiment,
    RecoveryEvent,
)


EXPERIMENT_FILENAME = (
    "generated_experiment.py"
)


def build_candidate(
    experiment_id: str,
    full_code: str,
) -> ImplementedExperiment:
    """
    Create an isolated starter-kit workspace and write the Researcher's
    complete candidate experiment code into it.
    """

    workspace = (
        WORKSPACES_DIR
        / experiment_id
    )

    try:

        if workspace.exists():

            shutil.rmtree(
                workspace
            )

        shutil.copytree(
            STARTER_KIT_DIR,
            workspace,
        )

        experiment_path = (
            workspace
            / EXPERIMENT_FILENAME
        )

        experiment_path.write_text(
            full_code,
            encoding="utf-8",
        )

    except OSError as error:

        recovery_event = RecoveryEvent(
            stage="candidate_materialization",
            error=str(error),
            action=(
                "Candidate workspace creation failed and "
                "the error was returned to the research loop."
            ),
            success=False,
        )

        return ImplementedExperiment(
            experiment_id=(
                experiment_id
            ),
            workspace_path=str(
                workspace.resolve()
            ),
            command=[],
            test_command=[],
            full_code=full_code,
            status="failed",
            error=str(error),
            recovery_events=[
                recovery_event
            ],
        )

    validation_command = [
        "python",
        EXPERIMENT_FILENAME,
        "--data_dir",
        str(DATA_DIR),
        "--split",
        "valid",
    ]

    test_command = [
        "python",
        EXPERIMENT_FILENAME,
        "--data_dir",
        str(DATA_DIR),
        "--split",
        "test",
    ]

    return ImplementedExperiment(
        experiment_id=(
            experiment_id
        ),
        workspace_path=str(
            workspace.resolve()
        ),
        command=(
            validation_command
        ),
        test_command=(
            test_command
        ),
        full_code=full_code,
        status="success",
    )