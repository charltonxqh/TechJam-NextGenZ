"""
Description: Uses TRAE to implement proposed ML experiments in isolated workspaces and produce runnable experiment code.
Owner: Hayden
Input: ExperimentSpec
Output: ImplementedExperiment
"""

import difflib
import json
import shutil
from pathlib import Path

from src.config import (
    STARTER_KIT_DIR,
    WORKSPACES_DIR,
    EXPERIMENT_TIMEOUT,
)

from src.schemas import (
    ExperimentSpec,
    ImplementedExperiment,
    RecoveryEvent,
)

from src.tools.subprocess_runner import run_command


class TraeCodingAgent:

    def implement(
        self,
        spec: ExperimentSpec,
    ) -> ImplementedExperiment:
        """
        Ask TRAE to implement one proposed ML experiment.
        """

        # =============================================
        # 1. Prepare isolated workspace
        # =============================================

        workspace = self._prepare_workspace(
            spec
        )

        # =============================================
        # 2. Build TRAE task
        # =============================================

        task = self._build_task(
            spec
        )

        trajectory_path = (
            workspace
            / "trae_trajectory.json"
        )

        command = [
            "trae-cli",
            "run",
            task,
            "--working-dir",
            str(workspace.resolve()),
            "--trajectory-file",
            str(trajectory_path.resolve()),
            "--must-patch",
        ]

        # =============================================
        # 3. Run TRAE
        # =============================================

        result = run_command(
            command=command,
            cwd=workspace,
            timeout=EXPERIMENT_TIMEOUT,
        )

        # Generate diff even if TRAE fails.
        code_diff = self._generate_code_diff(
            workspace
        )

        # =============================================
        # 4. TRAE command failed
        # =============================================

        if result.return_code != 0:

            error_message = (
                result.stderr
                or result.stdout
                or "TRAE coding agent failed."
            )

            recovery_event = RecoveryEvent(
                stage="coding",
                error=error_message,
                action=(
                    "Coding failure was captured and "
                    "returned to the autonomous research "
                    "loop instead of terminating the run."
                ),
                success=False,
            )

            return ImplementedExperiment(
                experiment_id=(
                    spec.experiment_id
                ),
                workspace_path=str(
                    workspace.resolve()
                ),
                command=[],
                status="failed",
                error=error_message,
                code_diff=code_diff,
                recovery_events=[
                    recovery_event
                ],
            )

        # =============================================
        # 5. Load TRAE-generated manifest
        # =============================================

        try:
            return self._load_implemented_experiment(
                spec=spec,
                workspace=workspace,
                code_diff=code_diff,
            )

        except (
            FileNotFoundError,
            ValueError,
            json.JSONDecodeError,
        ) as error:

            recovery_event = RecoveryEvent(
                stage="coding_manifest",
                error=str(error),
                action=(
                    "Invalid or missing experiment "
                    "manifest was captured and returned "
                    "to the research loop."
                ),
                success=False,
            )

            return ImplementedExperiment(
                experiment_id=(
                    spec.experiment_id
                ),
                workspace_path=str(
                    workspace.resolve()
                ),
                command=[],
                status="failed",
                error=str(error),
                code_diff=code_diff,
                recovery_events=[
                    recovery_event
                ],
            )

    # =========================================================
    # Workspace
    # =========================================================

    def _prepare_workspace(
        self,
        spec: ExperimentSpec,
    ) -> Path:
        """
        Create a clean isolated workspace for one experiment.
        """

        workspace = (
            WORKSPACES_DIR
            / spec.experiment_id
        )

        if workspace.exists():
            shutil.rmtree(
                workspace
            )

        shutil.copytree(
            STARTER_KIT_DIR,
            workspace,
        )

        return workspace

    # =========================================================
    # TRAE prompt
    # =========================================================

    def _build_task(
        self,
        spec: ExperimentSpec,
    ) -> str:
        """
        Convert ExperimentSpec into instructions for TRAE.
        """

        parameters = json.dumps(
            spec.parameters,
            indent=2,
            ensure_ascii=False,
        )

        return f"""
You are implementing one ML research experiment for the KuaiRand recommender-system benchmark.

Experiment ID:
{spec.experiment_id}

Hypothesis:
{spec.hypothesis}

Rationale:
{spec.rationale}

Change type:
{spec.change_type}

Parameters:
{parameters}

Your task:
Implement the minimum code changes necessary to test this hypothesis.

Requirements:

1. Inspect the existing starter-kit code before making changes.

2. Preserve the original baseline implementation.

3. Do NOT modify evaluate.py.

4. Do NOT change the official train/validation split.

5. Do NOT use hidden-test information for research decisions.

6. The experiment must train and evaluate on the validation split.

7. Create a runnable experiment entry point.

8. The experiment must print final validation metrics in exactly this format:

GAUC <value> | nDCG@5 <value> | primary <value>

9. Create a file named experiment_manifest.json in the project root.

The manifest must contain:

{{
  "command": [
    "python",
    "<experiment_entry_point>.py"
  ]
}}

10. Do not run the hidden test set.

11. Keep changes focused on the proposed hypothesis.

12. Do not modify unrelated project files.
""".strip()

    # =========================================================
    # Manifest
    # =========================================================

    def _load_implemented_experiment(
        self,
        spec: ExperimentSpec,
        workspace: Path,
        code_diff: str,
    ) -> ImplementedExperiment:
        """
        Read the TRAE-generated experiment manifest.
        """

        manifest_path = (
            workspace
            / "experiment_manifest.json"
        )

        if not manifest_path.exists():
            raise FileNotFoundError(
                "TRAE finished but did not create "
                "experiment_manifest.json."
            )

        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            manifest = json.load(
                file
            )

        command = manifest.get(
            "command"
        )

        if not isinstance(
            command,
            list,
        ):
            raise ValueError(
                "experiment_manifest.json "
                "must contain a 'command' list."
            )

        if not command:
            raise ValueError(
                "Experiment command cannot be empty."
            )

        if not all(
            isinstance(item, str)
            for item in command
        ):
            raise ValueError(
                "Every item in experiment command "
                "must be a string."
            )

        return ImplementedExperiment(
            experiment_id=(
                spec.experiment_id
            ),
            workspace_path=str(
                workspace.resolve()
            ),
            command=command,
            status="success",
            code_diff=code_diff,
        )

    # =========================================================
    # Code diff
    # =========================================================

    def _generate_code_diff(
        self,
        workspace: Path,
    ) -> str:
        """
        Compare the original starter kit against the TRAE workspace
        and return a unified text diff.
        """

        original_files = {
            path.relative_to(
                STARTER_KIT_DIR
            )
            for path in STARTER_KIT_DIR.rglob("*")
            if path.is_file()
        }

        workspace_files = {
            path.relative_to(
                workspace
            )
            for path in workspace.rglob("*")
            if path.is_file()
        }

        all_files = sorted(
            original_files
            | workspace_files,
            key=str,
        )

        ignored_files = {
            "trae_trajectory.json",
            "experiment_manifest.json",
        }

        diff_sections = []

        for relative_path in all_files:

            if (
                relative_path.name
                in ignored_files
            ):
                continue

            if "__pycache__" in relative_path.parts:
                continue

            if ".git" in relative_path.parts:
                continue

            if relative_path.suffix == ".pyc":
                continue

            original_path = (
                STARTER_KIT_DIR
                / relative_path
            )

            workspace_path = (
                workspace
                / relative_path
            )

            original_text = self._read_text_file(
                original_path
            )

            workspace_text = self._read_text_file(
                workspace_path
            )

            # Skip binary/unreadable files.
            if (
                original_text is None
                or workspace_text is None
            ):
                continue

            if (
                original_text
                == workspace_text
            ):
                continue

            diff = difflib.unified_diff(
                original_text.splitlines(
                    keepends=True
                ),
                workspace_text.splitlines(
                    keepends=True
                ),
                fromfile=(
                    f"a/{relative_path}"
                ),
                tofile=(
                    f"b/{relative_path}"
                ),
            )

            diff_sections.append(
                "".join(diff)
            )

        return "\n".join(
            diff_sections
        )

    def _read_text_file(
        self,
        path: Path,
    ) -> str | None:
        """
        Read a text file for diff generation.

        Missing files represent new/deleted files.
        Binary or undecodable files are skipped.
        """

        if not path.exists():
            return ""

        try:
            return path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:
            return None