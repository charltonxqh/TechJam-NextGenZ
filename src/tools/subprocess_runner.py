"""
Description: Executes terminal commands and captures their output, errors, exit status, and runtime.
Owner: Hayden
Input: Command and execution configuration
Output: CommandResult
"""

import subprocess
import time
from pathlib import Path

from src.schemas import CommandResult


def run_command(
    command: list[str],
    cwd: str | Path | None = None,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
) -> CommandResult:
    """
    Execute one terminal command and return its result.
    """

    start_time = time.time()

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )

        runtime = time.time() - start_time

        return CommandResult(
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            runtime_seconds=runtime,
        )

    except subprocess.TimeoutExpired as error:
        runtime = time.time() - start_time

        return CommandResult(
            return_code=124,
            stdout=error.stdout or "",
            stderr=(
                (error.stderr or "")
                + "\nCommand timed out."
            ),
            runtime_seconds=runtime,
        )

    except Exception as error:
        runtime = time.time() - start_time

        return CommandResult(
            return_code=-1,
            stdout="",
            stderr=str(error),
            runtime_seconds=runtime,
        )