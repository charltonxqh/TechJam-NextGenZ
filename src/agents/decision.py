"""
Description: Deterministically decides whether an experiment should replace the current best, be rejected, or be retried.
Owner: Charlton / David
Input: ExperimentResult and previous best validation Primary
Output: keep, reject, or retry decision
"""

from typing import Literal

from src.schemas import (
    ExperimentResult,
)


Decision = Literal[
    "keep",
    "reject",
    "retry",
]


def decide_experiment(
    result: ExperimentResult,
    previous_best_primary: float,
) -> Decision:
    """
    Decide how the autonomous loop should treat one experiment.
    """

    if (
        result.status != "success"
        or result.primary is None
    ):
        return "retry"

    if (
        result.primary
        > previous_best_primary
    ):
        return "keep"

    return "reject"