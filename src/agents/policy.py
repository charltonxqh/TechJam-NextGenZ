"""
Description: Enforces deterministic rules for deciding whether the autonomous research process should continue, retry, or stop.
Owner: Charlton / David
Input: Current research state and experiment history
Output: Continue, retry, or stop decision
"""

from src.config import (
    MAX_ITERATIONS,
    CONVERGENCE_EPSILON,
    CONVERGENCE_PATIENCE,
)


def has_reached_iteration_limit(iteration: int) -> bool:
    return iteration >= MAX_ITERATIONS


def has_converged(improvements: list[float]) -> bool:
    """
    Converged when the last N research iterations each improved
    validation Primary by no more than epsilon.
    """

    if len(improvements) < CONVERGENCE_PATIENCE:
        return False

    recent = improvements[-CONVERGENCE_PATIENCE:]

    return all(
        improvement <= CONVERGENCE_EPSILON
        for improvement in recent
    )


def should_stop(
    iteration: int,
    improvements: list[float],
) -> bool:

    if has_reached_iteration_limit(iteration):
        return True

    if has_converged(improvements):
        return True

    return False