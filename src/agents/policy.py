"""
Description: Enforces deterministic rules for deciding whether the autonomous research process should continue or stop.
Owner: Charlton / David
Input: Current research state and best validation Primary history
Output: Continue or stop decision
"""

from src.config import (
    MAX_ITERATIONS,
    CONVERGENCE_EPSILON,
    CONVERGENCE_PATIENCE,
)


def has_reached_iteration_limit(
    iteration: int,
) -> bool:

    return (
        iteration
        >= MAX_ITERATIONS
    )


def has_converged(
    best_primary_history: list[float],
) -> bool:
    """
    Converged when the validation-best Primary improves by no more
    than epsilon across the most recent N autonomous iterations.

    The history includes the baseline/reference score immediately
    before those N iterations.
    """

    required_scores = (
        CONVERGENCE_PATIENCE
        + 1
    )

    if (
        len(best_primary_history)
        < required_scores
    ):
        return False

    window = (
        best_primary_history[
            -required_scores:
        ]
    )

    window_gain = (
        window[-1]
        - window[0]
    )

    return (
        window_gain
        <= CONVERGENCE_EPSILON
    )


def should_stop(
    iteration: int,
    best_primary_history: list[float],
) -> bool:

    if has_reached_iteration_limit(
        iteration
    ):
        return True

    if has_converged(
        best_primary_history
    ):
        return True

    return False