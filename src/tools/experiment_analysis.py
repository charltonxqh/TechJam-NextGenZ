"""
Description: Produces deterministic factual diagnostics from an experiment result for future research decisions.
Owner: Hayden / Charlton / David
Input: ExperimentResult, previous best Primary, and baseline Primary
Output: ExperimentDiagnostics and prompt-ready diagnostic summary
"""

from src.schemas import (
    ExperimentDiagnostics,
    ExperimentResult,
)


def analyze_experiment(
    result: ExperimentResult,
    previous_best_primary: float,
    baseline_primary: float,
) -> ExperimentDiagnostics:
    """
    Convert raw experiment results into factual diagnostics.

    This function does not use an LLM and does not speculate about why
    an experiment succeeded or failed.
    """

    delta_vs_best = None
    delta_vs_baseline = None

    if (
        result.status == "success"
        and result.primary is not None
    ):

        delta_vs_best = (
            result.primary
            - previous_best_primary
        )

        delta_vs_baseline = (
            result.primary
            - baseline_primary
        )

    return ExperimentDiagnostics(
        experiment_id=(
            result.experiment_id
        ),
        status=(
            result.status
        ),
        gauc=result.gauc,
        ndcg5=result.ndcg5,
        primary=result.primary,
        delta_vs_best=(
            delta_vs_best
        ),
        delta_vs_baseline=(
            delta_vs_baseline
        ),
        runtime_seconds=(
            result.runtime_seconds
        ),
        error=(
            result.error
        ),
    )


def format_diagnostics(
    diagnostics: ExperimentDiagnostics,
) -> str:
    """
    Convert diagnostics into compact factual text for memory and prompts.
    """

    if (
        diagnostics.status
        != "success"
    ):

        return (
            "Experiment status: failed\n"
            f"Error: "
            f"{diagnostics.error or 'Unknown error'}"
        )

    return (
        "Experiment status: success\n"
        f"GAUC: {diagnostics.gauc}\n"
        f"nDCG@5: {diagnostics.ndcg5}\n"
        f"Primary: {diagnostics.primary}\n"
        f"Delta vs previous best: "
        f"{diagnostics.delta_vs_best:+.6f}\n"
        f"Delta vs baseline: "
        f"{diagnostics.delta_vs_baseline:+.6f}\n"
        f"Runtime seconds: "
        f"{diagnostics.runtime_seconds}"
    )