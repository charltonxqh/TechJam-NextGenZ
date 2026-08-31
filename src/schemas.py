"""
Description: Defines the shared data structures used for communication between system components.
Owner: Shared
Input: N/A
Output: ResearchAction, ExperimentSpec, ImplementedExperiment, ExperimentResult, CommandResult, and diagnostics
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class Metrics:
    gauc: float
    ndcg5: float
    primary: float


@dataclass
class RunState:
    iteration: int

    best_experiment_id: str
    best_primary: float

    improvements: list[float] = field(
        default_factory=list
    )

    best_primary_history: list[float] = field(
        default_factory=list
    )

    manual_interventions: int = 0

    final_test_gauc: float | None = None
    final_test_ndcg5: float | None = None
    final_test_primary: float | None = None


@dataclass
class ExperimentSpec:
    """
    Describes the ML experiment proposed by the research agent.
    """

    experiment_id: str
    hypothesis: str
    rationale: str
    change_type: str

    parameters: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ResearchAction:
    """
    Stores one complete research decision produced by the Researcher.

    The Researcher reasons from the current validation-best implementation
    and returns the complete resulting candidate experiment code after
    making one focused research change.
    """

    spec: ExperimentSpec
    full_code: str


@dataclass
class RecoveryEvent:
    stage: str
    error: str
    action: str
    success: bool


@dataclass
class IterationLog:
    iteration: int
    experiment_id: str

    hypothesis: str
    rationale: str

    code_diff: str

    gauc: float | None
    ndcg5: float | None
    primary: float | None

    status: str
    runtime_seconds: float | None

    recovery_events: list[RecoveryEvent] = field(
        default_factory=list
    )

    manual_interventions: int = 0

    reflection: str | None = None


@dataclass
class RunSummary:
    total_iterations: int
    total_manual_interventions: int

    baseline_primary: float
    best_primary: float
    best_experiment_id: str

    total_runtime_seconds: float

    final_test_gauc: float | None = None
    final_test_ndcg5: float | None = None
    final_test_primary: float | None = None


@dataclass
class ImplementedExperiment:
    """
    Describes a runnable experiment produced from Researcher-generated code.
    """

    experiment_id: str
    workspace_path: str
    command: list[str]

    test_command: list[str] = field(
        default_factory=list
    )

    full_code: str = ""

    status: str = "success"
    error: str | None = None

    recovery_events: list[RecoveryEvent] = field(
        default_factory=list
    )


@dataclass
class ExperimentResult:
    """
    Stores the result after executing and evaluating an experiment.
    """

    experiment_id: str
    status: str

    gauc: float | None = None
    ndcg5: float | None = None
    primary: float | None = None

    runtime_seconds: float | None = None

    stdout: str = ""
    stderr: str = ""

    error: str | None = None

    recovery_events: list[RecoveryEvent] = field(
        default_factory=list
    )


@dataclass
class ExperimentDiagnostics:
    """
    Stores deterministic factual diagnostics for one experiment.
    """

    experiment_id: str
    status: str

    gauc: float | None = None
    ndcg5: float | None = None
    primary: float | None = None

    delta_vs_best: float | None = None
    delta_vs_baseline: float | None = None

    runtime_seconds: float | None = None

    error: str | None = None


@dataclass
class CommandResult:
    """
    Stores the result of executing a terminal command.
    """

    return_code: int
    stdout: str
    stderr: str
    runtime_seconds: float


class ResearchProposal(BaseModel):

    hypothesis: str = Field(
        description=(
            "A clear and falsifiable ML research hypothesis."
        )
    )

    rationale: str = Field(
        description=(
            "Why this experiment is worth testing "
            "based on available evidence."
        )
    )

    change_type: str = Field(
        description=(
            "Short category such as loss, feature, model, "
            "training, multi_task, temporal, or sequence."
        )
    )

    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured parameters describing "
            "the proposed experiment."
        ),
    )

    full_code: str = Field(
        description=(
            "The complete runnable Python experiment file after applying "
            "exactly one focused research change to the current-best code."
        )
    )