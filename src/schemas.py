"""
Description: Defines the shared data structures used for communication between system components.
Owner: Shared
Input: N/A
Output: ExperimentSpec, ImplementedExperiment, ExperimentResult, CommandResult, and Reflection
"""

from dataclasses import dataclass, field
from typing import Any, Literal
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

    improvements: list[float] = field(default_factory=list)

    manual_interventions: int = 0


@dataclass
class ExperimentSpec:
    """
    Describes the ML experiment proposed by the research agent.
    """

    experiment_id: str
    hypothesis: str
    rationale: str
    change_type: str
    parameters: dict[str, Any] = field(default_factory=dict)


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


@dataclass
class ImplementedExperiment:
    """
    Describes a runnable experiment produced by the coding agent.
    """

    experiment_id: str
    workspace_path: str
    command: list[str]

    status: str = "success"
    error: str | None = None

    code_diff: str = ""
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
class CommandResult:
    """
    Stores the result of executing a terminal command.
    """

    return_code: int
    stdout: str
    stderr: str
    runtime_seconds: float


@dataclass
class Reflection:
    """
    Stores the research agent's interpretation of an experiment result.
    """

    verdict: str
    analysis: str
    next_direction: str | None = None


class ResearchProposal(BaseModel):
    hypothesis: str = Field(
        description="A clear and falsifiable ML research hypothesis."
    )

    rationale: str = Field(
        description="Why this experiment is worth testing based on available evidence."
    )

    change_type: str = Field(
        description="Short category such as loss, feature, model, training, multi_task, temporal, or sequence."
    )

    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured parameters describing the proposed experiment."
    )


class ReflectionOutput(BaseModel):
    verdict: Literal[
        "keep",
        "reject",
        "retry",
    ]

    analysis: str = Field(
        description="What was learned from the experiment."
    )

    next_direction: str | None = Field(
        default=None,
        description="A short research direction that should inform the next experiment."
    )