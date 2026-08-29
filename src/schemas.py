"""
Description: Defines the shared data structures used for communication between different components of the system.
Owner: Shared
Input: N/A
Output: Shared schemas such as ExperimentSpec and ExperimentResult
"""

import dataclass


@dataclass
class ExperimentSpec:
    """
    What the research agent wants to investigate.
    """
    experiment_id: str
    hypothesis: str
    rationale: str
    change_type: str
    parameters: dict


@dataclass
class ImplementedExperiment:
    """
    Runnable experiment produced by the coding agent.
    """
    experiment_id: str
    workspace_path: str
    command: list[str]


@dataclass
class ExperimentResult:
    """
    Result after running and evaluating the experiment.
    """
    experiment_id: str
    status: str

    gauc: float | None = None
    ndcg5: float | None = None
    primary: float | None = None

    runtime_seconds: float | None = None
    error: str | None = None


@dataclass
class CommandResult:
    return_code: int
    stdout: str
    stderr: str
    runtime_seconds: float