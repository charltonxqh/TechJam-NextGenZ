"""
Description: Defines the shared data structures used for communication between system components.
Owner: Shared
Input: N/A
Output: Shared experiment, research-action, execution, diagnostics, and run-state schemas
"""

from dataclasses import (
    dataclass,
    field,
)

from typing import (
    Any,
    Literal,
    Union,
)

from pydantic import (
    BaseModel,
    Field,
)


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

    best_primary_history: list[
        float
    ] = field(
        default_factory=list
    )

    manual_interventions: int = 0

    final_test_gauc: (
        float
        | None
    ) = None

    final_test_ndcg5: (
        float
        | None
    ) = None

    final_test_primary: (
        float
        | None
    ) = None


@dataclass
class ExperimentSpec:
    """
    Describes the ML experiment proposed by the research agent.
    """

    experiment_id: str

    hypothesis: str
    rationale: str
    change_type: str

    parameters: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    implementation_instructions: list[
        str
    ] = field(
        default_factory=list
    )


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

    runtime_seconds: (
        float
        | None
    )

    recovery_events: list[
        RecoveryEvent
    ] = field(
        default_factory=list
    )

    manual_interventions: int = 0

    reflection: (
        str
        | None
    ) = None


@dataclass
class RunSummary:
    total_iterations: int
    total_manual_interventions: int

    baseline_primary: float
    best_primary: float
    best_experiment_id: str

    total_runtime_seconds: float

    final_test_gauc: (
        float
        | None
    ) = None

    final_test_ndcg5: (
        float
        | None
    ) = None

    final_test_primary: (
        float
        | None
    ) = None


@dataclass
class ImplementedExperiment:
    """
    Describes a runnable experiment produced by the coding agent.
    """

    experiment_id: str
    workspace_path: str
    command: list[str]

    test_command: list[
        str
    ] = field(
        default_factory=list
    )

    full_code: str = ""

    status: str = "success"

    error: (
        str
        | None
    ) = None

    code_diff: str = ""

    recovery_events: list[
        RecoveryEvent
    ] = field(
        default_factory=list
    )


@dataclass
class ExperimentResult:
    """
    Stores the result after executing and evaluating an experiment.
    """

    experiment_id: str
    status: str

    gauc: (
        float
        | None
    ) = None

    ndcg5: (
        float
        | None
    ) = None

    primary: (
        float
        | None
    ) = None

    runtime_seconds: (
        float
        | None
    ) = None

    stdout: str = ""
    stderr: str = ""

    error: (
        str
        | None
    ) = None

    recovery_events: list[
        RecoveryEvent
    ] = field(
        default_factory=list
    )


@dataclass
class ExperimentDiagnostics:
    """
    Stores deterministic factual analysis of one experiment result.
    """

    experiment_id: str
    status: str

    gauc: (
        float
        | None
    ) = None

    ndcg5: (
        float
        | None
    ) = None

    primary: (
        float
        | None
    ) = None

    delta_vs_best: (
        float
        | None
    ) = None

    delta_vs_baseline: (
        float
        | None
    ) = None

    runtime_seconds: (
        float
        | None
    ) = None

    error: (
        str
        | None
    ) = None


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

    next_direction: (
        str
        | None
    ) = None


ResearchActionType = Literal[
    "research",
    "eda",
    "load_skill",
    "experiment",
]


ResearchSource = Literal[
    "web",
    "arxiv",
    "both",
]


class ResearchRequest(
    BaseModel
):
    """
    Requests external research before deciding on an experiment.
    """

    action_type: Literal[
        "research"
    ] = "research"

    reason: str = Field(
        description=(
            "Why external research "
            "is needed before choosing "
            "the next experiment."
        )
    )

    knowledge_gap: str = Field(
        description=(
            "The concrete unresolved "
            "technical question that cannot "
            "be answered from the currently "
            "available EDA, research knowledge, "
            "experiment memory, or loaded skills."
        )
    )

    research_query: str = Field(
        description=(
            "Focused search query "
            "addressing the stated "
            "knowledge gap."
        )
    )

    research_source: (
        ResearchSource
    ) = Field(
        default="both",
        description=(
            "Whether to search "
            "the general web, arXiv, "
            "or both."
        ),
    )


class EDARequest(
    BaseModel
):
    """
    Requests deterministic dataset analysis before deciding on an experiment.
    """

    action_type: Literal[
        "eda"
    ] = "eda"

    reason: str = Field(
        description=(
            "Why this dataset property "
            "needs to be measured."
        )
    )

    eda_tool: str = Field(
        description=(
            "Name of the deterministic "
            "EDA tool to execute."
        )
    )


class SkillRequest(
    BaseModel
):
    """
    Requests full procedural guidance from one or more relevant skills.
    """

    action_type: Literal[
        "load_skill"
    ] = "load_skill"

    reason: str = Field(
        description=(
            "Why procedural guidance "
            "from these skills is useful "
            "for the current research step."
        )
    )

    skills: list[str] = Field(
        min_length=1,
        description=(
            "Names of relevant skills "
            "selected from the available "
            "skill metadata catalog."
        ),
    )


class ExperimentProposal(
    BaseModel
):
    """
    Proposes one controlled ML experiment for the Coder to implement.
    """

    action_type: Literal[
        "experiment"
    ] = "experiment"

    reason: str = Field(
        description=(
            "Why the currently available "
            "evidence is sufficient to "
            "run this experiment."
        )
    )

    hypothesis: str = Field(
        description=(
            "A clear and falsifiable "
            "ML research hypothesis."
        )
    )

    rationale: str = Field(
        description=(
            "Why this experiment is "
            "justified by the available "
            "dataset, research, and "
            "experiment evidence."
        )
    )

    change_type: str = Field(
        description=(
            "Short category such as "
            "loss, feature, model, "
            "training, multi_task, "
            "temporal, or sequence."
        )
    )

    parameters: dict[
        str,
        Any,
    ] = Field(
        description=(
            "Structured parameters "
            "describing the proposed "
            "experiment."
        )
    )

    implementation_instructions: list[
        str
    ] = Field(
        description=(
            "Concrete implementation "
            "constraints that tell the "
            "Coder exactly how the "
            "hypothesis should be "
            "implemented."
        )
    )


class ResearchProposal(
    BaseModel
):
    """
    Structured next-action decision produced by the Researcher.
    """

    decision: Union[
        ResearchRequest,
        EDARequest,
        SkillRequest,
        ExperimentProposal,
    ] = Field(
        discriminator=(
            "action_type"
        )
    )


@dataclass
class ResearchAction:
    """
    Internal representation of one Researcher decision.
    """

    action_type: (
        ResearchActionType
    )

    reason: str

    knowledge_gap: (
        str
        | None
    ) = None

    research_query: (
        str
        | None
    ) = None

    research_source: (
        ResearchSource
        | None
    ) = None

    eda_tool: (
        str
        | None
    ) = None

    skills: (
        list[str]
        | None
    ) = None

    spec: (
        ExperimentSpec
        | None
    ) = None


class ReflectionOutput(
    BaseModel
):

    verdict: Literal[
        "keep",
        "reject",
        "retry",
    ]

    analysis: str = Field(
        description=(
            "What was learned "
            "from the experiment."
        )
    )

    next_direction: (
        str
        | None
    ) = Field(
        default=None,
        description=(
            "A short research direction "
            "that should inform the "
            "next experiment."
        ),
    )