"""
records.py
----------
Defines the data structures for one baseline reference or one iteration of
the ML agent's improvement loop. This is the "unit of memory" - everything
else in the memory module operates on lists of these.

Kept deliberately independent of any specific agent architecture: whoever
builds the main loop just needs to construct one of these per iteration
and hand it to MemoryStore.add().
"""

from dataclasses import (
    dataclass,
    field,
    asdict,
)

from typing import (
    Optional,
    Dict,
    Any,
    List,
)

from enum import Enum

import time


class FailureType(
    str,
    Enum,
):
    """Taxonomy of failure modes, used for pattern-spotting across a run."""

    NONE = "none"
    CODE_ERROR = "code_error"
    TIMEOUT = "timeout"
    BAD_OUTPUT = "bad_output"
    NO_IMPROVEMENT = "no_improvement"
    OTHER = "other"


@dataclass
class Metrics:
    """Scores for one iteration. Matches the challenge's GAUC / nDCG@5 / primary."""

    gauc: Optional[float] = None
    ndcg5: Optional[float] = None
    primary: Optional[float] = None

    def as_dict(
        self,
    ) -> Dict[str, Any]:

        return asdict(
            self
        )


@dataclass
class ResourceUsage:
    """Cost tracking, needed for the Feasibility & Practicality deliverable."""

    input_tokens: int = 0
    output_tokens: int = 0
    wall_clock_sec: float = 0.0
    gpu_hours: float = 0.0

    def as_dict(
        self,
    ) -> Dict[str, Any]:

        return asdict(
            self
        )


@dataclass
class IterationRecord:
    """
    One baseline reference or one full round of the agent's loop:
    hypothesis -> candidate code -> run -> diagnostics -> decision.

    The baseline uses iteration=0 and is_baseline=True. It is stored so
    scores can be compared against it, but it is not treated as a previous
    research hypothesis.
    """

    iteration: int
    hypothesis: str
    stage: str
    code_diff: str

    code_summary: str = ""
    likely_reason: str = ""

    metrics: Metrics = field(
        default_factory=Metrics
    )

    failure: FailureType = (
        FailureType.NONE
    )

    error_message: Optional[str] = None

    manual_intervention: bool = False

    resource_usage: ResourceUsage = field(
        default_factory=ResourceUsage
    )

    timestamp: float = field(
        default_factory=time.time
    )

    notes: Optional[str] = None

    experiment_id: str = ""
    rationale: str = ""

    is_baseline: bool = False

    verdict: Optional[str] = None
    analysis: Optional[str] = None
    next_direction: Optional[str] = None

    diagnostics: Dict[str, Any] = field(
        default_factory=dict
    )

    recovery_events: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    def as_dict(
        self,
    ) -> Dict[str, Any]:

        d = asdict(
            self
        )

        d["failure"] = (
            self.failure.value
        )

        return d

    @staticmethod
    def from_dict(
        d: Dict[str, Any],
    ) -> "IterationRecord":

        d = dict(
            d
        )

        d["metrics"] = Metrics(
            **d.get(
                "metrics",
                {},
            )
        )

        d["resource_usage"] = (
            ResourceUsage(
                **d.get(
                    "resource_usage",
                    {},
                )
            )
        )

        d["failure"] = FailureType(
            d.get(
                "failure",
                "none",
            )
        )

        return IterationRecord(
            **d
        )