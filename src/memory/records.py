"""
records.py
----------
Defines the data structures for a single iteration of the ML agent's
improvement loop. This is the "unit of memory" - everything else in the
memory module operates on lists of these.

Kept deliberately independent of any specific agent architecture: whoever
builds the main loop just needs to construct one of these per iteration
and hand it to MemoryStore.add().
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from enum import Enum
import time


class FailureType(str, Enum):
    """Taxonomy of failure modes, used for pattern-spotting across a run."""
    NONE = "none"                  # iteration succeeded
    CODE_ERROR = "code_error"      # exception / traceback
    TIMEOUT = "timeout"            # exceeded time budget
    BAD_OUTPUT = "bad_output"      # ran, but produced invalid/unusable output
    NO_IMPROVEMENT = "no_improvement"  # ran fine, but score did not improve
    OTHER = "other"


@dataclass
class Metrics:
    """Scores for one iteration. Matches the challenge's GAUC / nDCG@5 / primary."""
    gauc: Optional[float] = None
    ndcg5: Optional[float] = None
    primary: Optional[float] = None  # mean(gauc, ndcg5) - the number convergence is checked on

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceUsage:
    """Cost tracking, needed for the Feasibility & Practicality deliverable."""
    input_tokens: int = 0
    output_tokens: int = 0
    wall_clock_sec: float = 0.0
    gpu_hours: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IterationRecord:
    """
    One full round of the agent's loop:
    hypothesis -> code change -> run -> result.

    This is intentionally a flat, serializable structure (dataclass ->
    dict -> JSON) so it can be written to disk, sent over an API, or
    displayed in a UI without any custom logic.
    """
    iteration: int
    hypothesis: str                     # what the agent intended to try and why
    stage: str                          # which pipeline stage this targets:
                                         # "data" | "features" | "model" | "training" | "eval"
    code_diff: str                      # the actual code change applied (or full code if simpler)
    code_summary: str = ""              # short LLM-generated summary of what the code does
    likely_reason: str = ""             # LLM's guess at why the score moved this way
    metrics: Metrics = field(default_factory=Metrics)
    failure: FailureType = FailureType.NONE
    error_message: Optional[str] = None
    manual_intervention: bool = False   # did a human have to step in this round?
    resource_usage: ResourceUsage = field(default_factory=ResourceUsage)
    timestamp: float = field(default_factory=time.time)
    notes: Optional[str] = None         # free-text reflection, e.g. "overfit, dropped 0.01"

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["failure"] = self.failure.value  # enum -> plain string for JSON
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "IterationRecord":
        d = dict(d)
        d["metrics"] = Metrics(**d.get("metrics", {}))
        d["resource_usage"] = ResourceUsage(**d.get("resource_usage", {}))
        d["failure"] = FailureType(d.get("failure", "none"))
        return IterationRecord(**d)
