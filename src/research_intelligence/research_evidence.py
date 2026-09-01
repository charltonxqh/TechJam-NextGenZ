"""
Description: Defines structured evidence extracted from online ML research.
Owner: Charlton / David
Input: Retrieved research source
Output: Normalized ResearchEvidence record
"""

from dataclasses import (
    asdict,
    dataclass,
    field,
)


@dataclass
class ResearchEvidence:
    evidence_id: str

    source_type: str
    title: str
    url: str

    topic: str

    problem_addressed: str
    method: str

    assumptions: list[str] = field(
        default_factory=list
    )

    relevant_findings: str = ""
    applicability: str = ""
    implementation_hint: str = ""

    confidence: float = 0.5

    def as_dict(
        self,
    ) -> dict:

        return asdict(
            self
        )