"""
Description: Extracts structured ML research evidence from retrieved online sources.
Owner: Charlton / David
Input: Retrieved web, paper, or documentation content
Output: ResearchEvidence
"""

from pydantic import (
    BaseModel,
    Field,
)

from src.config import (
    LIGHT_MODEL,
)

from src.research_intelligence.research_evidence import (
    ResearchEvidence,
)

from src.tools.llm_client import (
    GeminiClient,
)


class EvidenceExtractionOutput(
    BaseModel
):

    topic: str = Field(
        description=(
            "Short research topic describing "
            "the source's main relevant idea."
        )
    )

    problem_addressed: str = Field(
        description=(
            "What machine-learning or "
            "recommendation problem the "
            "source addresses."
        )
    )

    method: str = Field(
        description=(
            "The main technical method or "
            "approach described by the source."
        )
    )

    assumptions: list[str] = Field(
        default_factory=list,
        description=(
            "Important assumptions required "
            "for the method to be applicable."
        ),
    )

    relevant_findings: str = Field(
        description=(
            "The factual finding or idea from "
            "the source that is relevant to "
            "the current task."
        )
    )

    applicability: str = Field(
        description=(
            "How the source may or may not "
            "apply to the current ML task."
        )
    )

    implementation_hint: str = Field(
        description=(
            "High-level implementation guidance "
            "supported by the source."
        )
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence that the extracted "
            "evidence is supported by the "
            "provided source."
        ),
    )


EVIDENCE_SYSTEM_PROMPT = """
You extract factual machine-learning research evidence from a retrieved source.

You are NOT proposing the next experiment.

Extract only claims supported by the supplied source.

Separate:
- problem addressed
- method
- assumptions
- relevant findings
- applicability to the current task
- implementation hints

The source may be:
- an academic paper abstract
- a technical webpage
- documentation
- a repository page
- a benchmark or competition solution
- an engineering article

Do not invent:
- reported gains
- implementation details
- assumptions
- equations
- experimental results

If the source is only an abstract or snippet, do not pretend that you have
read the full paper.

If relevance or applicability is uncertain, explicitly say so.

The relevant_findings field must contain a meaningful factual research
finding. Do not return an empty string.

Return only valid JSON matching the requested schema.
""".strip()


class EvidenceExtractor:

    def __init__(
        self,
        llm_client: GeminiClient,
    ) -> None:

        self.llm_client = (
            llm_client
        )

    def extract(
        self,
        evidence_id: str,
        source: dict,
        task_context: str,
    ) -> ResearchEvidence:

        content = (
            source.get(
                "content"
            )
            or source.get(
                "summary"
            )
            or source.get(
                "snippet"
            )
            or ""
        )

        if not (
            content
            and content.strip()
        ):

            raise ValueError(
                "Retrieved source contains "
                "no readable content."
            )

        prompt = f"""
CURRENT ML TASK CONTEXT:
{task_context}

SOURCE TYPE:
{source.get("source_type", "unknown")}

SOURCE TITLE:
{source.get("title", "")}

SOURCE URL:
{source.get("url", "")}

SOURCE CONTENT:
<source_content>
{content}
</source_content>

Extract only useful evidence supported by this source.
""".strip()

        output = (
            self.llm_client
            .generate_structured(
                system_prompt=(
                    EVIDENCE_SYSTEM_PROMPT
                ),
                prompt=prompt,
                model=LIGHT_MODEL,
                response_schema=(
                    EvidenceExtractionOutput
                ),
            )
        )

        if not (
            output.relevant_findings
            and output.relevant_findings
            .strip()
        ):

            raise ValueError(
                "Evidence extraction returned "
                "an empty relevant finding."
            )

        return ResearchEvidence(
            evidence_id=(
                evidence_id
            ),
            source_type=(
                source.get(
                    "source_type",
                    "unknown",
                )
            ),
            title=(
                source.get(
                    "title",
                    "",
                )
            ),
            url=(
                source.get(
                    "url",
                    "",
                )
            ),
            topic=(
                output.topic
            ),
            problem_addressed=(
                output.problem_addressed
            ),
            method=(
                output.method
            ),
            assumptions=(
                output.assumptions
            ),
            relevant_findings=(
                output.relevant_findings
            ),
            applicability=(
                output.applicability
            ),
            implementation_hint=(
                output.implementation_hint
            ),
            confidence=(
                output.confidence
            ),
        )