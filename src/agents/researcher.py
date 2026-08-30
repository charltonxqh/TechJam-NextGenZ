"""
Description: Analyzes previous research and proposes the next ML hypothesis and experiment to investigate.
Owner: Charlton / David
Input: Research state, experiment history, and available research context
Output: ExperimentSpec
"""

from dataclasses import asdict

from src.config import RESEARCHER_MODEL
from src.schemas import (
    ExperimentResult,
    ExperimentSpec,
    ResearchProposal,
)
from src.prompts.researcher import (
    RESEARCHER_SYSTEM_PROMPT,
    build_researcher_prompt,
)
from src.tools.llm_client import GeminiClient


class Researcher:

    def __init__(
        self,
        llm_client: GeminiClient,
    ) -> None:
        self.llm_client = llm_client

    def propose(
        self,
        experiment_id: str,
        baseline_result: ExperimentResult,
        history: list[dict],
        research_context: str = "",
    ) -> ExperimentSpec:
        """
        Generate one next research hypothesis.
        """

        prompt = build_researcher_prompt(
            baseline_result=asdict(
                baseline_result
            ),
            history=history,
            research_context=research_context,
        )

        proposal = self.llm_client.generate_structured(
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            prompt=prompt,
            model=RESEARCHER_MODEL,
            response_schema=ResearchProposal,
        )

        return ExperimentSpec(
            experiment_id=experiment_id,
            hypothesis=proposal.hypothesis,
            rationale=proposal.rationale,
            change_type=proposal.change_type,
            parameters=proposal.parameters,
        )