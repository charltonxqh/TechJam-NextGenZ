"""
Description: Analyzes research memory and current-best code, then proposes and implements one next ML experiment.
Owner: Charlton / David
Input: Current-best code, validation score, research memory, and research context
Output: ResearchAction containing ExperimentSpec and complete candidate code
"""

from src.config import (
    RESEARCHER_MODEL,
)

from src.schemas import (
    ExperimentSpec,
    ResearchAction,
    ResearchProposal,
)

from src.prompts.researcher import (
    RESEARCHER_SYSTEM_PROMPT,
    build_researcher_prompt,
)

from src.tools.llm_client import (
    GeminiClient,
)


class Researcher:

    def __init__(
        self,
        llm_client: GeminiClient,
    ) -> None:

        self.llm_client = (
            llm_client
        )

    def propose(
        self,
        experiment_id: str,
        memory_context: str,
        current_best_code: str,
        current_best_primary: float,
        baseline_primary: float,
        research_context: str = "",
    ) -> ResearchAction:
        """
        Generate one hypothesis together with the complete resulting
        candidate experiment code.
        """

        prompt = build_researcher_prompt(
            memory_context=(
                memory_context
            ),
            current_best_code=(
                current_best_code
            ),
            current_best_primary=(
                current_best_primary
            ),
            baseline_primary=(
                baseline_primary
            ),
            research_context=(
                research_context
            ),
        )

        proposal = (
            self.llm_client.generate_structured(
                system_prompt=(
                    RESEARCHER_SYSTEM_PROMPT
                ),
                prompt=prompt,
                model=RESEARCHER_MODEL,
                response_schema=(
                    ResearchProposal
                ),
            )
        )

        spec = ExperimentSpec(
            experiment_id=(
                experiment_id
            ),
            hypothesis=(
                proposal.hypothesis
            ),
            rationale=(
                proposal.rationale
            ),
            change_type=(
                proposal.change_type
            ),
            parameters=(
                proposal.parameters
            ),
        )

        return ResearchAction(
            spec=spec,
            full_code=(
                proposal.full_code
            ),
        )