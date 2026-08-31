"""
Description: Analyzes research memory, discovered research knowledge, and current-best code to choose the next autonomous research action and repair failed implementations.
Owner: Charlton / David
Input: Current-best code, validation score, research memory, research context, generic research skills, information-action budget, and candidate failures
Output: ResearchAction requesting research, EDA, or one implemented experiment
"""

from src.config import (
    RESEARCHER_MODEL,
)

from src.schemas import (
    EDARequest,
    ExperimentProposal,
    ExperimentSpec,
    ResearchAction,
    ResearchProposal,
    ResearchRequest,
)

from src.prompts.researcher import (
    REPAIR_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    build_repair_prompt,
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
        skills_context: str = "",
        information_actions_used: int = 0,
        information_action_budget: int = 4,
    ) -> ResearchAction:
        """
        Choose whether to gather more evidence or run one experiment.
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
            skills_context=(
                skills_context
            ),
            information_actions_used=(
                information_actions_used
            ),
            information_action_budget=(
                information_action_budget
            ),
        )

        proposal = (
            self.llm_client
            .generate_structured(
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

        decision = (
            proposal.decision
        )

        if (
            information_actions_used
            >= information_action_budget
            and decision.action_type
            != "experiment"
        ):

            raise ValueError(
                "Researcher requested another "
                "information-gathering action "
                "after the information-action "
                "budget was exhausted."
            )

        if isinstance(
            decision,
            ResearchRequest,
        ):

            return ResearchAction(
                action_type="research",
                reason=(
                    decision.reason
                ),
                research_query=(
                    decision.research_query
                ),
                research_source=(
                    decision.research_source
                ),
            )

        if isinstance(
            decision,
            EDARequest,
        ):

            return ResearchAction(
                action_type="eda",
                reason=(
                    decision.reason
                ),
                eda_tool=(
                    decision.eda_tool
                ),
            )

        if not isinstance(
            decision,
            ExperimentProposal,
        ):

            raise TypeError(
                "Unsupported Researcher "
                "decision type."
            )

        spec = ExperimentSpec(
            experiment_id=(
                experiment_id
            ),
            hypothesis=(
                decision.hypothesis
            ),
            rationale=(
                decision.rationale
            ),
            change_type=(
                decision.change_type
            ),
            parameters=(
                decision.parameters
            ),
        )

        return ResearchAction(
            action_type="experiment",
            reason=(
                decision.reason
            ),
            spec=(
                spec
            ),
            full_code=(
                decision.full_code
            ),
        )

    def repair_candidate(
        self,
        spec: ExperimentSpec,
        current_best_code: str,
        candidate_code: str,
        error: str,
        repair_attempt: int,
    ) -> str:
        """
        Repair the implementation of the current hypothesis without
        changing the scientific experiment.
        """

        prompt = build_repair_prompt(
            hypothesis=(
                spec.hypothesis
            ),
            rationale=(
                spec.rationale
            ),
            change_type=(
                spec.change_type
            ),
            parameters=(
                spec.parameters
            ),
            current_best_code=(
                current_best_code
            ),
            candidate_code=(
                candidate_code
            ),
            error=(
                error
            ),
            repair_attempt=(
                repair_attempt
            ),
        )

        repaired = (
            self.llm_client
            .generate_structured(
                system_prompt=(
                    REPAIR_SYSTEM_PROMPT
                ),
                prompt=prompt,
                model=RESEARCHER_MODEL,
                response_schema=(
                    ExperimentProposal
                ),
            )
        )

        return (
            repaired.full_code
        )