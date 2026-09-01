"""
Description: Analyzes research memory, discovered research knowledge, skill metadata, and current-best code to choose the next autonomous research action.
Owner: Charlton / David
Input: Current-best code, validation score, research memory, research context, skill catalog, loaded skills, information-action state, research requirements, evidence-sufficiency state, and allowed actions
Output: ResearchAction requesting research, EDA, skill loading, or one scientific ExperimentSpec
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
    SkillRequest,
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
        skill_catalog: str = "",
        loaded_skills_context: str = "",
        loaded_skill_names: list[str] | None = None,
        information_actions_used: int = 0,
        information_action_budget: int = 4,
        research_actions_this_iteration: int = 0,
        require_external_research: bool = False,
        research_requirement_reason: str = "",
        allowed_actions: list[str] | None = None,
        completed_eda_tools: list[str] | None = None,
        attempted_research_queries: list[str] | None = None,
        evidence_sufficiency_checkpoint: bool = False,
    ) -> ResearchAction:
        """
        Choose whether to gather evidence, load procedural guidance,
        or propose one experiment.
        """

        prompt = (
            build_researcher_prompt(
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
                skill_catalog=(
                    skill_catalog
                ),
                loaded_skills_context=(
                    loaded_skills_context
                ),
                loaded_skill_names=(
                    loaded_skill_names
                ),
                information_actions_used=(
                    information_actions_used
                ),
                information_action_budget=(
                    information_action_budget
                ),
                research_actions_this_iteration=(
                    research_actions_this_iteration
                ),
                require_external_research=(
                    require_external_research
                ),
                research_requirement_reason=(
                    research_requirement_reason
                ),
                allowed_actions=(
                    allowed_actions
                ),
                completed_eda_tools=(
                    completed_eda_tools
                ),
                attempted_research_queries=(
                    attempted_research_queries
                ),
                evidence_sufficiency_checkpoint=(
                    evidence_sufficiency_checkpoint
                ),
            )
        )

        proposal = (
            self.llm_client
            .generate_structured(
                system_prompt=(
                    RESEARCHER_SYSTEM_PROMPT
                ),
                prompt=(
                    prompt
                ),
                model=(
                    RESEARCHER_MODEL
                ),
                response_schema=(
                    ResearchProposal
                ),
            )
        )

        decision = (
            proposal.decision
        )

        if isinstance(
            decision,
            ResearchRequest,
        ):

            return ResearchAction(
                action_type=(
                    "research"
                ),
                reason=(
                    decision.reason
                ),
                knowledge_gap=(
                    decision.knowledge_gap
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
                action_type=(
                    "eda"
                ),
                reason=(
                    decision.reason
                ),
                eda_tool=(
                    decision.eda_tool
                ),
            )

        if isinstance(
            decision,
            SkillRequest,
        ):

            return ResearchAction(
                action_type=(
                    "load_skill"
                ),
                reason=(
                    decision.reason
                ),
                skills=(
                    decision.skills
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

        spec = (
            ExperimentSpec(
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
                implementation_instructions=(
                    decision
                    .implementation_instructions
                ),
            )
        )

        return ResearchAction(
            action_type=(
                "experiment"
            ),
            reason=(
                decision.reason
            ),
            spec=(
                spec
            ),
        )