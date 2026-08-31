"""
Description: Implements researcher-specified ML experiments and repairs technical implementation failures without changing the hypothesis.
Owner: Charlton / David
Input: ExperimentSpec, current-best code, data-loading context, failed candidate, and implementation error
Output: Complete candidate Python source code
"""

from src.config import (
    RESEARCHER_MODEL,
)

from src.prompts.coder import (
    CODER_REPAIR_SYSTEM_PROMPT,
    CODER_SYSTEM_PROMPT,
    build_coder_prompt,
    build_coder_repair_prompt,
)

from src.schemas import (
    ExperimentSpec,
)

from src.tools.llm_client import (
    GeminiClient,
)


class Coder:

    def __init__(
        self,
        llm_client: GeminiClient,
    ) -> None:

        self.llm_client = (
            llm_client
        )

    def implement(
        self,
        spec: ExperimentSpec,
        current_best_code: str,
        data_context: str,
    ) -> str:
        """
        Implement exactly one researcher-specified experiment.
        """

        prompt = (
            build_coder_prompt(
                experiment_id=(
                    spec.experiment_id
                ),
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
                implementation_instructions=(
                    spec
                    .implementation_instructions
                ),
                current_best_code=(
                    current_best_code
                ),
                data_context=(
                    data_context
                ),
            )
        )

        output = (
            self.llm_client
            .generate_text(
                system_prompt=(
                    CODER_SYSTEM_PROMPT
                ),
                prompt=(
                    prompt
                ),
                model=(
                    RESEARCHER_MODEL
                ),
            )
        )

        return (
            self._clean_code(
                output
            )
        )

    def repair(
        self,
        spec: ExperimentSpec,
        current_best_code: str,
        data_context: str,
        candidate_code: str,
        error: str,
        repair_attempt: int,
    ) -> str:
        """
        Repair the current candidate without changing the experiment.
        """

        prompt = (
            build_coder_repair_prompt(
                experiment_id=(
                    spec.experiment_id
                ),
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
                implementation_instructions=(
                    spec
                    .implementation_instructions
                ),
                current_best_code=(
                    current_best_code
                ),
                data_context=(
                    data_context
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
        )

        output = (
            self.llm_client
            .generate_text(
                system_prompt=(
                    CODER_REPAIR_SYSTEM_PROMPT
                ),
                prompt=(
                    prompt
                ),
                model=(
                    RESEARCHER_MODEL
                ),
            )
        )

        return (
            self._clean_code(
                output
            )
        )

    def _clean_code(
        self,
        output: str,
    ) -> str:
        """
        Remove optional Markdown code fences from coding-model output.
        """

        code = (
            output.strip()
        )

        if code.startswith(
            "```"
        ):

            lines = (
                code.splitlines()
            )

            if lines:

                lines = (
                    lines[1:]
                )

            if (
                lines
                and lines[-1]
                .strip()
                == "```"
            ):

                lines = (
                    lines[:-1]
                )

            code = (
                "\n"
                .join(
                    lines
                )
                .strip()
            )

        return code