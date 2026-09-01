"""
Description: Semantically verifies that generated candidate code faithfully implements the Researcher's ExperimentSpec before execution.
Owner: Charlton / David
Input: ExperimentSpec, current-best code, candidate code, and data-loading context
Output: Structured implementation-fidelity verification
"""

import difflib

from pydantic import (
    BaseModel,
    Field,
)

from src.config import (
    IMPLEMENTATION_VERIFIER_MODEL,
)

from src.prompts.implementation_verifier import (
    IMPLEMENTATION_VERIFIER_SYSTEM_PROMPT,
    build_implementation_verifier_prompt,
)

from src.schemas import (
    ExperimentSpec,
)

from src.tools.llm_client import (
    GeminiClient,
)


class ImplementationVerification(
    BaseModel
):
    """
    Structured semantic implementation-verification result.
    """

    faithful: bool = Field(
        description=(
            "True only when the candidate "
            "actually implements the requested "
            "ExperimentSpec."
        )
    )

    summary: str = Field(
        description=(
            "Short factual explanation of "
            "whether the implementation matches "
            "the requested experiment."
        )
    )

    issues: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete implementation mismatches "
            "that must be repaired."
        ),
    )


class ImplementationVerifier:

    def __init__(
        self,
        llm_client: GeminiClient,
    ) -> None:

        self.llm_client = (
            llm_client
        )

    def verify(
        self,
        spec: ExperimentSpec,
        current_best_code: str,
        candidate_code: str,
        data_context: str,
    ) -> ImplementationVerification:
        """
        Verify semantic fidelity between the Researcher's experiment
        specification and the Coder's generated candidate.
        """

        code_diff = (
            self._compute_diff(
                current_best_code=(
                    current_best_code
                ),
                candidate_code=(
                    candidate_code
                ),
            )
        )

        prompt = (
            build_implementation_verifier_prompt(
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
                candidate_code=(
                    candidate_code
                ),
                data_context=(
                    data_context
                ),
                code_diff=(
                    code_diff
                ),
            )
        )

        return (
            self.llm_client
            .generate_structured(
                system_prompt=(
                    IMPLEMENTATION_VERIFIER_SYSTEM_PROMPT
                ),
                prompt=(
                    prompt
                ),
                model=(
                    IMPLEMENTATION_VERIFIER_MODEL
                ),
                response_schema=(
                    ImplementationVerification
                ),
            )
        )

    def format_failure(
        self,
        verification: ImplementationVerification,
    ) -> str:
        """
        Format verification feedback for the Coder repair prompt.
        """

        lines = [
            (
                "[IMPLEMENTATION_VERIFICATION_FAILED]"
            ),
            "",
            verification.summary,
        ]

        if verification.issues:

            lines.extend(
                [
                    "",
                    "Issues:",
                ]
            )

            lines.extend(
                (
                    f"- {issue}"
                )
                for issue
                in verification.issues
            )

        return "\n".join(
            lines
        )

    def _compute_diff(
        self,
        current_best_code: str,
        candidate_code: str,
    ) -> str:
        """
        Compute a unified diff to make semantic verification easier.
        """

        return "".join(
            difflib.unified_diff(
                current_best_code.splitlines(
                    keepends=True
                ),
                candidate_code.splitlines(
                    keepends=True
                ),
                fromfile=(
                    "current_best"
                ),
                tofile=(
                    "candidate"
                ),
            )
        )