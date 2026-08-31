"""
Description: Deterministically validates generated experiment code before execution.
Owner: Charlton / David
Input: Complete candidate Python source code
Output: CandidateValidationResult
"""

import ast
import re

from dataclasses import dataclass, field


@dataclass
class CandidateValidationResult:
    valid: bool

    errors: list[str] = field(
        default_factory=list
    )

    def format_errors(
        self,
    ) -> str:

        if self.valid:

            return (
                "Candidate validation passed."
            )

        return "\n".join(
            f"- {error}"
            for error
            in self.errors
        )


PLACEHOLDER_PATTERNS = [
    (
        r"\.\.\.",
        "Candidate contains an ellipsis placeholder (...).",
    ),
    (
        r"for brevity",
        "Candidate contains placeholder wording: 'for brevity'.",
    ),
    (
        r"rest of (?:the )?(?:logic|code)",
        "Candidate refers to omitted 'rest of logic/code'.",
    ),
    (
        r"same as current_best",
        "Candidate refers to omitted current-best code instead of implementing it.",
    ),
    (
        r"same as (?:the )?baseline",
        "Candidate refers to omitted baseline code instead of implementing it.",
    ),
    (
        r"placeholder",
        "Candidate contains placeholder implementation text.",
    ),
    (
        r"implementation omitted",
        "Candidate explicitly omits implementation.",
    ),
    (
        r"backprop omitted",
        "Candidate explicitly omits backpropagation implementation.",
    ),
    (
        r"to be implemented",
        "Candidate contains unfinished implementation text.",
    ),
    (
        r"\bTODO\b",
        "Candidate contains an unresolved TODO.",
    ),
]


REQUIRED_TEXT = {
    "--split": (
        "Candidate must preserve the --split CLI argument."
    ),
    "valid": (
        "Candidate must support validation mode."
    ),
    "test": (
        "Candidate must support final test mode."
    ),
    "GAUC": (
        "Candidate must print GAUC."
    ),
    "nDCG@5": (
        "Candidate must print nDCG@5."
    ),
    "primary": (
        "Candidate must print Primary."
    ),
    "__main__": (
        "Candidate must contain a runnable __main__ entry point."
    ),
}


def _check_syntax(
    code: str,
) -> list[str]:

    try:

        tree = ast.parse(
            code
        )

    except SyntaxError as error:

        location = ""

        if error.lineno is not None:

            location = (
                f" at line "
                f"{error.lineno}"
            )

        return [
            (
                "Python syntax error"
                f"{location}: "
                f"{error.msg}"
            )
        ]

    errors = []

    for node in ast.walk(
        tree
    ):

        if isinstance(
            node,
            ast.Pass,
        ):

            errors.append(
                (
                    "Candidate contains a bare "
                    "'pass' statement, which may "
                    "indicate incomplete generated code."
                )
            )

    return errors


def _check_placeholders(
    code: str,
) -> list[str]:

    errors = []

    lowered = (
        code.lower()
    )

    for (
        pattern,
        message,
    ) in PLACEHOLDER_PATTERNS:

        if re.search(
            pattern,
            lowered,
            flags=re.IGNORECASE,
        ):

            errors.append(
                message
            )

    return errors


def _check_required_structure(
    code: str,
) -> list[str]:

    errors = []

    for (
        required_text,
        message,
    ) in REQUIRED_TEXT.items():

        if (
            required_text
            not in code
        ):

            errors.append(
                message
            )

    return errors


def validate_candidate(
    code: str,
) -> CandidateValidationResult:
    """
    Validate a generated experiment before materializing or executing it.

    The validator checks only deterministic implementation requirements.
    It does not judge whether the scientific hypothesis is good.
    """

    errors = []

    if not (
        code
        and code.strip()
    ):

        errors.append(
            "Candidate code is empty."
        )

        return (
            CandidateValidationResult(
                valid=False,
                errors=errors,
            )
        )

    errors.extend(
        _check_syntax(
            code
        )
    )

    errors.extend(
        _check_placeholders(
            code
        )
    )

    errors.extend(
        _check_required_structure(
            code
        )
    )

    return CandidateValidationResult(
        valid=(
            len(errors)
            == 0
        ),
        errors=errors,
    )