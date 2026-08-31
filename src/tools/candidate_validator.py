"""
Description: Deterministically validates generated experiment code before execution.
Owner: Charlton / David
Input: Complete candidate Python source code
Output: CandidateValidationResult
"""

import ast
import re

from dataclasses import (
    dataclass,
    field,
)


@dataclass
class ValidationIssue:
    code: str
    message: str

    line: int | None = None
    column: int | None = None
    snippet: str | None = None


@dataclass
class CandidateValidationResult:
    valid: bool

    issues: list[
        ValidationIssue
    ] = field(
        default_factory=list
    )

    @property
    def errors(
        self,
    ) -> list[str]:
        """
        Preserve compatibility with callers that expect an errors list.
        """

        return [
            issue.message
            for issue
            in self.issues
        ]

    def format_errors(
        self,
    ) -> str:

        if self.valid:

            return (
                "Candidate validation passed."
            )

        formatted = []

        for issue in self.issues:

            location = ""

            if issue.line is not None:

                location = (
                    f"Line "
                    f"{issue.line}"
                )

                if (
                    issue.column
                    is not None
                ):

                    location += (
                        f", column "
                        f"{issue.column}"
                    )

                location += ":\n"

            snippet = ""

            if issue.snippet:

                snippet = (
                    f"    "
                    f"{issue.snippet}\n"
                )

            formatted.append(
                (
                    f"[{issue.code}]\n"
                    f"{location}"
                    f"{snippet}"
                    f"{issue.message}"
                )
            )

        return "\n\n".join(
            formatted
        )


PLACEHOLDER_PATTERNS = [
    (
        r"\bfor brevity\b",
        "Candidate contains placeholder wording: 'for brevity'.",
    ),
    (
        r"\brest of (?:the )?(?:logic|code)\b",
        "Candidate refers to omitted 'rest of logic/code'.",
    ),
    (
        r"\bsame as current_best(?:_code)?\b",
        "Candidate refers to omitted current-best code instead of implementing it.",
    ),
    (
        r"\bsame as (?:the )?baseline\b",
        "Candidate refers to omitted baseline code instead of implementing it.",
    ),
    (
        r"\bimplementation omitted\b",
        "Candidate explicitly omits implementation.",
    ),
    (
        r"\bbackprop(?:agation)? omitted\b",
        "Candidate explicitly omits backpropagation implementation.",
    ),
    (
        r"\bto be implemented\b",
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


def _get_line(
    code_lines: list[str],
    line_number: int | None,
) -> str | None:

    if line_number is None:

        return None

    index = (
        line_number
        - 1
    )

    if (
        index < 0
        or index
        >= len(
            code_lines
        )
    ):

        return None

    return (
        code_lines[
            index
        ]
        .strip()
    )


def _check_syntax(
    code: str,
) -> tuple[
    ast.AST | None,
    list[
        ValidationIssue
    ],
]:

    try:

        tree = (
            ast.parse(
                code
            )
        )

    except SyntaxError as error:

        code_lines = (
            code.splitlines()
        )

        return (
            None,
            [
                ValidationIssue(
                    code=(
                        "SYNTAX_ERROR"
                    ),
                    message=(
                        error.msg
                    ),
                    line=(
                        error.lineno
                    ),
                    column=(
                        error.offset
                    ),
                    snippet=(
                        _get_line(
                            code_lines,
                            error.lineno,
                        )
                    ),
                )
            ],
        )

    return (
        tree,
        [],
    )


def _check_ast_placeholders(
    tree: ast.AST,
    code: str,
) -> list[
    ValidationIssue
]:

    issues = []

    code_lines = (
        code.splitlines()
    )

    for node in ast.walk(
        tree
    ):

        if (
            isinstance(
                node,
                ast.Expr,
            )
            and isinstance(
                node.value,
                ast.Constant,
            )
            and node.value.value
            is Ellipsis
        ):

            issues.append(
                ValidationIssue(
                    code=(
                        "INCOMPLETE_CODE"
                    ),
                    message=(
                        "Candidate contains an "
                        "Ellipsis expression used "
                        "as executable placeholder "
                        "code."
                    ),
                    line=(
                        getattr(
                            node,
                            "lineno",
                            None,
                        )
                    ),
                    column=(
                        getattr(
                            node,
                            "col_offset",
                            None,
                        )
                    ),
                    snippet=(
                        _get_line(
                            code_lines,
                            getattr(
                                node,
                                "lineno",
                                None,
                            ),
                        )
                    ),
                )
            )

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):

            continue

        if (
            len(
                node.body
            )
            == 1
            and isinstance(
                node.body[0],
                ast.Pass,
            )
        ):

            pass_node = (
                node.body[0]
            )

            issues.append(
                ValidationIssue(
                    code=(
                        "INCOMPLETE_CODE"
                    ),
                    message=(
                        "Candidate contains a "
                        "function or class whose "
                        "entire implementation is "
                        "'pass'."
                    ),
                    line=(
                        getattr(
                            pass_node,
                            "lineno",
                            None,
                        )
                    ),
                    column=(
                        getattr(
                            pass_node,
                            "col_offset",
                            None,
                        )
                    ),
                    snippet=(
                        _get_line(
                            code_lines,
                            getattr(
                                pass_node,
                                "lineno",
                                None,
                            ),
                        )
                    ),
                )
            )

    return issues


def _check_placeholder_text(
    code: str,
) -> list[
    ValidationIssue
]:

    issues = []

    code_lines = (
        code.splitlines()
    )

    for (
        line_number,
        line,
    ) in enumerate(
        code_lines,
        start=1,
    ):

        for (
            pattern,
            message,
        ) in PLACEHOLDER_PATTERNS:

            match = re.search(
                pattern,
                line,
                flags=(
                    re.IGNORECASE
                ),
            )

            if not match:

                continue

            issues.append(
                ValidationIssue(
                    code=(
                        "PLACEHOLDER_TEXT"
                    ),
                    message=(
                        message
                    ),
                    line=(
                        line_number
                    ),
                    column=(
                        match.start()
                    ),
                    snippet=(
                        line.strip()
                    ),
                )
            )

    return issues


def _check_required_structure(
    code: str,
) -> list[
    ValidationIssue
]:

    issues = []

    for (
        required_text,
        message,
    ) in REQUIRED_TEXT.items():

        if (
            required_text
            not in code
        ):

            issues.append(
                ValidationIssue(
                    code=(
                        "MISSING_STRUCTURE"
                    ),
                    message=(
                        message
                    ),
                )
            )

    return issues


def validate_candidate(
    code: str,
) -> CandidateValidationResult:
    """
    Validate a generated experiment before materializing or executing it.

    The validator checks only deterministic implementation requirements.
    It does not judge whether the scientific hypothesis is good.
    """

    issues = []

    if not (
        code
        and code.strip()
    ):

        issues.append(
            ValidationIssue(
                code=(
                    "EMPTY_CODE"
                ),
                message=(
                    "Candidate code is empty."
                ),
            )
        )

        return (
            CandidateValidationResult(
                valid=False,
                issues=issues,
            )
        )

    tree, syntax_issues = (
        _check_syntax(
            code
        )
    )

    issues.extend(
        syntax_issues
    )

    if tree is not None:

        issues.extend(
            _check_ast_placeholders(
                tree=(
                    tree
                ),
                code=(
                    code
                ),
            )
        )

    issues.extend(
        _check_placeholder_text(
            code
        )
    )

    issues.extend(
        _check_required_structure(
            code
        )
    )

    return (
        CandidateValidationResult(
            valid=(
                len(
                    issues
                )
                == 0
            ),
            issues=(
                issues
            ),
        )
    )