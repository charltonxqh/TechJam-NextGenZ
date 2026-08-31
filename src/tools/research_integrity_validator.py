"""
Description: Deterministically validates candidate experiments for target leakage, same-impression outcome leakage, and protected benchmark-protocol violations before execution.
Owner: Charlton / David
Input: Complete candidate Python source code
Output: ResearchIntegrityValidationResult
"""

import ast

from dataclasses import (
    dataclass,
    field,
)


TARGET_FIELD = (
    "long_view"
)


SAME_IMPRESSION_OUTCOME_FIELDS = {
    "is_click",
    "is_like",
    "is_comment",
    "is_follow",
    "is_forward",
    "is_hate",
    "play_time_ms",
    "play_time",
    "time_ms",
    "profile_stay_time",
}


PROTECTED_SPLITS = {
    "train": (
        20220408,
        20220421,
    ),
    "valid": (
        20220422,
        20220428,
    ),
    "test": (
        20220429,
        20220508,
    ),
}


FEATURE_DECLARATION_NAMES = {
    "fields",
    "features",
    "feature_fields",
    "feature_columns",
    "input_fields",
    "input_features",
    "categorical_fields",
    "categorical_features",
    "numeric_fields",
    "numeric_features",
    "model_fields",
    "model_features",
}


FEATURE_BUILDER_NAMES = {
    "raw",
    "feature",
    "features",
    "build_feature",
    "build_features",
    "make_feature",
    "make_features",
    "encode_feature",
    "encode_features",
    "transform_feature",
    "transform_features",
}


@dataclass
class ResearchIntegrityIssue:

    code: str
    message: str

    line: int | None = None
    column: int | None = None
    snippet: str | None = None


@dataclass
class ResearchIntegrityValidationResult:

    valid: bool

    issues: list[
        ResearchIntegrityIssue
    ] = field(
        default_factory=list
    )

    def format_errors(
        self,
    ) -> str:

        if self.valid:

            return (
                "Research integrity "
                "validation passed."
            )

        formatted = []

        for issue in self.issues:

            location = ""

            if (
                issue.line
                is not None
            ):

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


def _target_names(
    target: ast.AST,
) -> list[str]:

    names = []

    if isinstance(
        target,
        ast.Name,
    ):

        names.append(
            target.id
        )

    elif isinstance(
        target,
        (
            ast.Tuple,
            ast.List,
        ),
    ):

        for element in target.elts:

            names.extend(
                _target_names(
                    element
                )
            )

    elif isinstance(
        target,
        ast.Subscript,
    ):

        names.extend(
            _target_names(
                target.value
            )
        )

    elif isinstance(
        target,
        ast.Attribute,
    ):

        names.append(
            target.attr
        )

    return names


def _extract_string_literals(
    node: ast.AST,
) -> set[str]:

    values = set()

    for child in ast.walk(
        node
    ):

        if (
            isinstance(
                child,
                ast.Constant,
            )
            and isinstance(
                child.value,
                str,
            )
        ):

            values.add(
                child.value
            )

    return values


def _is_feature_declaration_name(
    name: str,
) -> bool:

    normalized = (
        name.lower()
    )

    if (
        normalized
        in FEATURE_DECLARATION_NAMES
    ):

        return True

    if (
        normalized.endswith(
            "_fields"
        )
        or normalized.endswith(
            "_features"
        )
        or normalized.endswith(
            "_feature_columns"
        )
    ):

        return True

    return False


def _check_feature_declarations(
    tree: ast.AST,
    code: str,
) -> list[
    ResearchIntegrityIssue
]:

    issues = []

    code_lines = (
        code.splitlines()
    )

    forbidden = (
        SAME_IMPRESSION_OUTCOME_FIELDS
        | {
            TARGET_FIELD
        }
    )

    for node in ast.walk(
        tree
    ):

        targets = []

        value = None

        if isinstance(
            node,
            ast.Assign,
        ):

            targets = (
                node.targets
            )

            value = (
                node.value
            )

        elif isinstance(
            node,
            ast.AnnAssign,
        ):

            targets = [
                node.target
            ]

            value = (
                node.value
            )

        else:

            continue

        if value is None:

            continue

        assignment_names = []

        for target in targets:

            assignment_names.extend(
                _target_names(
                    target
                )
            )

        if not any(
            _is_feature_declaration_name(
                name
            )
            for name
            in assignment_names
        ):

            continue

        strings = (
            _extract_string_literals(
                value
            )
        )

        leaked_fields = sorted(
            strings
            & forbidden
        )

        for leaked_field in leaked_fields:

            if (
                leaked_field
                == TARGET_FIELD
            ):

                issue_code = (
                    "TARGET_AS_INPUT"
                )

                message = (
                    "The target field "
                    f"'{TARGET_FIELD}' is "
                    "declared as a model input "
                    "feature. The target may be "
                    "used as a training/evaluation "
                    "label but must never be a "
                    "prediction-time input."
                )

            else:

                issue_code = (
                    "SAME_IMPRESSION_OUTCOME_LEAKAGE"
                )

                message = (
                    f"'{leaked_field}' is a "
                    "same-impression behavioral "
                    "outcome and is declared as "
                    "a model input feature. "
                    "Actual feedback from the row "
                    "being scored is unavailable "
                    "at ranking time and must not "
                    "be used as input. Historical "
                    "training-only aggregates or "
                    "training-only auxiliary "
                    "targets remain allowed."
                )

            issues.append(
                ResearchIntegrityIssue(
                    code=(
                        issue_code
                    ),
                    message=(
                        message
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

    return issues


def _infer_loaded_row_schema(
    tree: ast.AST,
) -> dict[
    int,
    str,
]:
    """
    Infer tuple positions created by rows.append((...)).

    Example:

    rows.append((
        int(r["date"]),
        r["user_id"],
        ...
        1 if r["long_view"] != "0" else 0,
        1 if r["is_click"] != "0" else 0,
    ))

    becomes approximately:

    {
        1: "user_id",
        2: "video_id",
        6: "long_view",
        7: "is_click",
    }
    """

    mapping = {}

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.Call,
        ):

            continue

        if not isinstance(
            node.func,
            ast.Attribute,
        ):

            continue

        if (
            node.func.attr
            != "append"
        ):

            continue

        if (
            not node.args
            or not isinstance(
                node.args[0],
                (
                    ast.Tuple,
                    ast.List,
                ),
            )
        ):

            continue

        collection = (
            node.args[0]
        )

        for index, element in enumerate(
            collection.elts
        ):

            strings = (
                _extract_string_literals(
                    element
                )
            )

            relevant = (
                strings
                & (
                    SAME_IMPRESSION_OUTCOME_FIELDS
                    | {
                        TARGET_FIELD,
                        "date",
                        "user_id",
                        "video_id",
                        "author_id",
                        "tab",
                        "duration_ms",
                    }
                )
            )

            if (
                len(
                    relevant
                )
                == 1
            ):

                mapping[
                    index
                ] = (
                    next(
                        iter(
                            relevant
                        )
                    )
                )

    return mapping


def _subscript_constant_index(
    node: ast.Subscript,
) -> int | None:

    slice_node = (
        node.slice
    )

    if (
        isinstance(
            slice_node,
            ast.Constant,
        )
        and isinstance(
            slice_node.value,
            int,
        )
    ):

        return (
            slice_node.value
        )

    return None


def _subscript_string_key(
    node: ast.Subscript,
) -> str | None:

    slice_node = (
        node.slice
    )

    if (
        isinstance(
            slice_node,
            ast.Constant,
        )
        and isinstance(
            slice_node.value,
            str,
        )
    ):

        return (
            slice_node.value
        )

    return None


def _check_feature_builder_functions(
    tree: ast.AST,
    code: str,
    row_schema: dict[
        int,
        str,
    ],
) -> list[
    ResearchIntegrityIssue
]:

    issues = []

    code_lines = (
        code.splitlines()
    )

    forbidden = (
        SAME_IMPRESSION_OUTCOME_FIELDS
        | {
            TARGET_FIELD
        }
    )

    for function in ast.walk(
        tree
    ):

        if not isinstance(
            function,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):

            continue

        function_name = (
            function.name.lower()
        )

        if (
            function_name
            not in FEATURE_BUILDER_NAMES
            and "feature"
            not in function_name
        ):

            continue

        for node in ast.walk(
            function
        ):

            if not isinstance(
                node,
                ast.Return,
            ):

                continue

            if node.value is None:

                continue

            for child in ast.walk(
                node.value
            ):

                if not isinstance(
                    child,
                    ast.Subscript,
                ):

                    continue

                string_key = (
                    _subscript_string_key(
                        child
                    )
                )

                leaked_field = None

                if (
                    string_key
                    in forbidden
                ):

                    leaked_field = (
                        string_key
                    )

                numeric_index = (
                    _subscript_constant_index(
                        child
                    )
                )

                if (
                    leaked_field
                    is None
                    and numeric_index
                    is not None
                ):

                    mapped_field = (
                        row_schema.get(
                            numeric_index
                        )
                    )

                    if (
                        mapped_field
                        in forbidden
                    ):

                        leaked_field = (
                            mapped_field
                        )

                if leaked_field is None:

                    continue

                if (
                    leaked_field
                    == TARGET_FIELD
                ):

                    issue_code = (
                        "TARGET_AS_INPUT"
                    )

                    message = (
                        f"Feature-building function "
                        f"'{function.name}' includes "
                        f"the target '{TARGET_FIELD}' "
                        "in its returned model "
                        "features."
                    )

                else:

                    issue_code = (
                        "SAME_IMPRESSION_OUTCOME_LEAKAGE"
                    )

                    message = (
                        f"Feature-building function "
                        f"'{function.name}' includes "
                        f"same-impression outcome "
                        f"'{leaked_field}' in model "
                        "features. This outcome is "
                        "not available when the "
                        "impression is ranked."
                    )

                issues.append(
                    ResearchIntegrityIssue(
                        code=(
                            issue_code
                        ),
                        message=(
                            message
                        ),
                        line=(
                            getattr(
                                child,
                                "lineno",
                                None,
                            )
                        ),
                        column=(
                            getattr(
                                child,
                                "col_offset",
                                None,
                            )
                        ),
                        snippet=(
                            _get_line(
                                code_lines,
                                getattr(
                                    child,
                                    "lineno",
                                    None,
                                ),
                            )
                        ),
                    )
                )

    return issues


def _contains_forbidden_row_value(
    node: ast.AST,
    row_schema: dict[
        int,
        str,
    ],
) -> str | None:

    forbidden = (
        SAME_IMPRESSION_OUTCOME_FIELDS
        | {
            TARGET_FIELD
        }
    )

    for child in ast.walk(
        node
    ):

        if not isinstance(
            child,
            ast.Subscript,
        ):

            continue

        string_key = (
            _subscript_string_key(
                child
            )
        )

        if (
            string_key
            in forbidden
        ):

            return (
                string_key
            )

        numeric_index = (
            _subscript_constant_index(
                child
            )
        )

        if (
            numeric_index
            is not None
        ):

            mapped_field = (
                row_schema.get(
                    numeric_index
                )
            )

            if (
                mapped_field
                in forbidden
            ):

                return (
                    mapped_field
                )

    return None


def _looks_like_model_input_name(
    name: str,
) -> bool:

    lowered = (
        name.lower()
    )

    if lowered in {
        "x",
        "xtr",
        "xva",
        "xte",
        "x_train",
        "x_valid",
        "x_validation",
        "x_test",
    }:

        return True

    if (
        lowered.startswith(
            "x_"
        )
        or lowered.endswith(
            "_features"
        )
        or lowered.endswith(
            "_inputs"
        )
        or "model_input"
        in lowered
    ):

        return True

    return False


def _check_direct_model_input_assignments(
    tree: ast.AST,
    code: str,
    row_schema: dict[
        int,
        str,
    ],
) -> list[
    ResearchIntegrityIssue
]:

    issues = []

    code_lines = (
        code.splitlines()
    )

    for node in ast.walk(
        tree
    ):

        targets = []

        value = None

        if isinstance(
            node,
            ast.Assign,
        ):

            targets = (
                node.targets
            )

            value = (
                node.value
            )

        elif isinstance(
            node,
            ast.AnnAssign,
        ):

            targets = [
                node.target
            ]

            value = (
                node.value
            )

        else:

            continue

        if value is None:

            continue

        names = []

        for target in targets:

            names.extend(
                _target_names(
                    target
                )
            )

        if not any(
            _looks_like_model_input_name(
                name
            )
            for name
            in names
        ):

            continue

        leaked_field = (
            _contains_forbidden_row_value(
                value,
                row_schema,
            )
        )

        if leaked_field is None:

            continue

        if (
            leaked_field
            == TARGET_FIELD
        ):

            issue_code = (
                "TARGET_AS_INPUT"
            )

            message = (
                f"The target "
                f"'{TARGET_FIELD}' is used "
                "while constructing a model "
                "input array."
            )

        else:

            issue_code = (
                "SAME_IMPRESSION_OUTCOME_LEAKAGE"
            )

            message = (
                f"Same-impression outcome "
                f"'{leaked_field}' is used "
                "while constructing a model "
                "input array. Same-row "
                "behavioral outcomes cannot "
                "be prediction-time features."
            )

        issues.append(
            ResearchIntegrityIssue(
                code=(
                    issue_code
                ),
                message=(
                    message
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

    return issues


def _check_official_splits(
    tree: ast.AST,
    code: str,
) -> list[
    ResearchIntegrityIssue
]:

    issues = []

    code_lines = (
        code.splitlines()
    )

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.Assign,
        ):

            continue

        target_names = []

        for target in node.targets:

            target_names.extend(
                _target_names(
                    target
                )
            )

        if (
            "SPLITS"
            not in target_names
        ):

            continue

        try:

            value = (
                ast.literal_eval(
                    node.value
                )
            )

        except Exception:

            continue

        if (
            value
            != PROTECTED_SPLITS
        ):

            issues.append(
                ResearchIntegrityIssue(
                    code=(
                        "DATA_SPLIT_MODIFIED"
                    ),
                    message=(
                        "Candidate modifies the "
                        "official train/validation/"
                        "test date split. The "
                        "benchmark split is fixed "
                        "and must remain unchanged."
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

    return issues


def _check_evaluate_override(
    tree: ast.AST,
    code: str,
) -> list[
    ResearchIntegrityIssue
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
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            and node.name
            == "evaluate"
        ):

            issues.append(
                ResearchIntegrityIssue(
                    code=(
                        "EVALUATION_OVERRIDE"
                    ),
                    message=(
                        "Candidate defines its own "
                        "'evaluate' function. The "
                        "official evaluate.py "
                        "implementation and metric "
                        "definitions are protected "
                        "and must be used unchanged."
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

    return issues


def _deduplicate_issues(
    issues: list[
        ResearchIntegrityIssue
    ],
) -> list[
    ResearchIntegrityIssue
]:

    output = []

    seen = set()

    for issue in issues:

        key = (
            issue.code,
            issue.line,
            issue.message,
        )

        if key in seen:

            continue

        seen.add(
            key
        )

        output.append(
            issue
        )

    return output


def validate_research_integrity(
    code: str,
) -> ResearchIntegrityValidationResult:
    """
    Validate one candidate against hard research-integrity constraints.

    Allowed examples:
    - training-only is_click auxiliary target
    - historical click/like aggregates computed from earlier training data
    - user/item/content/context features available before ranking

    Blocked examples:
    - same-row is_click/is_like/etc. as model input
    - same-row play_time_ms as model input
    - long_view as model input
    - modified official date splits
    - replacement of the official evaluator
    """

    if not (
        code
        and code.strip()
    ):

        return (
            ResearchIntegrityValidationResult(
                valid=False,
                issues=[
                    ResearchIntegrityIssue(
                        code=(
                            "EMPTY_CODE"
                        ),
                        message=(
                            "Candidate code is empty."
                        ),
                    )
                ],
            )
        )

    try:

        tree = (
            ast.parse(
                code
            )
        )

    except SyntaxError as error:

        return (
            ResearchIntegrityValidationResult(
                valid=False,
                issues=[
                    ResearchIntegrityIssue(
                        code=(
                            "SYNTAX_ERROR"
                        ),
                        message=(
                            "Research-integrity "
                            "validation could not "
                            "run because the "
                            "candidate is not valid "
                            "Python."
                        ),
                        line=(
                            error.lineno
                        ),
                        column=(
                            error.offset
                        ),
                    )
                ],
            )
        )

    row_schema = (
        _infer_loaded_row_schema(
            tree
        )
    )

    issues = []

    issues.extend(
        _check_feature_declarations(
            tree=(
                tree
            ),
            code=(
                code
            ),
        )
    )

    issues.extend(
        _check_feature_builder_functions(
            tree=(
                tree
            ),
            code=(
                code
            ),
            row_schema=(
                row_schema
            ),
        )
    )

    issues.extend(
        _check_direct_model_input_assignments(
            tree=(
                tree
            ),
            code=(
                code
            ),
            row_schema=(
                row_schema
            ),
        )
    )

    issues.extend(
        _check_official_splits(
            tree=(
                tree
            ),
            code=(
                code
            ),
        )
    )

    issues.extend(
        _check_evaluate_override(
            tree=(
                tree
            ),
            code=(
                code
            ),
        )
    )

    issues = (
        _deduplicate_issues(
            issues
        )
    )

    return (
        ResearchIntegrityValidationResult(
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