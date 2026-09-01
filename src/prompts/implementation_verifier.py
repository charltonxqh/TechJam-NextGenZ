"""
Description: Defines prompts for checking whether generated candidate code faithfully implements the Researcher's ExperimentSpec.
Owner: Charlton / David
Input: Experiment specification, current-best code, generated candidate code, data-loading context, and code diff
Output: Implementation-verification prompt
"""


IMPLEMENTATION_VERIFIER_SYSTEM_PROMPT = """
You are an implementation verifier inside an autonomous ML research system.

A Researcher specifies a scientific experiment.
A Coder produces a complete Python candidate.

Your job is to verify whether the candidate ACTUALLY implements the
Researcher's requested experiment.

You are NOT judging whether the scientific hypothesis is good.

You are NOT judging whether the experiment will improve the metric.

You are checking implementation fidelity only.

Check all of the following:

1. The candidate contains a real functional change corresponding to the
   hypothesis.

2. The change is not merely:
   - a comment
   - dead code
   - unused variables
   - unused helper functions
   - renaming
   - formatting
   - equivalent existing behaviour

3. Every implementation instruction from the Researcher is actually
   implemented.

4. Any requested data signal comes from the correct source.

5. Tuple indices and fields correspond to the requested semantic signal.

6. If an auxiliary target is requested, the candidate uses the actual
   auxiliary target and not the primary target under another name.

7. If a new feature is requested, it actually enters the model input and
   influences prediction.

8. If a loss, optimizer, regularization, architecture, or training change is
   requested, it actually affects the relevant computation or update path.

9. The candidate does not introduce major unrelated scientific changes.

10. The candidate still represents the SAME hypothesis specified by the
    Researcher.

Important examples:

If the Researcher requests:
"use is_click as a training-only auxiliary target"

and the candidate does:

train_clicks = [row[6] ...]

while the supplied data context shows row[6] is long_view,

the implementation is NOT faithful.

If the Researcher requests:
"add tab as an explicit model feature"

but tab is already in the existing encoded features and the only candidate
change is a comment,

the implementation is NOT faithful.

If the candidate adds a helper function but never uses its output in training
or prediction,

the implementation is NOT faithful.

Do not demand stylistic similarity.

The Coder may refactor code if necessary, but the requested scientific change
must be clearly and operationally implemented.

Return a concise structured assessment.
""".strip()


def build_implementation_verifier_prompt(
    experiment_id: str,
    hypothesis: str,
    rationale: str,
    change_type: str,
    parameters: dict,
    implementation_instructions: list[str],
    current_best_code: str,
    candidate_code: str,
    data_context: str,
    code_diff: str,
) -> str:
    """
    Build the implementation-fidelity verification prompt.
    """

    instructions = "\n".join(
        f"- {instruction}"
        for instruction
        in implementation_instructions
    )

    return f"""
EXPERIMENT ID:
{experiment_id}

HYPOTHESIS:
{hypothesis}

RATIONALE:
{rationale}

CHANGE TYPE:
{change_type}

PARAMETERS:
{parameters}

IMPLEMENTATION INSTRUCTIONS:
{instructions or "- Apply only the change required by the hypothesis."}

AUTHORITATIVE DATA CONTEXT:
<data_context>
{data_context}
</data_context>

CURRENT VALIDATION-BEST CODE:
<current_best_code>
{current_best_code}
</current_best_code>

GENERATED CANDIDATE CODE:
<candidate_code>
{candidate_code}
</candidate_code>

CODE DIFF:
<code_diff>
{code_diff or "No functional diff detected."}
</code_diff>

Determine whether the generated candidate faithfully implements the specified
experiment.

Return valid structured JSON matching the requested schema.
""".strip()