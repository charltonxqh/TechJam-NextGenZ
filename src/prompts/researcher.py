"""
Description: Defines prompts used by the research agent to generate ML hypotheses and experiments.
Owner: Charlton / David
Input: Baseline results, experiment history, and research context
Output: Researcher prompts
"""

import json


RESEARCHER_SYSTEM_PROMPT = """
You are an autonomous machine learning researcher working on a recommender-system benchmark.

Your goal is to propose one technically meaningful experiment that can improve validation performance.

The benchmark is evaluated using:
- GAUC
- nDCG@5
- Primary = mean(GAUC, nDCG@5)

You must reason from previous experiment results and avoid repeating failed ideas without a clear justification.

Do not modify:
- the official evaluation logic
- the official train/validation split
- the hidden test set

Return only valid JSON matching the requested schema.
""".strip()


def build_researcher_prompt(
    baseline_result,
    history: list[dict],
    research_context: str = "",
) -> str:
    """
    Build the user prompt for proposing the next experiment.
    """

    baseline_json = json.dumps(
        baseline_result,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    history_json = json.dumps(
        history,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    return f"""
Current baseline / best reference:
{baseline_json}

Previous experiments:
{history_json}

Additional research context:
{research_context or "None"}

Propose exactly ONE next experiment.

Return valid JSON in exactly this structure:

{{
  "hypothesis": "A clear, testable hypothesis.",
  "rationale": "Why this experiment is worth testing based on previous evidence.",
  "change_type": "A short category such as loss, feature, model, training, multi_task, temporal, or sequence.",
  "parameters": {{
    "key": "value"
  }}
}}

Requirements:
- Propose only one experiment.
- The hypothesis must be falsifiable.
- Use previous experiment results as evidence.
- Do not repeat an already-tested experiment unless there is a concrete reason.
- Prefer changes that can be implemented and evaluated within the experiment budget.
- Do not use hidden-test information.
- Do not change the official evaluation implementation.
""".strip()