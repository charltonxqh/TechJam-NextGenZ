"""
Description: Defines prompts used by the reflection agent to analyze completed experiments.
Owner: Charlton / David
Input: Experiment hypothesis, results, and previous research
Output: Reflection prompts
"""

import json


REFLECTOR_SYSTEM_PROMPT = """
You are an autonomous ML research reviewer.

Your task is to analyze the result of a completed recommender-system experiment
and determine what was learned.

Focus on:
- whether the hypothesis was supported
- changes in GAUC
- changes in nDCG@5
- changes in Primary
- whether the research direction should be kept, rejected, or retried
- what insight should influence the next experiment

Do not propose a full new experiment.
That is the Researcher's responsibility.

Return only valid JSON.
""".strip()


def build_reflector_prompt(
    experiment_spec,
    experiment_result,
    previous_best_primary: float,
    history: list[dict],
) -> str:

    return f"""
Experiment:
{json.dumps(experiment_spec, indent=2, default=str)}

Experiment result:
{json.dumps(experiment_result, indent=2, default=str)}

Previous best Primary:
{previous_best_primary}

Previous experiment history:
{json.dumps(history, indent=2, default=str)}

Analyze this experiment.

Return JSON in exactly this structure:

{{
  "verdict": "keep | reject | retry",
  "analysis": "What the experiment taught us.",
  "next_direction": "Short research direction or null"
}}

Guidelines:
- Use the metrics as evidence.
- A small improvement may be noise, so avoid overclaiming.
- If the experiment failed because of an implementation/runtime issue, prefer retry.
- If the hypothesis appears promising, use keep.
- If the evidence does not support the hypothesis, use reject.
""".strip()