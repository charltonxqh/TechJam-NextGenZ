"""
Description: Analyzes experiment results and determines what was learned from the tested hypothesis.
Owner: Charlton / David
Input: ExperimentSpec, ExperimentResult, and experiment history
Output: Reflection
"""

from dataclasses import asdict

from src.config import REFLECTOR_MODEL
from src.schemas import (
    ExperimentResult,
    ExperimentSpec,
    Reflection,
    ReflectionOutput,
)
from src.prompts.reflector import (
    REFLECTOR_SYSTEM_PROMPT,
    build_reflector_prompt,
)
from src.tools.llm_client import GeminiClient


class Reflector:

    def __init__(
        self,
        llm_client: GeminiClient,
    ) -> None:
        self.llm_client = llm_client

    def reflect(
        self,
        spec: ExperimentSpec,
        result: ExperimentResult,
        previous_best_primary: float,
        history: list[dict],
    ) -> Reflection:
        """
        Analyze one completed experiment.
        """

        prompt = build_reflector_prompt(
            experiment_spec=asdict(spec),
            experiment_result=asdict(result),
            previous_best_primary=previous_best_primary,
            history=history,
        )

        output = self.llm_client.generate_structured(
            system_prompt=REFLECTOR_SYSTEM_PROMPT,
            prompt=prompt,
            model=REFLECTOR_MODEL,
            response_schema=ReflectionOutput,
        )

        return Reflection(
            verdict=output.verdict,
            analysis=output.analysis,
            next_direction=output.next_direction,
        )