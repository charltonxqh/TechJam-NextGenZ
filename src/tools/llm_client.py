"""
Description: Provides a unified interface for calling Google Gemini models and returning structured or text responses.
Owner: Hayden
Input: System prompt, user prompt, model name, and optional response schema
Output: Parsed structured LLM response or raw text response
"""

import json

from typing import Type, TypeVar

from google import genai

from pydantic import (
    BaseModel,
    ValidationError,
)

from src.config import GEMINI_API_KEY


T = TypeVar(
    "T",
    bound=BaseModel,
)


class GeminiClient:

    def __init__(
        self,
    ) -> None:

        if not GEMINI_API_KEY:

            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

    def generate_structured(
        self,
        system_prompt: str,
        prompt: str,
        model: str,
        response_schema: Type[T],
    ) -> T:
        """
        Generate and validate a structured Gemini response.
        """

        schema_json = json.dumps(
            response_schema
            .model_json_schema(),
            indent=2,
            ensure_ascii=False,
        )

        combined_prompt = f"""
SYSTEM INSTRUCTIONS:

{system_prompt}

USER REQUEST:

{prompt}

REQUIRED JSON SCHEMA:

{schema_json}

Return ONLY one valid JSON object matching the schema above exactly.
""".strip()

        interaction = (
            self.client
            .interactions
            .create(
                model=model,
                input=combined_prompt,
                response_format={
                    "type": "text",
                    "mime_type": (
                        "application/json"
                    ),
                },
            )
        )

        if not interaction.output_text:

            raise ValueError(
                "Gemini returned an empty response."
            )

        try:

            return (
                response_schema
                .model_validate_json(
                    interaction.output_text
                )
            )

        except ValidationError as error:

            repair_prompt = f"""
SYSTEM INSTRUCTIONS:

{system_prompt}

The previous response failed local Pydantic schema validation.

ORIGINAL USER REQUEST:

{prompt}

REQUIRED JSON SCHEMA:

{schema_json}

INVALID RESPONSE:

{interaction.output_text}

VALIDATION ERROR:

{error}

Correct the JSON so that it matches the required schema exactly.

Preserve the intended substantive content.

Do not omit required fields such as discriminator fields.

Return ONLY the corrected JSON object.
""".strip()

            repair_interaction = (
                self.client
                .interactions
                .create(
                    model=model,
                    input=repair_prompt,
                    response_format={
                        "type": "text",
                        "mime_type": (
                            "application/json"
                        ),
                    },
                )
            )

            if not (
                repair_interaction
                .output_text
            ):

                raise ValueError(
                    "Gemini returned an empty "
                    "response while repairing "
                    "structured output."
                )

            return (
                response_schema
                .model_validate_json(
                    repair_interaction
                    .output_text
                )
            )

    def generate_text(
        self,
        system_prompt: str,
        prompt: str,
        model: str,
    ) -> str:
        """
        Generate an unconstrained text response.
        """

        combined_prompt = f"""
SYSTEM INSTRUCTIONS:

{system_prompt}

USER REQUEST:

{prompt}
""".strip()

        interaction = (
            self.client
            .interactions
            .create(
                model=model,
                input=combined_prompt,
            )
        )

        if not interaction.output_text:

            raise ValueError(
                "Gemini returned an empty response."
            )

        return (
            interaction.output_text
        )