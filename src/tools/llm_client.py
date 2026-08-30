"""
Description: Provides a unified interface for calling Google Gemini models and returning structured responses.
Owner: Hayden
Input: System prompt, user prompt, model name, and response schema
Output: Parsed structured LLM response
"""

from typing import TypeVar, Type

from google import genai
from pydantic import BaseModel

from src.config import GEMINI_API_KEY


T = TypeVar("T", bound=BaseModel)


class GeminiClient:

    def __init__(self) -> None:
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
        Generate a response constrained to the provided Pydantic schema.
        """

        interaction = self.client.interactions.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": response_schema.model_json_schema(),
            },
        )

        if not interaction.output_text:
            raise ValueError(
                "Gemini returned an empty response."
            )

        return response_schema.model_validate_json(
            interaction.output_text
        )