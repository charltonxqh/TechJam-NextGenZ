"""
Description: Provides shared utilities for parsing and handling experiment evaluation metrics.
Owner: Hayden
Input: Raw experiment output
Output: Parsed GAUC, nDCG@5, and Primary metrics
"""

import re

from src.schemas import Metrics


def parse_metrics(
    output: str,
    split: str | None = None,
) -> Metrics:

    pattern = re.compile(
        r"(?:(valid|test)\s+)?"
        r"GAUC\s+([0-9.]+)\s*\|\s*"
        r"nDCG@5\s+([0-9.]+)\s*\|\s*"
        r"primary\s+([0-9.]+)",
        re.IGNORECASE,
    )

    matches = pattern.findall(output)

    if not matches:
        raise ValueError(
            "Could not find GAUC, nDCG@5, and Primary in output."
        )

    if split is not None:
        split = split.lower()

        matches = [
            match
            for match in matches
            if match[0].lower() == split
        ]

        if not matches:
            raise ValueError(
                f"Could not find metrics for split '{split}'."
            )

    _, gauc, ndcg5, primary = matches[-1]

    return Metrics(
        gauc=float(gauc),
        ndcg5=float(ndcg5),
        primary=float(primary),
    )