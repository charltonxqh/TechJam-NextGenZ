"""
Description: Records the autonomous online-research process for debugging, provenance, and auditability.
Owner: Charlton / David
Input: Research-action events
Output: Session-scoped research_trace.jsonl
"""

import json

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path

from typing import (
    Any,
)


class ResearchTrace:

    def __init__(
        self,
        path: str,
    ) -> None:

        self.path = Path(
            path
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def record(
        self,
        event_type: str,
        data: dict[
            str,
            Any,
        ],
    ) -> None:

        event = {
            "timestamp": (
                datetime.now(
                    timezone.utc
                )
                .isoformat()
            ),
            "event_type": (
                event_type
            ),
            **data,
        }

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )