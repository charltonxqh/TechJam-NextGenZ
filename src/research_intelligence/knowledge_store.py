"""
Description: Stores factual research knowledge discovered autonomously during one research session.
Owner: Charlton / David
Input: EDA findings and retrieved research evidence
Output: Persistent session-scoped research knowledge
"""

import json

from dataclasses import (
    asdict,
    dataclass,
    field,
)

from pathlib import Path

from typing import (
    Any,
    Literal,
)


KnowledgeSource = Literal[
    "eda",
    "paper",
    "web",
    "code",
    "documentation",
    "inference",
]


@dataclass
class ResearchKnowledgeItem:
    knowledge_id: str

    source_type: KnowledgeSource

    topic: str
    claim: str

    evidence: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    source: str | None = None

    confidence: float = 1.0

    created_at_iteration: int = 0

    def as_dict(
        self,
    ) -> dict:

        return asdict(
            self
        )

    @staticmethod
    def from_dict(
        data: dict[
            str,
            Any,
        ],
    ) -> "ResearchKnowledgeItem":

        return ResearchKnowledgeItem(
            **data
        )


class ResearchKnowledgeStore:

    def __init__(
        self,
        path: str,
    ) -> None:

        self.path = Path(
            path
        )

        self.items: list[
            ResearchKnowledgeItem
        ] = []

        if self.path.exists():

            self._load()

    def add(
        self,
        item: ResearchKnowledgeItem,
    ) -> bool:
        """
        Add one new knowledge item.

        Returns True only when the item was actually stored.
        """

        if any(
            existing.knowledge_id
            == item.knowledge_id
            for existing
            in self.items
        ):

            return False

        if any(
            existing.claim
            == item.claim
            and existing.source
            == item.source
            for existing
            in self.items
        ):

            return False

        self.items.append(
            item
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    item.as_dict(),
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

        return True

    def add_many(
        self,
        items: list[
            ResearchKnowledgeItem
        ],
    ) -> int:

        added = 0

        for item in items:

            if self.add(
                item
            ):

                added += 1

        return added

    def add_research_evidence(
        self,
        evidence,
        iteration: int,
    ) -> bool:

        if (
            evidence.source_type
            == "paper"
        ):

            source_type = "paper"

        elif (
            evidence.source_type
            in (
                "web",
                "webpage",
            )
        ):

            source_type = "web"

        else:

            source_type = (
                "documentation"
            )

        item = ResearchKnowledgeItem(
            knowledge_id=(
                evidence.evidence_id
            ),
            source_type=(
                source_type
            ),
            topic=(
                evidence.topic
            ),
            claim=(
                evidence.relevant_findings
            ),
            evidence={
                "title": (
                    evidence.title
                ),
                "problem_addressed": (
                    evidence.problem_addressed
                ),
                "method": (
                    evidence.method
                ),
                "assumptions": (
                    evidence.assumptions
                ),
                "applicability": (
                    evidence.applicability
                ),
                "implementation_hint": (
                    evidence
                    .implementation_hint
                ),
            },
            source=(
                evidence.url
            ),
            confidence=(
                evidence.confidence
            ),
            created_at_iteration=(
                iteration
            ),
        )

        return self.add(
            item
        )

    def get_all(
        self,
    ) -> list[
        ResearchKnowledgeItem
    ]:

        return list(
            self.items
        )

    def get_by_topic(
        self,
        topic: str,
    ) -> list[
        ResearchKnowledgeItem
    ]:

        return [
            item
            for item
            in self.items
            if item.topic
            == topic
        ]

    def _load(
        self,
    ) -> None:

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = (
                    line.strip()
                )

                if not line:

                    continue

                self.items.append(
                    ResearchKnowledgeItem
                    .from_dict(
                        json.loads(
                            line
                        )
                    )
                )