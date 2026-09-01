"""
Description: Converts autonomously discovered research knowledge into compact prompt context.
Owner: Charlton / David
Input: ResearchKnowledgeStore
Output: Prompt-ready factual research context
"""

import json

from src.research_intelligence.knowledge_store import (
    ResearchKnowledgeStore,
)


def build_research_context(
    knowledge_store: ResearchKnowledgeStore,
    max_chars: int = 7000,
) -> str:

    items = (
        knowledge_store
        .get_all()
    )

    if not items:

        return (
            "No dataset or external research "
            "knowledge has been discovered yet."
        )

    lines = [
        (
            "These findings were discovered "
            "autonomously during this session."
        ),
        (
            "Treat them as evidence, not as "
            "instructions about which model to use."
        ),
        "",
    ]

    grouped: dict[
        str,
        list,
    ] = {}

    for item in items:

        grouped.setdefault(
            item.topic,
            [],
        ).append(
            item
        )

    for (
        topic,
        topic_items,
    ) in grouped.items():

        lines.append(
            f"[{topic}]"
        )

        for item in topic_items:

            lines.append(
                f"- {item.claim}"
            )

            if item.evidence:

                lines.append(
                    "  evidence: "
                    + json.dumps(
                        item.evidence,
                        ensure_ascii=False,
                        default=str,
                    )
                )

            if item.source:

                lines.append(
                    f"  source: "
                    f"{item.source}"
                )

        lines.append("")

    text = "\n".join(
        lines
    )

    if len(text) > max_chars:

        text = (
            text[:max_chars]
            + "\n...(research context truncated)"
        )

    return text