"""
Description: Exposes deterministic targeted EDA analyses for autonomous research.
Owner: Charlton / David
Input: EDA tool name and dataset path
Output: ResearchKnowledgeItem records
"""

from pathlib import Path

from src.research_intelligence.eda.profiler import (
    run_foundational_eda,
)

from src.research_intelligence.knowledge_store import (
    ResearchKnowledgeItem,
)


SUPPORTED_EDA_TOOLS = {
    "dataset_profile",
}


def run_eda_tool(
    name: str,
    data_dir: Path,
) -> list[
    ResearchKnowledgeItem
]:

    if (
        name
        == "dataset_profile"
    ):

        return run_foundational_eda(
            data_dir
        )

    raise ValueError(
        f"Unknown EDA tool: "
        f"{name}. "
        f"Supported tools: "
        f"{sorted(SUPPORTED_EDA_TOOLS)}"
    )