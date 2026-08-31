"""
Description: Loads generic research methodology skills for the autonomous ML researcher.
Owner: Charlton / David
Input: Skill names
Output: Prompt-ready research skill text
"""

from pathlib import Path


SKILLS_DIR = (
    Path(__file__).parent
    / "skills"
)


def load_skill(
    name: str,
) -> str:

    path = (
        SKILLS_DIR
        / f"{name}.md"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Research skill not found: "
            f"{name}"
        )

    return path.read_text(
        encoding="utf-8"
    )


def load_skills(
    names: list[str],
) -> str:

    sections = []

    for name in names:

        content = load_skill(
            name
        )

        sections.append(
            f'<skill name="{name}">\n'
            f"{content}\n"
            f"</skill>"
        )

    return "\n\n".join(
        sections
    )