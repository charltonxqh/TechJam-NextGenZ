"""
Description: Discovers skill metadata and loads full skill content on demand for autonomous research.
Owner: Charlton / David
Input: Skill names or the local research-intelligence skills directory
Output: Lightweight skill catalog metadata or selected full skill content
"""

from dataclasses import (
    asdict,
    dataclass,
)

from pathlib import (
    Path,
)


SKILLS_DIR = (
    Path(__file__)
    .resolve()
    .parent
    / "skills"
)


@dataclass
class SkillMetadata:
    name: str
    description: str
    path: str

    def as_dict(
        self,
    ) -> dict:

        return asdict(
            self
        )


def _resolve_skill_path(
    skill_name: str,
) -> Path | None:
    """
    Resolve one skill name while supporting both folder-based SKILL.md
    and flat Markdown skill layouts.
    """

    candidates = [
        (
            SKILLS_DIR
            / skill_name
            / "SKILL.md"
        ),
        (
            SKILLS_DIR
            / skill_name
            / "skill.md"
        ),
        (
            SKILLS_DIR
            / f"{skill_name}.md"
        ),
        (
            SKILLS_DIR
            / f"{skill_name.upper()}.md"
        ),
    ]

    for candidate in candidates:

        if candidate.exists():

            return candidate

    return None


def _split_front_matter(
    content: str,
) -> tuple[
    dict[str, str],
    str,
]:
    """
    Parse lightweight YAML-style front matter without requiring
    an additional YAML dependency.

    Only scalar metadata fields are needed for skill routing.
    """

    stripped = (
        content.lstrip()
    )

    if not stripped.startswith(
        "---"
    ):

        return (
            {},
            content,
        )

    lines = (
        stripped.splitlines()
    )

    if (
        not lines
        or lines[0].strip()
        != "---"
    ):

        return (
            {},
            content,
        )

    closing_index = None

    for index in range(
        1,
        len(
            lines
        ),
    ):

        if (
            lines[index]
            .strip()
            == "---"
        ):

            closing_index = (
                index
            )

            break

    if closing_index is None:

        return (
            {},
            content,
        )

    metadata = {}

    for line in lines[
        1:closing_index
    ]:

        stripped_line = (
            line.strip()
        )

        if (
            not stripped_line
            or stripped_line.startswith(
                "#"
            )
            or ":"
            not in stripped_line
        ):

            continue

        key, value = (
            stripped_line
            .split(
                ":",
                1,
            )
        )

        key = (
            key.strip()
        )

        value = (
            value.strip()
            .strip(
                "\"'"
            )
        )

        if (
            key
            and value
        ):

            metadata[
                key
            ] = (
                value
            )

    body = "\n".join(
        lines[
            closing_index
            + 1:
        ]
    ).lstrip()

    return (
        metadata,
        body,
    )


def _fallback_description(
    body: str,
) -> str:
    """
    Derive a short fallback description for an older skill file that
    does not yet contain metadata.
    """

    for line in body.splitlines():

        stripped = (
            line.strip()
        )

        if not stripped:

            continue

        if stripped.startswith(
            "#"
        ):

            continue

        return (
            stripped[:240]
        )

    return (
        "Procedural research guidance."
    )


def get_skill_metadata(
    skill_name: str,
) -> SkillMetadata:
    """
    Read only lightweight routing metadata for one skill.
    """

    path = (
        _resolve_skill_path(
            skill_name
        )
    )

    if path is None:

        raise FileNotFoundError(
            f"Skill not found: "
            f"{skill_name}"
        )

    content = (
        path.read_text(
            encoding="utf-8"
        )
    )

    metadata, body = (
        _split_front_matter(
            content
        )
    )

    name = (
        metadata.get(
            "name"
        )
        or skill_name
    )

    description = (
        metadata.get(
            "description"
        )
        or _fallback_description(
            body
        )
    )

    return SkillMetadata(
        name=name,
        description=(
            description
        ),
        path=str(
            path
        ),
    )


def discover_skills(
) -> list[
    SkillMetadata
]:
    """
    Discover available skills and return metadata only.
    """

    if not SKILLS_DIR.exists():

        return []

    discovered = {}

    for path in SKILLS_DIR.rglob(
        "*.md"
    ):

        if (
            path.name.lower()
            == "skill.md"
        ):

            skill_name = (
                path.parent.name
            )

        else:

            skill_name = (
                path.stem
            )

        if skill_name in discovered:

            continue

        try:

            metadata = (
                get_skill_metadata(
                    skill_name
                )
            )

        except FileNotFoundError:

            continue

        discovered[
            metadata.name
        ] = (
            metadata
        )

    return sorted(
        discovered.values(),
        key=lambda item: (
            item.name
        ),
    )


def build_skill_catalog(
) -> str:
    """
    Build the lightweight skill catalog shown to the Researcher.

    Full skill bodies are intentionally excluded.
    """

    skills = (
        discover_skills()
    )

    if not skills:

        return (
            "No skills available."
        )

    lines = []

    for skill in skills:

        lines.append(
            (
                f"- {skill.name}: "
                f"{skill.description}"
            )
        )

    return "\n".join(
        lines
    )


def get_skill_catalog_records(
) -> list[dict]:
    """
    Return serializable skill metadata for run logging.
    """

    return [
        skill.as_dict()
        for skill
        in discover_skills()
    ]


def load_skill(
    skill_name: str,
) -> str:
    """
    Load the full body of one selected skill.

    Metadata is not repeated because it has already been exposed through
    the lightweight catalog.
    """

    path = (
        _resolve_skill_path(
            skill_name
        )
    )

    if path is None:

        raise FileNotFoundError(
            f"Skill not found: "
            f"{skill_name}"
        )

    content = (
        path.read_text(
            encoding="utf-8"
        )
    )

    _, body = (
        _split_front_matter(
            content
        )
    )

    return (
        body.strip()
    )


def load_skills(
    skill_names: list[str],
) -> str:
    """
    Load only the explicitly selected skills.

    This function remains available for callers that want multiple
    on-demand skills in one context block.
    """

    blocks = []

    for skill_name in skill_names:

        content = (
            load_skill(
                skill_name
            )
        )

        blocks.append(
            (
                f"=== SKILL: "
                f"{skill_name} ===\n"
                f"{content}"
            )
        )

    return "\n\n".join(
        blocks
    )