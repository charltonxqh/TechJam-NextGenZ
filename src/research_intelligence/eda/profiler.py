"""
Description: Performs deterministic foundational EDA on the train and validation splits and converts observations into research knowledge.
Owner: Charlton / David
Input: KuaiRand-Pure data directory
Output: ResearchKnowledgeItem objects
"""

import csv

from collections import (
    Counter,
)

from pathlib import Path

from statistics import (
    mean,
    median,
)

from src.research_intelligence.knowledge_store import (
    ResearchKnowledgeItem,
)


TRAIN_START = 20220408
TRAIN_END = 20220421

VALID_START = 20220422
VALID_END = 20220428


LOG_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)


def _safe_float(
    value: str | None,
) -> float | None:

    if value in (
        None,
        "",
    ):
        return None

    try:

        return float(
            value
        )

    except ValueError:

        return None


def _is_positive(
    value: str | None,
) -> bool:

    if value in (
        None,
        "",
        "0",
        "0.0",
    ):
        return False

    return True


def run_foundational_eda(
    data_dir: Path,
) -> list[
    ResearchKnowledgeItem
]:
    """
    Profile broad characteristics relevant to recommender-system research.

    This stage is hypothesis-neutral. It measures the dataset and leaves
    interpretation and model selection to the Researcher.
    """

    split_rows = {
        "train": 0,
        "valid": 0,
    }

    positive_counts = {
        "train": 0,
        "valid": 0,
    }

    duration_values = {
        "train": [],
        "valid": [],
    }

    user_counts = {
        "train": Counter(),
        "valid": Counter(),
    }

    video_counts = {
        "train": Counter(),
        "valid": Counter(),
    }

    tab_counts = {
        "train": Counter(),
        "valid": Counter(),
    }

    auxiliary_positive = {
        "train": Counter(),
        "valid": Counter(),
    }

    auxiliary_seen = {
        "train": Counter(),
        "valid": Counter(),
    }

    candidate_auxiliary_fields = {
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
    }

    for filename in LOG_FILES:

        path = (
            data_dir
            / filename
        )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(
                file
            )

            for row in reader:

                date_value = row.get(
                    "date"
                )

                if date_value is None:
                    continue

                try:

                    date = int(
                        date_value
                    )

                except ValueError:

                    continue

                if (
                    TRAIN_START
                    <= date
                    <= TRAIN_END
                ):

                    split = "train"

                elif (
                    VALID_START
                    <= date
                    <= VALID_END
                ):

                    split = "valid"

                else:

                    # Explicitly exclude the test period.
                    continue

                split_rows[
                    split
                ] += 1

                if _is_positive(
                    row.get(
                        "long_view"
                    )
                ):

                    positive_counts[
                        split
                    ] += 1

                user_id = row.get(
                    "user_id"
                )

                if user_id is not None:

                    user_counts[
                        split
                    ][user_id] += 1

                video_id = row.get(
                    "video_id"
                )

                if video_id is not None:

                    video_counts[
                        split
                    ][video_id] += 1

                tab = row.get(
                    "tab"
                )

                if tab is not None:

                    tab_counts[
                        split
                    ][tab] += 1

                duration = _safe_float(
                    row.get(
                        "duration_ms"
                    )
                )

                if duration is not None:

                    duration_values[
                        split
                    ].append(
                        duration
                    )

                for field in (
                    candidate_auxiliary_fields
                ):

                    if field not in row:
                        continue

                    auxiliary_seen[
                        split
                    ][field] += 1

                    if _is_positive(
                        row.get(
                            field
                        )
                    ):

                        auxiliary_positive[
                            split
                        ][field] += 1

    knowledge: list[
        ResearchKnowledgeItem
    ] = []

    # =====================================================
    # Dataset size and label distribution
    # =====================================================

    for split in (
        "train",
        "valid",
    ):

        row_count = (
            split_rows[
                split
            ]
        )

        if row_count == 0:
            continue

        positive_rate = (
            positive_counts[
                split
            ]
            / row_count
        )

        knowledge.append(
            ResearchKnowledgeItem(
                knowledge_id=(
                    f"eda_{split}_"
                    f"target_rate"
                ),
                source_type="eda",
                topic="target_distribution",
                claim=(
                    f"{split} long_view "
                    f"positive rate is "
                    f"{positive_rate:.4f}."
                ),
                evidence={
                    "split": split,
                    "rows": row_count,
                    "positive_rate": (
                        positive_rate
                    ),
                },
                source=(
                    "autonomous foundational EDA"
                ),
            )
        )

    # =====================================================
    # User-history characteristics
    # =====================================================

    train_user_history = list(
        user_counts[
            "train"
        ].values()
    )

    if train_user_history:

        knowledge.append(
            ResearchKnowledgeItem(
                knowledge_id=(
                    "eda_train_user_history"
                ),
                source_type="eda",
                topic="user_history",
                claim=(
                    "Training user interaction "
                    "history distribution was "
                    "measured."
                ),
                evidence={
                    "unique_users": len(
                        train_user_history
                    ),
                    "mean_interactions": mean(
                        train_user_history
                    ),
                    "median_interactions": median(
                        train_user_history
                    ),
                    "min_interactions": min(
                        train_user_history
                    ),
                    "max_interactions": max(
                        train_user_history
                    ),
                },
                source=(
                    "autonomous foundational EDA"
                ),
            )
        )

    # =====================================================
    # Ranking group structure
    # =====================================================

    for split in (
        "train",
        "valid",
    ):

        group_sizes = list(
            user_counts[
                split
            ].values()
        )

        if not group_sizes:
            continue

        singleton_fraction = (
            sum(
                size == 1
                for size
                in group_sizes
            )
            / len(
                group_sizes
            )
        )

        knowledge.append(
            ResearchKnowledgeItem(
                knowledge_id=(
                    f"eda_{split}_"
                    f"user_groups"
                ),
                source_type="eda",
                topic="ranking_groups",
                claim=(
                    f"{split} user-level "
                    f"ranking-group structure "
                    f"was measured."
                ),
                evidence={
                    "groups": len(
                        group_sizes
                    ),
                    "mean_group_size": mean(
                        group_sizes
                    ),
                    "median_group_size": median(
                        group_sizes
                    ),
                    "singleton_fraction": (
                        singleton_fraction
                    ),
                },
                source=(
                    "autonomous foundational EDA"
                ),
            )
        )

    # =====================================================
    # User / item cardinality
    # =====================================================

    knowledge.append(
        ResearchKnowledgeItem(
            knowledge_id=(
                "eda_train_cardinality"
            ),
            source_type="eda",
            topic="feature_cardinality",
            claim=(
                "Training user and video "
                "cardinalities were measured."
            ),
            evidence={
                "unique_users": len(
                    user_counts[
                        "train"
                    ]
                ),
                "unique_videos": len(
                    video_counts[
                        "train"
                    ]
                ),
            },
            source=(
                "autonomous foundational EDA"
            ),
        )
    )

    # =====================================================
    # Duration
    # =====================================================

    for split in (
        "train",
        "valid",
    ):

        durations = (
            duration_values[
                split
            ]
        )

        if not durations:
            continue

        knowledge.append(
            ResearchKnowledgeItem(
                knowledge_id=(
                    f"eda_{split}_duration"
                ),
                source_type="eda",
                topic="duration",
                claim=(
                    f"{split} video-duration "
                    f"distribution was measured."
                ),
                evidence={
                    "mean_duration_ms": mean(
                        durations
                    ),
                    "median_duration_ms": median(
                        durations
                    ),
                },
                source=(
                    "autonomous foundational EDA"
                ),
            )
        )

    # =====================================================
    # Tab distribution
    # =====================================================

    for split in (
        "train",
        "valid",
    ):

        total = sum(
            tab_counts[
                split
            ].values()
        )

        if total == 0:
            continue

        distribution = {
            key: (
                count
                / total
            )
            for (
                key,
                count,
            )
            in tab_counts[
                split
            ].items()
        }

        knowledge.append(
            ResearchKnowledgeItem(
                knowledge_id=(
                    f"eda_{split}_"
                    f"tab_distribution"
                ),
                source_type="eda",
                topic="tab_distribution",
                claim=(
                    f"{split} tab "
                    f"distribution was measured."
                ),
                evidence={
                    "distribution": (
                        distribution
                    )
                },
                source=(
                    "autonomous foundational EDA"
                ),
            )
        )

    # =====================================================
    # Auxiliary-label density
    # =====================================================

    for field in sorted(
        candidate_auxiliary_fields
    ):

        seen = (
            auxiliary_seen[
                "train"
            ][field]
        )

        if seen == 0:
            continue

        positive_rate = (
            auxiliary_positive[
                "train"
            ][field]
            / seen
        )

        knowledge.append(
            ResearchKnowledgeItem(
                knowledge_id=(
                    f"eda_aux_{field}"
                ),
                source_type="eda",
                topic="auxiliary_labels",
                claim=(
                    f"Training density of "
                    f"{field} was measured."
                ),
                evidence={
                    "field": field,
                    "observations": seen,
                    "positive_rate": (
                        positive_rate
                    ),
                },
                source=(
                    "autonomous foundational EDA"
                ),
            )
        )

    return knowledge