"""Helpers for evidence-source overlap statistics."""

from __future__ import annotations


def build_db_only_vs_db_reader_counts(combined_counts: dict[str, int]) -> dict[str, int]:
    """Collapse overlap counts into DB-only vs DB+Reader 2-set regions.

    DB+Reader means a pair has at least one reader-supported pathway.
    """
    required = {
        "100_db_only",
        "010_mixed",
        "001_reader_only",
        "110_db_and_mixed",
        "101_db_and_reader",
        "011_mixed_and_reader",
        "111_all_three",
        "union_pairs",
    }
    missing = required - set(combined_counts)
    if missing:
        raise ValueError(f"combined_counts missing keys: {sorted(missing)}")

    db_only_only_pairs = int(combined_counts["100_db_only"])
    db_only_and_db_plus_reader_pairs = int(
        combined_counts["110_db_and_mixed"]
        + combined_counts["101_db_and_reader"]
        + combined_counts["111_all_three"]
    )
    db_plus_reader_only_pairs = int(
        combined_counts["010_mixed"]
        + combined_counts["001_reader_only"]
        + combined_counts["011_mixed_and_reader"]
    )
    total_explained_pairs = int(combined_counts["union_pairs"])

    if (
        db_only_only_pairs
        + db_plus_reader_only_pairs
        + db_only_and_db_plus_reader_pairs
        != total_explained_pairs
    ):
        raise ValueError("Collapsed 2-set region sum does not match total explained pairs")

    return {
        "db_only_only_pairs": db_only_only_pairs,
        "db_plus_reader_only_pairs": db_plus_reader_only_pairs,
        "db_only_and_db_plus_reader_pairs": db_only_and_db_plus_reader_pairs,
        "total_explained_pairs": total_explained_pairs,
    }


def build_priority_db_only_vs_db_reader_counts(combined_counts: dict[str, int]) -> dict[str, int]:
    """Collapse overlap counts into a disjoint DB-only vs DB+Reader split.

    Priority rule:
    - if a pair has at least one DB-only pathway, count it as DB-only
    - otherwise, if it has any reader-supported pathway, count it as DB+Reader
    """
    overlap = build_db_only_vs_db_reader_counts(combined_counts)
    db_only_pairs = int(
        overlap["db_only_only_pairs"] + overlap["db_only_and_db_plus_reader_pairs"]
    )
    db_plus_reader_pairs = int(overlap["db_plus_reader_only_pairs"])
    total_explained_pairs = int(overlap["total_explained_pairs"])

    if db_only_pairs + db_plus_reader_pairs != total_explained_pairs:
        raise ValueError("Priority-collapsed counts do not sum to total explained pairs")

    return {
        "db_only_pairs": db_only_pairs,
        "db_plus_reader_pairs": db_plus_reader_pairs,
        "total_explained_pairs": total_explained_pairs,
    }
