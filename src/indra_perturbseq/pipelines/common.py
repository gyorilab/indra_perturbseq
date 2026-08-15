"""Shared helpers for pipeline CLIs.

These utilities intentionally handle only orchestration concerns so hop
modules can focus on path discovery and enrichment logic.
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd

from indra_perturbseq.gene_lists import load_source_genes


def warn_deprecated_flags(
    argv: list[str] | None,
    replacements: dict[str, str],
    logger: logging.Logger,
) -> None:
    """Log warning for deprecated CLI flags present in *argv*."""
    tokens = list(argv) if argv is not None else sys.argv[1:]
    seen = set(tokens)
    for old, new in replacements.items():
        if old in seen:
            logger.warning(
                "Flag '%s' is deprecated and will be removed in a future "
                "release; use '%s' instead.",
                old,
                new,
            )


def load_sources_from_args(args) -> list[str]:
    """Load source genes using standard CLI argument names."""
    return load_source_genes(
        args.source_genes_csv,
        gene_column=getattr(args, "gene_column", "Gene"),
        filter_column=getattr(args, "filter_column", "analysis_flag"),
        filter_value=getattr(args, "filter_value", "Use_for_analysis"),
        explicit_genes=getattr(args, "genes", None),
        limit=getattr(args, "limit_genes", 0),
    )


def split_self_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into non-self and self-target outputs."""
    if "source" not in df.columns or "target" not in df.columns:
        raise ValueError("Dataframe must contain 'source' and 'target' columns")
    main_df = df[df["source"] != df["target"]].copy()
    self_df = df[df["source"] == df["target"]].copy()
    return main_df, self_df


def ensure_parent_dir(path: str) -> None:
    """Create parent directory for *path* when needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def write_split_outputs(
    df: pd.DataFrame,
    output_main: str,
    output_self_targets: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe by self-target status and write both CSVs."""
    main_df, self_df = split_self_targets(df)
    ensure_parent_dir(output_main)
    ensure_parent_dir(output_self_targets)
    main_df.to_csv(output_main, index=False)
    self_df.to_csv(output_self_targets, index=False)
    logger.info("Non-self rows: %d -> %s", len(main_df), output_main)
    logger.info("Self rows:     %d -> %s", len(self_df), output_self_targets)
    return main_df, self_df
