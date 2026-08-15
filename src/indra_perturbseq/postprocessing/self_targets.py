"""Separate self-targeting genes from pathway data.

Rows where ``source == target`` are written to one CSV, and the remaining
rows to another.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def split_self_targets(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into self-targeting and non-self-targeting subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``source`` and ``target`` columns.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(self_targeting, non_self_targeting)`` copies.
    """
    self_mask = df["source"] == df["target"]
    self_df = df[self_mask].copy()
    non_self_df = df[~self_mask].copy()
    pct = len(self_df) / len(df) * 100 if len(df) else 0.0
    logger.info(
        "Self-targeting: %d (%.1f%%), non-self-targeting: %d",
        len(self_df), pct, len(non_self_df),
    )
    return self_df, non_self_df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Separate self-targeting genes (source == target) into "
                    "distinct CSV files.",
    )
    ap.add_argument("--input", required=True, help="Input CSV.")
    ap.add_argument("--output-self", required=True,
                    help="Output CSV for self-targeting rows.")
    ap.add_argument("--output-non-self", required=True,
                    help="Output CSV for non-self-targeting rows.")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.input, low_memory=False)
    logger.info("Loaded %d rows from %s", len(df), args.input)

    self_df, non_self_df = split_self_targets(df)

    self_df.to_csv(args.output_self, index=False)
    logger.info("Self-targeting -> %s (%d rows)", args.output_self, len(self_df))

    non_self_df.to_csv(args.output_non_self, index=False)
    logger.info("Non-self-targeting -> %s (%d rows)",
                args.output_non_self, len(non_self_df))


if __name__ == "__main__":
    main()
