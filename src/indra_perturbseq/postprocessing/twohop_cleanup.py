"""Clean 2-hop pathway data.
1. Remove rows whose intermediate is a non-human entity (MeSH, UniProt,.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

logger = logging.getLogger(__name__)

NON_HUMAN_PREFIXES = (
    "mesh:", "uniprot:", "chebi:", "go:",
    "UP:", "MESH:", "CHEBI:", "GO:",
)


def remove_non_human_intermediates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose ``intermediate`` contains a non-human identifier.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with an ``intermediate`` column.

    Returns
    -------
    pd.DataFrame
        Filtered copy of *df*.
    """
    df = df.copy()
    mask = pd.Series(True, index=df.index)
    for prefix in NON_HUMAN_PREFIXES:
        mask &= ~df["intermediate"].str.contains(prefix, na=False)
    before = len(df)
    df = df[mask]
    logger.info("Non-human filter: %d -> %d rows (-%d)",
                before, len(df), before - len(df))
    return df


def deduplicate_triplets(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per (source, intermediate, target) with highest mean evidence.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``source``, ``intermediate``, ``target``,
        ``evidence_1``, ``evidence_2``.

    Returns
    -------
    pd.DataFrame
        Deduplicated copy.
    """
    df = df.copy()
    df["_mean_ev"] = (df["evidence_1"] + df["evidence_2"]) / 2
    idx_keep = df.groupby(
        ["source", "intermediate", "target"],
    )["_mean_ev"].idxmax()
    before = len(df)
    df = df.loc[idx_keep].drop(columns=["_mean_ev"])
    logger.info("Dedup triplets: %d -> %d rows (-%d)",
                before, len(df), before - len(df))
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleanup pipeline on *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Raw 2-hop data.

    Returns
    -------
    pd.DataFrame
        Cleaned data.
    """
    df = remove_non_human_intermediates(df)
    df = deduplicate_triplets(df)
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Clean 2-hop INDRA data: remove non-human intermediates "
                    "and deduplicate triplets.",
    )
    ap.add_argument("--input", required=True, help="Input CSV.")
    ap.add_argument("--output", required=True, help="Output CSV.")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.input, low_memory=False)
    logger.info("Loaded %d rows from %s", len(df), args.input)

    df = clean(df)
    df.to_csv(args.output, index=False)
    logger.info("Saved %d rows -> %s", len(df), args.output)


if __name__ == "__main__":
    main()
