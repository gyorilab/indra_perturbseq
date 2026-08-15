"""Filter 2-hop data for endothelial intermediate overlap.

Keeps only rows whose ``intermediate`` gene appears in a supplied
endothelial gene list.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_gene_set(path: str, column: str = "gene") -> set[str]:
    """Read a single-column gene list CSV into a set.

    Parameters
    ----------
    path : str
        Path to a CSV containing at least the *column* column.
    column : str
        Column name to extract.

    Returns
    -------
    set[str]
        Unique gene identifiers.
    """
    df = pd.read_csv(path, low_memory=False)
    if column not in df.columns:
        raise ValueError(
            f"Gene list CSV must contain column '{column}'. "
            f"Found: {df.columns.tolist()}"
        )
    genes = set(df[column].dropna().unique())
    logger.info("Loaded %d genes from %s", len(genes), path)
    return genes


def filter_by_intermediate(
    df: pd.DataFrame,
    gene_set: set[str],
) -> pd.DataFrame:
    """Keep rows whose ``intermediate`` is in *gene_set*.

    Parameters
    ----------
    df : pd.DataFrame
        2-hop data with an ``intermediate`` column.
    gene_set : set[str]
        Allowed intermediate gene symbols.

    Returns
    -------
    pd.DataFrame
        Filtered copy.
    """
    mask = df["intermediate"].isin(gene_set)
    before = len(df)
    out = df[mask].copy()
    logger.info(
        "Endothelial filter: %d -> %d rows (%.1f%% retained)",
        before, len(out), len(out) / before * 100 if before else 0,
    )
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Filter 2-hop data to rows whose intermediate gene "
                    "is in an endothelial gene list.",
    )
    ap.add_argument("--input", required=True, help="Input 2-hop CSV.")
    ap.add_argument("--gene-list", required=True,
                    help="CSV with a 'gene' column of endothelial genes.")
    ap.add_argument("--gene-column", default="gene",
                    help="Column name in the gene list CSV.")
    ap.add_argument("--output", required=True,
                    help="Output filtered CSV.")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.input, low_memory=False)
    logger.info("Loaded %d rows from %s", len(df), args.input)

    genes = load_gene_set(args.gene_list, column=args.gene_column)
    df_filtered = filter_by_intermediate(df, genes)

    df_filtered.to_csv(args.output, index=False)
    logger.info("Saved -> %s", args.output)


if __name__ == "__main__":
    main()
