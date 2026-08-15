"""Visualize per-gene cell counts from an scRNA-seq AnnData object.

Reads an h5ad file, tallies cells per perturbation gene, and writes a
summary CSV (and optionally a bar-chart image).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)


def compute_cell_counts(
    adata: sc.AnnData,
    gene_column: str = "Gene",
) -> pd.DataFrame:
    """Count cells per perturbation gene.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object whose ``.obs`` contains *gene_column*.
    gene_column : str
        Column in ``adata.obs`` storing perturbation labels.

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame (``gene``, ``cell_count``) sorted descending.

    Raises
    ------
    ValueError
        If *gene_column* is missing from ``adata.obs``.
    """
    if gene_column not in adata.obs.columns:
        raise ValueError(
            f"Column '{gene_column}' not in adata.obs. "
            f"Available: {adata.obs.columns.tolist()}"
        )
    counts = adata.obs[gene_column].value_counts()
    df = pd.DataFrame({"gene": counts.index, "cell_count": counts.values})
    return df.sort_values("cell_count", ascending=False).reset_index(drop=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] or None
        Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Compute and export per-gene cell counts from scRNA-seq AnnData.",
    )
    parser.add_argument("--adata", required=True, type=Path, help="Path to .h5ad AnnData file.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path.")
    parser.add_argument("--gene-column", default="Gene", help="adata.obs column with perturbation labels.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for cell-count CLI."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Reading AnnData from %s", args.adata)
    adata = sc.read_h5ad(args.adata)

    df = compute_cell_counts(adata, gene_column=args.gene_column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved cell counts (%d genes) to %s", len(df), args.output)


if __name__ == "__main__":
    main()
