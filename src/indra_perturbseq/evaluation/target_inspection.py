"""Count how many source-gene parents each DEG target has.

For every source gene that passes a flag filter, loads its DEG CSV,
identifies significant targets (p < threshold), and records which
sources each target is a DEG of.  Outputs a CSV with target, parent
count, and semicolon-separated parent list.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_source_genes(
    target_validation: str,
    gene_column: str = "Gene",
    filter_column: str = "analysis_flag",
    filter_value: str = "Use_for_analysis",
) -> list[str]:
    """Load filtered source gene symbols from a target-validation CSV.

    Parameters
    ----------
    target_validation : str
        Path to ``target_validation_expanded.csv`` or equivalent.
    gene_column : str
        Column containing gene symbols.
    filter_column : str
        Column used for filtering.
    filter_value : str
        Required value in *filter_column*.

    Returns
    -------
    list[str]
        Deduplicated, stripped gene symbols.
    """
    df = pd.read_csv(target_validation, low_memory=False)
    return (
        df.loc[df[filter_column] == filter_value, gene_column]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )


def build_parent_map(
    sources: list[str],
    deg_dir: str,
    pval_threshold: float = 0.05,
    deg_suffix: str = "_vs_control.csv",
    ignore_targets: set[str] | None = None,
) -> dict[str, set[str]]:
    """Build a mapping from target gene to set of parent source genes.

    Parameters
    ----------
    sources : list[str]
        Source gene symbols.
    deg_dir : str
        Directory containing DEG CSV files.
    pval_threshold : float
        Significance threshold on ``pvals`` column.
    deg_suffix : str
        Filename suffix for DEG files.
    ignore_targets : set[str] or None
        Target symbols to exclude (e.g. hub genes).

    Returns
    -------
    dict[str, set[str]]
        Mapping of target gene to set of source (parent) genes.
    """
    if ignore_targets is None:
        ignore_targets = set()
    ignore_upper = {g.upper() for g in ignore_targets}

    parents: dict[str, set[str]] = defaultdict(set)
    missing = 0

    for src in sources:
        deg_path = os.path.join(deg_dir, f"{src}{deg_suffix}")
        if not os.path.exists(deg_path):
            missing += 1
            continue

        deg = pd.read_csv(deg_path, usecols=["names", "pvals"],
                          low_memory=False)
        deg = deg.dropna(subset=["names", "pvals"])
        deg["pvals"] = pd.to_numeric(deg["pvals"], errors="coerce")
        deg = deg[deg["pvals"] < pval_threshold]

        for tgt in deg["names"].astype(str).str.strip().unique():
            if tgt and tgt.upper() not in ignore_upper:
                parents[tgt].add(src)

    logger.info("Sources used: %d (missing DEG files: %d)",
                len(sources) - missing, missing)
    return dict(parents)


def parent_map_to_dataframe(
    parents: dict[str, set[str]],
) -> pd.DataFrame:
    """Convert a parent map to a sorted DataFrame.

    Parameters
    ----------
    parents : dict[str, set[str]]

    Returns
    -------
    pd.DataFrame
        Columns: ``target``, ``n_parents``, ``parents``.
    """
    return (
        pd.DataFrame({
            "target": list(parents.keys()),
            "n_parents": [len(v) for v in parents.values()],
            "parents": [";".join(sorted(v)) for v in parents.values()],
        })
        .sort_values(["n_parents", "target"], ascending=[False, True])
        .reset_index(drop=True)
    )


def log_parent_stats(df: pd.DataFrame) -> None:
    """Log summary statistics about parent counts.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``n_parents`` column.
    """
    arr = df["n_parents"].to_numpy()
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    iqr = q3 - q1
    hi_cut = q3 + 1.5 * iqr

    logger.info("Targets: %d", len(df))
    logger.info(
        "n_parents min/median/mean/max: %d / %.1f / %.2f / %d",
        arr.min(), np.median(arr), arr.mean(), arr.max(),
    )
    logger.info("High-outlier cutoff (Q3+1.5*IQR): %.1f", hi_cut)
    logger.info("Targets with 1 parent: %d", int((arr == 1).sum()))
    logger.info("Targets with 2 parents: %d", int((arr == 2).sum()))
    logger.info("High outliers (n_parents > %.1f): %d",
                hi_cut, int((arr > hi_cut).sum()))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Count source-gene parents per DEG target.",
    )
    ap.add_argument("--target-validation", required=True,
                    help="target_validation_expanded.csv or equivalent")
    ap.add_argument("--deg-dir", required=True,
                    help="Directory with <GENE>_vs_control.csv files")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--pval-threshold", type=float, default=0.05)
    ap.add_argument("--deg-suffix", default="_vs_control.csv")
    ap.add_argument("--gene-column", default="Gene")
    ap.add_argument("--filter-column", default="analysis_flag")
    ap.add_argument("--filter-value", default="Use_for_analysis")
    ap.add_argument("--ignore-targets", nargs="*",
                    default=["TP53", "CDKN1A"],
                    help="Target genes to exclude")
    args = ap.parse_args(argv)

    sources = load_source_genes(
        args.target_validation,
        gene_column=args.gene_column,
        filter_column=args.filter_column,
        filter_value=args.filter_value,
    )
    parents = build_parent_map(
        sources,
        args.deg_dir,
        pval_threshold=args.pval_threshold,
        deg_suffix=args.deg_suffix,
        ignore_targets=set(args.ignore_targets) if args.ignore_targets else None,
    )
    df = parent_map_to_dataframe(parents)
    df.to_csv(args.output_csv, index=False)
    logger.info("Wrote %s", args.output_csv)
    log_parent_stats(df)


if __name__ == "__main__":
    main()
