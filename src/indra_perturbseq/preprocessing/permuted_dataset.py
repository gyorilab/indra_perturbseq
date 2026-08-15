"""Create a permuted (shuffled source labels) negative-control dataset.
Reads per-gene DEG CSVs, collects significant source-target pairs, shuffles.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_source_target_pairs(
    deg_dir: Path,
    source_genes: list[str],
    *,
    pval_threshold: float = 0.05,
) -> pd.DataFrame:
    """Load DEG results and collect significant source-target pairs.

    Parameters
    ----------
    deg_dir : Path
        Directory containing ``<GENE>_vs_control.csv`` files.
    source_genes : list[str]
        Gene names to process.
    pval_threshold : float
        P-value cutoff for significance.

    Returns
    -------
    pd.DataFrame
        Columns: ``original_source``, ``target``, ``logfoldchange``, ``pval``.
    """
    rows: list[dict[str, object]] = []
    for gene in source_genes:
        csv_path = deg_dir / f"{gene}_vs_control.csv"
        if not csv_path.exists():
            logger.warning("Missing DE file for %s, skipping", gene)
            continue
        df = pd.read_csv(csv_path)
        df = df[df["pvals"] < pval_threshold]
        logger.info("%s: %d targets with p < %g", gene, len(df), pval_threshold)
        for _, row in df.iterrows():
            rows.append({
                "original_source": gene,
                "target": row["names"],
                "logfoldchange": row["logfoldchanges"],
                "pval": row["pvals"],
            })
    return pd.DataFrame(rows)


def permute_sources(
    targets_df: pd.DataFrame,
    source_genes: list[str],
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Shuffle source labels and remove accidental true matches.

    Parameters
    ----------
    targets_df : pd.DataFrame
        DataFrame with ``original_source`` and ``target`` columns.
    source_genes : list[str]
        Pool of source genes to sample from.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Permuted dataset with columns ``source``, ``target``,
        ``logfoldchange``, ``pval``, ``original_source``.
    """
    rng = np.random.default_rng(seed)
    targets_df = targets_df.copy()
    targets_df["permuted_source"] = rng.choice(source_genes, size=len(targets_df))

    true_pairs = set(zip(targets_df["original_source"], targets_df["target"]))
    is_true = targets_df.apply(
        lambda r: (r["permuted_source"], r["target"]) in true_pairs, axis=1,
    )
    n_removed = int(is_true.sum())
    logger.info("Removed %d accidental true matches", n_removed)

    result = targets_df[~is_true].copy()
    result = result.rename(columns={"permuted_source": "source"})
    return result[["source", "target", "logfoldchange", "pval", "original_source"]]


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
        description="Create permuted negative-control dataset from DEG results.",
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="CSV with columns Gene and analysis_flag (target validation file).",
    )
    parser.add_argument("--deg-dir", required=True, type=Path, help="Directory with per-gene DEG CSVs.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV for permuted pairs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--exclude-genes", nargs="*", default=["TP53", "CDKN1A"],
        help="Genes to exclude as sources (default: TP53 CDKN1A).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for permuted-dataset CLI."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Loading validation file from %s", args.input)
    perturb_df = pd.read_csv(args.input)
    perturb_df = perturb_df[perturb_df["analysis_flag"] == "Use_for_analysis"]
    perturb_df = perturb_df[~perturb_df["Gene"].isin(args.exclude_genes)]

    source_genes = perturb_df["Gene"].tolist()
    logger.info(
        "%d source genes (excluded: %s)", len(source_genes), args.exclude_genes,
    )

    targets_df = load_source_target_pairs(args.deg_dir, source_genes)
    if targets_df.empty:
        logger.error("No source-target pairs found; check --deg-dir")
        return

    logger.info(
        "Loaded %d source-target pairs (%d unique sources, %d unique targets)",
        len(targets_df),
        targets_df["original_source"].nunique(),
        targets_df["target"].nunique(),
    )

    permuted = permute_sources(targets_df, source_genes, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    permuted.to_csv(args.output, index=False)
    logger.info(
        "Saved %d permuted pairs (%d unique sources, %d unique targets) to %s",
        len(permuted),
        permuted["source"].nunique(),
        permuted["target"].nunique(),
        args.output,
    )


if __name__ == "__main__":
    main()
