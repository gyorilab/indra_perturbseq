"""Compare bulk RNA-seq vs scRNA-seq DEG overlap.

For each source gene, loads bulk and scRNA-seq DEG results, restricts to
an endothelial gene universe, and computes overlap statistics (Jaccard
index, intersection, fraction of overlap) for both the tested gene sets
and the FDR-significant gene sets.
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_gene_universe(path: str, gene_column: str = "gene") -> set[str]:
    """Load gene symbols from a CSV into a set.

    Parameters
    ----------
    path : str
        Path to CSV.
    gene_column : str
        Column name containing gene symbols.

    Returns
    -------
    set[str]
    """
    df = pd.read_csv(path, low_memory=False)
    if gene_column not in df.columns:
        raise ValueError(
            f"Gene universe CSV must have '{gene_column}' column. "
            f"Got: {df.columns.tolist()}"
        )
    s = df[gene_column].astype(str).str.strip()
    return set(s[s != ""])


def compare_deg_overlap(
    bulk_csv: str,
    scrna_csv: str,
    gene_universe: set[str],
    fdr_threshold: float,
) -> dict:
    """Compute overlap statistics between bulk and scRNA-seq DEGs.

    Parameters
    ----------
    bulk_csv : str
        Bulk RNA-seq DEG CSV (must have ``names``, ``pvals_adj``).
    scrna_csv : str
        scRNA-seq DEG CSV (must have ``names``, ``pvals_adj``).
    gene_universe : set[str]
        Restrict to this gene set.
    fdr_threshold : float
        Significance threshold for adjusted p-values.

    Returns
    -------
    dict
        Overlap statistics.
    """
    bulk = pd.read_csv(bulk_csv, low_memory=False)
    scrna = pd.read_csv(scrna_csv, low_memory=False)

    for df in (bulk, scrna):
        df["names"] = df["names"].astype(str).str.strip()
        df["pvals_adj"] = pd.to_numeric(df["pvals_adj"], errors="coerce")

    bulk = bulk[bulk["names"].isin(gene_universe)]
    scrna = scrna[scrna["names"].isin(gene_universe)]

    bulk_tested = set(bulk.loc[bulk["pvals_adj"].notna(), "names"])
    scrna_tested = set(scrna.loc[scrna["pvals_adj"].notna(), "names"])
    tested_inter = bulk_tested & scrna_tested
    tested_union = bulk_tested | scrna_tested

    bulk_sig = set(bulk.loc[bulk["pvals_adj"] < fdr_threshold, "names"])
    scrna_sig = set(scrna.loc[scrna["pvals_adj"] < fdr_threshold, "names"])
    sig_inter = bulk_sig & scrna_sig
    sig_union = bulk_sig | scrna_sig

    def _jaccard(inter: set, union: set) -> float:
        return (len(inter) / len(union)) if union else np.nan

    return {
        "bulk_tested_endo": len(bulk_tested),
        "scrna_tested_endo": len(scrna_tested),
        "tested_intersection": len(tested_inter),
        "tested_union": len(tested_union),
        "tested_jaccard": _jaccard(tested_inter, tested_union),
        "fdr_threshold": fdr_threshold,
        "bulk_sig_fdr": len(bulk_sig),
        "scrna_sig_fdr": len(scrna_sig),
        "sig_intersection": len(sig_inter),
        "sig_union": len(sig_union),
        "sig_jaccard": _jaccard(sig_inter, sig_union),
        "sig_overlap_frac_of_bulk": (
            len(sig_inter) / len(bulk_sig) if bulk_sig else np.nan
        ),
        "sig_overlap_frac_of_scrna": (
            len(sig_inter) / len(scrna_sig) if scrna_sig else np.nan
        ),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Compare bulk vs scRNA-seq DEG overlap per gene.",
    )
    ap.add_argument("--bulk-dir", required=True,
                    help="Directory with bulk <GENE>_vs_control.csv files")
    ap.add_argument("--scrna-dir", required=True,
                    help="Directory with scRNA-seq <GENE>_vs_control.csv files")
    ap.add_argument("--gene-universe-csv", required=True,
                    help="CSV with gene-universe column")
    ap.add_argument("--gene-universe-col", default="gene")
    ap.add_argument("--genes", nargs="+", required=True,
                    help="Source gene symbols to compare")
    ap.add_argument("--deg-suffix", default="_vs_control.csv")
    ap.add_argument("--fdr-threshold", type=float, default=0.05)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args(argv)

    gene_universe = load_gene_universe(
        args.gene_universe_csv, gene_column=args.gene_universe_col,
    )

    rows: list[dict] = []
    for gene in args.genes:
        bulk_path = os.path.join(args.bulk_dir, f"{gene}{args.deg_suffix}")
        scrna_path = os.path.join(args.scrna_dir, f"{gene}{args.deg_suffix}")
        stats = compare_deg_overlap(
            bulk_path, scrna_path, gene_universe, args.fdr_threshold,
        )
        stats["gene"] = gene
        rows.append(stats)
        logger.info("[%s] sig_jaccard=%.4f", gene, stats["sig_jaccard"])

    summary = pd.DataFrame(rows).sort_values("gene")
    summary.to_csv(args.output_csv, index=False)
    logger.info("Wrote %s", args.output_csv)
    logger.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
