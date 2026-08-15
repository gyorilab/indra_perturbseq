"""Build a combined endothelial gene list from auto-detected and manual genes.

Auto-detection reads an scRNA-seq AnnData file, identifies endothelial-like
cells via canonical markers, and extracts genes expressed above a minimum
threshold in those cells.  The result is merged with a curated manual gene
list to produce a deduplicated CSV.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)

ENDOTHELIAL_MARKERS: list[str] = ["PECAM1", "CDH5", "KDR", "VWF", "CLDN5"]

MANUAL_ENDOTHELIAL_GENES: list[str] = [
    "BCAR1", "BMP1", "CALCRL", "CCM2", "CDKN1A", "CDKN2B", "CFDP1",
    "COL4A1", "COL4A2", "EXOC3L2", "FBN2", "FGD6", "FLT1", "FURIN",
    "GDPD5", "GGT5", "GOSR2", "IBTK", "LAMB2", "LOX", "MORF4L1",
    "N4BP2L2", "NOS3", "PALLD", "PECAM1", "PGF", "PLPP3", "PREX1",
    "PRKAR1A", "SCUBE1", "SERPINH1", "SH3PXD2A", "SLK", "SMAD3",
    "SPRY4", "SVIL", "SWAP70", "TFPI", "TLNRD1", "TSPAN14", "ZEB2",
]


def detect_present_genes(
    adata: sc.AnnData,
    *,
    score_percentile: float = 85.0,
    min_cells: int = 3,
    min_percent: float = 0.5,
) -> list[str]:
    """Identify genes expressed in endothelial-like cells of *adata*.

    Parameters
    ----------
    adata : sc.AnnData
        Raw-count scRNA-seq AnnData object.
    score_percentile : float
        Percentile threshold for classifying cells as endothelial based on
        average marker expression.
    min_cells : int
        A gene must be expressed in at least this many endothelial cells.
    min_percent : float
        A gene must be expressed in at least this percentage of endothelial
        cells.

    Returns
    -------
    list[str]
        Sorted list of gene names passing the expression filters.
    """
    adata = adata.copy()
    available_markers = [g for g in ENDOTHELIAL_MARKERS if g in adata.var_names]
    if not available_markers:
        raise ValueError(
            "None of the endothelial markers found in adata.var_names: "
            f"{ENDOTHELIAL_MARKERS}"
        )
    logger.info("Available endothelial markers: %s", available_markers)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    scores = np.array(adata[:, available_markers].X.mean(axis=1)).ravel()
    threshold = np.percentile(scores, score_percentile)
    is_endo = scores >= threshold
    n_endo = int(is_endo.sum())
    logger.info(
        "Identified %d endothelial-like cells (%.2f%% of %d total)",
        n_endo,
        n_endo / adata.n_obs * 100,
        adata.n_obs,
    )

    adata_endo = adata[is_endo]
    expr_mask = adata_endo.X > 0
    cells_per_gene = np.array(expr_mask.sum(axis=0)).ravel()
    pct = (cells_per_gene / adata_endo.n_obs) * 100
    mask = (cells_per_gene >= min_cells) & (pct >= min_percent)
    present = sorted(adata_endo.var_names[mask].tolist())
    logger.info(
        "%d genes pass filters (>=%d cells, >=%.1f%%)", len(present), min_cells, min_percent
    )
    return present


def load_manual_genes(path: Path | None) -> list[str]:
    """Load manual gene names from a text file (one per line).

    Parameters
    ----------
    path : Path or None
        Path to a plain-text file with one gene name per line.  If *None*,
        falls back to :data:`MANUAL_ENDOTHELIAL_GENES`.

    Returns
    -------
    list[str]
        Gene name list.
    """
    if path is None:
        logger.info("Using built-in MANUAL_ENDOTHELIAL_GENES (%d genes)", len(MANUAL_ENDOTHELIAL_GENES))
        return list(MANUAL_ENDOTHELIAL_GENES)
    genes = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    logger.info("Loaded %d manual genes from %s", len(genes), path)
    return genes


def build_combined_gene_list(
    present_genes: list[str],
    manual_genes: list[str],
) -> list[str]:
    """Merge and deduplicate two gene lists.

    Parameters
    ----------
    present_genes : list[str]
        Auto-detected present genes.
    manual_genes : list[str]
        Manually curated genes.

    Returns
    -------
    list[str]
        Sorted, deduplicated union.
    """
    combined = sorted(set(present_genes) | set(manual_genes))
    present_set = set(present_genes)
    manual_set = set(manual_genes)
    overlap = present_set & manual_set
    missing = manual_set - present_set
    logger.info(
        "Combined: %d genes | manual total: %d | overlap: %d | missing from detected: %d",
        len(combined),
        len(manual_set),
        len(overlap),
        len(missing),
    )
    if missing:
        logger.debug("Manual genes absent from detected set: %s", sorted(missing))
    return combined


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
        description="Build combined endothelial gene list from scRNA-seq auto-detection and manual curation.",
    )
    parser.add_argument("--adata", required=True, type=Path, help="Path to .h5ad AnnData file.")
    parser.add_argument(
        "--manual-genes",
        type=Path,
        default=None,
        help="Text file with one gene name per line. Falls back to built-in list.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the gene-list builder CLI."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Reading AnnData from %s", args.adata)
    adata = sc.read_h5ad(args.adata)

    present_genes = detect_present_genes(adata)
    manual_genes = load_manual_genes(args.manual_genes)
    combined = build_combined_gene_list(present_genes, manual_genes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(combined, name="gene").to_csv(args.output, index=False)
    logger.info("Saved %d genes to %s", len(combined), args.output)


if __name__ == "__main__":
    main()
