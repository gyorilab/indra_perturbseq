"""Loading and filtering of gene lists (endothelial, Karen sources, etc.)."""

from __future__ import annotations

import logging

import pandas as pd

from indra_perturbseq.hgnc import normalize_hgnc_symbol

logger = logging.getLogger(__name__)


def load_gene_set(path: str, gene_col: str = "gene") -> set[str]:
    """Load a gene set from a CSV, normalizing symbols via HGNC.

    Parameters
    ----------
    path :
        CSV file path.
    gene_col :
        Column containing gene symbols.
    """
    df = pd.read_csv(path, low_memory=False)
    if gene_col not in df.columns:
        raise ValueError(
            f"CSV must have column '{gene_col}'. "
            f"Columns: {df.columns.tolist()}"
        )
    raw = set(df[gene_col].astype(str).str.strip())
    raw.discard("")
    out = {normalize_hgnc_symbol(g) or g for g in raw}
    out.discard("")
    logger.info("Loaded %d genes from %s (col=%s)", len(out), path, gene_col)
    return out


def load_source_genes(
    genes_csv: str,
    gene_col: str = "Gene",
    flag_col: str = "Karen_Flag",
    flag_value: str = "Use_for_analysis",
    explicit_genes: list[str] | None = None,
    limit: int = 0,
) -> list[str]:
    """Load source-gene list from the target-validation CSV.

    Parameters
    ----------
    genes_csv :
        Path to ``target_validation_expanded.csv``.
    gene_col :
        Column with gene symbols.
    flag_col :
        Column used for filtering (e.g. Karen_Flag).
    flag_value :
        Required value in *flag_col*.
    explicit_genes :
        If provided, use these instead of reading from CSV.
    limit :
        If > 0, truncate to this many genes.
    """
    if explicit_genes:
        genes = [s.strip() for s in explicit_genes if s.strip()]
    else:
        df = pd.read_csv(genes_csv, low_memory=False)
        if flag_col in df.columns:
            df = df[df[flag_col] == flag_value].copy()
        genes = [
            s.strip()
            for s in df[gene_col].dropna().astype(str).tolist()
            if s.strip()
        ]
    if limit > 0:
        genes = genes[:limit]
    logger.info("Source genes to process: %d", len(genes))
    return genes


def load_karen_sources(
    tv_path: str,
    source_col: str = "Gene",
    flag_col: str = "Karen_Flag",
    flag_value: str = "Use_for_analysis",
) -> list[str]:
    """Load unique, normalized Karen-flagged source genes.

    Returns a deduplicated list preserving first-occurrence order.
    """
    tv = pd.read_csv(tv_path, low_memory=False)
    for c in (source_col, flag_col):
        if c not in tv.columns:
            raise ValueError(f"Missing column '{c}' in {tv_path}")
    tv = tv[tv[flag_col] == flag_value].copy()
    raw = [s.strip() for s in tv[source_col].dropna().astype(str) if s.strip()]

    seen: set[str] = set()
    out: list[str] = []
    for r in raw:
        n = normalize_hgnc_symbol(r)
        if n and n not in seen:
            out.append(n)
            seen.add(n)
    return out
