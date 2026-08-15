"""Load and normalize gene lists used by analysis pipelines.
Supports filtered source genes and generic gene-set CSV inputs."""

from __future__ import annotations

import logging

import pandas as pd

from indra_perturbseq.hgnc import normalize_hgnc_symbol

logger = logging.getLogger(__name__)


def load_gene_set(path: str, gene_column: str = "gene") -> set[str]:
    """Load a gene set from a CSV, normalizing symbols via HGNC.

    Parameters
    ----------
    path :
        CSV file path.
    gene_column :
        Column containing gene symbols.
    """
    df = pd.read_csv(path, low_memory=False)
    if gene_column not in df.columns:
        raise ValueError(
            f"CSV must have column '{gene_column}'. "
            f"Columns: {df.columns.tolist()}"
        )
    raw = set(df[gene_column].astype(str).str.strip())
    raw.discard("")
    out = {normalize_hgnc_symbol(g) or g for g in raw}
    out.discard("")
    logger.info("Loaded %d genes from %s (col=%s)", len(out), path, gene_column)
    return out


def load_source_genes(
    source_genes_csv: str,
    gene_column: str = "Gene",
    filter_column: str = "analysis_flag",
    filter_value: str = "Use_for_analysis",
    explicit_genes: list[str] | None = None,
    limit: int = 0,
) -> list[str]:
    """Load source-gene list from the target-validation CSV.

    Parameters
    ----------
    source_genes_csv :
        Path to ``target_validation_expanded.csv``.
    gene_column :
        Column with gene symbols.
    filter_column :
        Column used for filtering.
    filter_value :
        Required value in *filter_column*.
    explicit_genes :
        If provided, use these instead of reading from CSV.
    limit :
        If > 0, truncate to this many genes.
    """
    if explicit_genes:
        raw_genes = [s.strip() for s in explicit_genes if s.strip()]
    else:
        df = pd.read_csv(source_genes_csv, low_memory=False)
        if filter_column in df.columns:
            df = df[df[filter_column] == filter_value].copy()
        raw_genes = [
            s.strip()
            for s in df[gene_column].dropna().astype(str).tolist()
            if s.strip()
        ]

    seen: set[str] = set()
    genes: list[str] = []
    for g in raw_genes:
        n = normalize_hgnc_symbol(g)
        if n and n not in seen:
            genes.append(n)
            seen.add(n)

    if limit > 0:
        genes = genes[:limit]
    logger.info("Source genes to process: %d", len(genes))
    return genes


def load_filtered_sources(
    target_validation: str,
    source_col: str = "Gene",
    filter_column: str = "analysis_flag",
    filter_value: str = "Use_for_analysis",
) -> list[str]:
    """Load unique, normalized filtered source genes.

    Returns a deduplicated list preserving first-occurrence order.
    """
    tv = pd.read_csv(target_validation, low_memory=False)
    for c in (source_col, filter_column):
        if c not in tv.columns:
            raise ValueError(f"Missing column '{c}' in {target_validation}")
    tv = tv[tv[filter_column] == filter_value].copy()
    raw = [s.strip() for s in tv[source_col].dropna().astype(str) if s.strip()]

    seen: set[str] = set()
    out: list[str] = []
    for r in raw:
        n = normalize_hgnc_symbol(r)
        if n and n not in seen:
            out.append(n)
            seen.add(n)
    return out
