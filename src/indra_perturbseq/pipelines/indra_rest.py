"""Direct pathway analysis via INDRA REST API.
For each source perturbation gene, identifies significantly affected.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import pandas as pd
import scanpy as sc
from indra.statements import Activation, DecreaseAmount, IncreaseAmount, Inhibition
from indra_perturbseq.gene_lists import load_source_genes
from indra_perturbseq.runtime import add_log_level_arg, configure_logging
from indra_perturbseq.services.indra_db import safe_get_statements

logger = logging.getLogger(__name__)


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


# ------------------------------------------------------------------
# Gene name helpers
# ------------------------------------------------------------------

def _clean_gene_name(name: str | float | None) -> str | None:
    """Normalise a gene symbol to uppercase, stripping TSS suffixes."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    s = str(name).upper().strip()
    if "-TSS" in s:
        s = s.split("-TSS")[0]
    return s or None


def _is_unusual_gene_name(name: str) -> bool:
    """Return ``True`` for Ensembl IDs, TSS suffixes, or non-uppercase names."""
    return name != name.upper() or "-TSS" in name or name.startswith("ENSG")


_STMT_TYPE_MAP = {
    IncreaseAmount: "increaseamount",
    DecreaseAmount: "decreaseamount",
    Activation: "increaseamount",
    Inhibition: "decreaseamount",
}


def _check_consistency(literature_effect: str, observed_effect: str) -> bool:
    """Check if the literature relationship is consistent with knockdown.

    For a knockdown experiment the expected pattern is:
    - literature says source *increases* target => knockdown causes *decrease*
    - literature says source *decreases* target => knockdown causes *increase*
    """
    return (
        (literature_effect == "increaseamount" and observed_effect == "decrease")
        or (literature_effect == "decreaseamount" and observed_effect == "increase")
    )


# ------------------------------------------------------------------
# Core analysis
# ------------------------------------------------------------------

def get_affected_genes(
    adata_de,
    target_gene: str,
    p_threshold: float = 0.05,
    max_genes: int | None = None,
) -> list[dict]:
    """Return significantly affected genes for *target_gene*.

    Tries successively relaxed FDR thresholds (0.05, 0.1, 0.2) if the
    strict threshold yields fewer than 10 genes.

    Parameters
    ----------
    adata_de :
        AnnData object with ``rank_genes_groups`` results.
    target_gene :
        Source perturbation gene symbol.
    p_threshold :
        Initial adjusted p-value threshold.
    max_genes :
        If set, cap the number of returned genes.

    Returns
    -------
    :
        List of dicts with keys ``gene``, ``logfc``, ``pvalue``,
        ``effect_direction``.
    """
    gene = _clean_gene_name(target_gene) or target_gene.upper()
    try:
        de_results = sc.get.rank_genes_groups_df(adata_de, group=gene)
    except Exception:
        logger.warning("No DE results for %s", gene)
        return []

    downstream = de_results[de_results["names"].str.upper() != gene]
    significant = pd.DataFrame()

    for thresh in (p_threshold, 0.1, 0.2):
        significant = downstream[downstream["pvals_adj"] < thresh]
        if len(significant) >= 10:
            break

    if significant.empty:
        return []

    significant = significant.sort_values("pvals_adj")
    if max_genes is not None:
        significant = significant.head(max_genes)

    affected: list[dict] = []
    for _, row in significant.iterrows():
        name = _clean_gene_name(row["names"])
        if name and not _is_unusual_gene_name(name):
            affected.append({
                "gene": name,
                "logfc": row["logfoldchanges"],
                "pvalue": row["pvals_adj"],
                "effect_direction": "increase" if row["logfoldchanges"] > 0 else "decrease",
            })
    return affected


def query_direct_pathways(
    source_gene: str,
    target_gene: str,
    evidence_limit: int = 10,
    rate_limit: float = 1.0,
) -> list[dict]:
    """Query INDRA REST API for direct relationships.

    Parameters
    ----------
    source_gene :
        Source gene symbol.
    target_gene :
        Target gene symbol.
    evidence_limit :
        Maximum evidence items per query.
    rate_limit :
        Sleep (seconds) between queries.

    Returns
    -------
    :
        List of pathway dicts.
    """
    source = _clean_gene_name(source_gene) or source_gene.upper()
    target = _clean_gene_name(target_gene) or target_gene.upper()
    if _is_unusual_gene_name(target):
        return []

    ip = safe_get_statements(subject=source, object=target, ev_limit=evidence_limit)
    time.sleep(rate_limit)
    if ip is None:
        return []

    pathways: list[dict] = []
    for stmt in ip.statements:
        edge_type = _STMT_TYPE_MAP.get(type(stmt))
        if edge_type is None:
            continue
        pathways.append({
            "source": source,
            "target": target,
            "relationship_type": type(stmt).__name__,
            "evidence_count": len(stmt.evidence),
            "pathway_string": f"{source} -> {target}",
            "final_edge_type": edge_type,
        })
    return pathways


def analyze_perturbation(
    adata_de,
    target_gene: str,
    max_genes: int | None = 50,
    evidence_limit: int = 10,
    rate_limit: float = 1.0,
) -> pd.DataFrame:
    """Analyze all direct pathways for a single perturbation gene.

    Parameters
    ----------
    adata_de :
        AnnData object with ``rank_genes_groups`` results.
    target_gene :
        Source perturbation gene.
    max_genes :
        Cap on downstream genes to query.
    evidence_limit :
        Evidence limit per INDRA REST query.
    rate_limit :
        Inter-query sleep.

    Returns
    -------
    :
        DataFrame of discovered pathways with consistency annotations.
    """
    gene = _clean_gene_name(target_gene) or target_gene.upper()
    affected = get_affected_genes(adata_de, gene, max_genes=max_genes)
    if not affected:
        return pd.DataFrame()

    all_pathways: list[dict] = []
    for i, info in enumerate(affected, start=1):
        logger.info("  [%d/%d] %s -> %s", i, len(affected), gene, info["gene"])
        pathways = query_direct_pathways(gene, info["gene"], evidence_limit=evidence_limit, rate_limit=rate_limit)
        for p in pathways:
            p.update({
                "perturbation_gene": gene,
                "affected_gene": info["gene"],
                "observed_effect": info["effect_direction"],
                "observed_logfc": info["logfc"],
                "observed_pvalue": info["pvalue"],
                "literature_experiment_consistent": _check_consistency(
                    p["final_edge_type"], info["effect_direction"],
                ),
            })
        all_pathways.extend(pathways)

    logger.info(
        "%s: %d pathways from %d affected genes",
        gene, len(all_pathways), len(affected),
    )
    return pd.DataFrame(all_pathways)


def run_all_perturbations(
    adata_de,
    genes: list[str],
    max_genes_per_perturbation: int | None = 50,
    evidence_limit: int = 10,
    rate_limit: float = 1.0,
) -> pd.DataFrame:
    """Run direct pathway analysis for all *genes*.

    Parameters
    ----------
    adata_de :
        AnnData object with ``rank_genes_groups`` results.
    genes :
        Perturbation gene symbols.
    max_genes_per_perturbation :
        Cap on downstream targets per source gene.
    evidence_limit :
        Evidence limit per query.
    rate_limit :
        Inter-query sleep in seconds.

    Returns
    -------
    :
        Combined pathway DataFrame.
    """
    frames: list[pd.DataFrame] = []
    for i, gene in enumerate(genes, start=1):
        logger.info("Perturbation %d/%d: %s", i, len(genes), gene)
        df = analyze_perturbation(
            adata_de, gene,
            max_genes=max_genes_per_perturbation,
            evidence_limit=evidence_limit,
            rate_limit=rate_limit,
        )
        if not df.empty:
            frames.append(df)

    if not frames:
        logger.warning("No pathways discovered for any perturbation")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _prepare_adata(
    adata_path: str,
    genes: list[str],
    min_cells: int = 100,
    controls: list[str] | None = None,
) -> tuple:
    """Load AnnData, run DE, and return ``(adata_de, robust_genes)``.

    Parameters
    ----------
    adata_path :
        Path to ``.h5ad`` file.
    genes :
        Candidate perturbation genes.
    min_cells :
        Minimum cell count per perturbation.
    controls :
        Control group labels.

    Returns
    -------
    :
        ``(adata_de, robust_genes)`` after running ``rank_genes_groups``.
    """
    if controls is None:
        controls = ["safe-targeting", "negative-control"]

    adata = sc.read_h5ad(adata_path)
    adata.obs["Gene_Clean"] = adata.obs["Gene"].astype(str)

    cell_counts = adata.obs["Gene_Clean"].value_counts()
    robust = [
        g for g in genes
        if g in cell_counts.index and cell_counts[g] >= min_cells
    ]
    logger.info("Robust genes (>=%d cells): %d / %d", min_cells, len(robust), len(genes))

    adata_de = adata[adata.obs["Gene_Clean"].isin(robust + controls)].copy()
    adata_de.obs["group_label"] = adata_de.obs["Gene_Clean"].apply(
        lambda x: "control" if x in controls else x,
    )

    logger.info("Running Wilcoxon DE analysis...")
    sc.tl.rank_genes_groups(
        adata_de, groupby="group_label", method="wilcoxon",
        reference="control", pts=True,
    )
    logger.info("DE analysis complete.")
    return adata_de, robust


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI entry point for INDRA REST direct pathway analysis."""
    ap = argparse.ArgumentParser(
        description="Direct pathway analysis via INDRA REST API.",
    )
    add_log_level_arg(ap, default="INFO")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--adata", help="Path to .h5ad with raw counts")
    group.add_argument("--adata-de",
                       help="Path to .h5ad with pre-computed rank_genes_groups")

    ap.add_argument("--source-genes-csv", required=True,
                    help="target_validation_expanded.csv")
    ap.add_argument("--output", required=True,
                    help="Output CSV for pathway results")

    ap.add_argument("--filter-column", default="analysis_flag")
    ap.add_argument("--filter-value", default="Use_for_analysis")
    ap.add_argument("--gene-column", default="Gene")
    ap.add_argument("--genes", nargs="+",
                    help="Explicit source genes (overrides CSV)")
    ap.add_argument("--limit-genes", type=int, default=0)
    ap.add_argument("--max-genes-per-perturbation", type=int, default=50)
    ap.add_argument("--min-cells", type=int, default=100)
    ap.add_argument("--evidence-limit", type=int, default=10)
    ap.add_argument("--rate-limit", type=float, default=1.0)
    args = ap.parse_args(argv)
    configure_logging(args.log_level)

    genes = load_source_genes(
        args.source_genes_csv,
        gene_column=args.gene_column,
        filter_column=args.filter_column,
        filter_value=args.filter_value,
        explicit_genes=args.genes,
        limit=args.limit_genes,
    )

    if args.adata_de:
        logger.info("Loading pre-computed DE AnnData: %s", args.adata_de)
        adata_de = sc.read_h5ad(args.adata_de)
    else:
        adata_de, genes = _prepare_adata(args.adata, genes, min_cells=args.min_cells)

    results = run_all_perturbations(
        adata_de, genes,
        max_genes_per_perturbation=args.max_genes_per_perturbation,
        evidence_limit=args.evidence_limit,
        rate_limit=args.rate_limit,
    )

    if results.empty:
        logger.warning("No pathways discovered. Output will be empty.")

    _ensure_parent_dir(args.output)
    results.to_csv(args.output, index=False)
    logger.info("Results: %d rows -> %s", len(results), args.output)


if __name__ == "__main__":
    main()
