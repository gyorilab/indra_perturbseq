"""OmniPath 1-hop coverage and synthetic 3-hop dataset creation.
Sub-commands.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import networkx as nx
import numpy as np
import pandas as pd

from indra_perturbseq.gene_lists import load_source_genes
from indra_perturbseq.runtime import add_log_level_arg, configure_logging

logger = logging.getLogger(__name__)


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


# 1-hop OmniPath coverage
def _build_uniprot_to_hgnc(uniprot_ids: set[str]) -> dict[str, str]:
    """Map UniProt accessions to HGNC symbols using INDRA clients."""
    from indra.databases import hgnc_client, uniprot_client

    mapping: dict[str, str] = {}
    for uid in uniprot_ids:
        hgnc_id = uniprot_client.get_hgnc_id(uid)
        if hgnc_id:
            symbol = hgnc_client.get_hgnc_name(hgnc_id)
            if symbol:
                mapping[uid] = symbol
    return mapping


def fetch_omnipath_graph(
    cache_path: str | None = None,
    force_reload: bool = False,
) -> nx.DiGraph:
    """Fetch OmniPath interactions, map UniProt -> HGNC, and build a graph.

    Parameters
    ----------
    cache_path :
        If provided, cache the mapped edge table as Parquet.
    force_reload :
        Re-download even if *cache_path* exists.

    Returns
    -------
    :
        Directed graph with ``sign`` edge attribute.
    """
    import omnipath as op

    if cache_path and os.path.exists(cache_path) and not force_reload:
        df = pd.read_parquet(cache_path)
    else:
        logger.info("Downloading OmniPath interactions...")
        df = op.interactions.OmniPath.get()
        df = df[["source", "target", "is_stimulation", "is_inhibition"]].dropna()

        all_ids = set(df["source"]).union(df["target"])
        uniprot_to_hgnc = _build_uniprot_to_hgnc(all_ids)

        df["source"] = df["source"].map(uniprot_to_hgnc)
        df["target"] = df["target"].map(uniprot_to_hgnc)
        df = df.dropna(subset=["source", "target"])

        if cache_path:
            _ensure_parent_dir(cache_path)
            df.to_parquet(cache_path)
            logger.info("Cached mapped OmniPath data: %s", cache_path)

    graph = nx.from_pandas_edgelist(
        df, source="source", target="target",
        edge_attr=["is_stimulation", "is_inhibition"],
        create_using=nx.DiGraph(),
    )
    for _, _, d in graph.edges(data=True):
        d["sign"] = 1 if d.get("is_stimulation") else (-1 if d.get("is_inhibition") else 0)

    logger.info(
        "OmniPath graph: %d nodes, %d edges",
        graph.number_of_nodes(), graph.number_of_edges(),
    )
    return graph


def _load_significant_descendants(
    file_path: str,
    pval_threshold: float = 0.05,
) -> dict[str, int]:
    """Load significant DEG targets with direction from a per-gene CSV."""
    df = pd.read_csv(file_path)
    sig = df[df["pvals"] < pval_threshold].copy()
    sig["direction"] = sig["logfoldchanges"].apply(lambda x: 1 if x > 0 else -1)
    return dict(zip(sig["names"], sig["direction"]))


def one_hop_coverage(
    graph: nx.DiGraph,
    perturbation: str,
    descendants: dict[str, int],
) -> tuple[list[str], int, int, float]:
    """Compute 1-hop coverage for a perturbation gene.

    Returns
    -------
    :
        ``(explained_list, explained_count, unexplained_count, coverage_fraction)``
    """
    if perturbation not in graph:
        return [], 0, len(descendants), 0.0

    explained = [d for d in descendants if graph.has_edge(perturbation, d)]
    n_exp = len(explained)
    n_unexp = len(descendants) - n_exp
    cov = n_exp / len(descendants) if descendants else 0.0
    return explained, n_exp, n_unexp, cov


def run_1hop_omnipath(
    genes: list[str],
    deg_dir: str,
    cache_path: str | None = None,
    force_reload: bool = False,
    pval_threshold: float = 0.05,
) -> pd.DataFrame:
    """Run OmniPath 1-hop coverage for all *genes*.

    Parameters
    ----------
    genes :
        Source perturbation gene symbols.
    deg_dir :
        Directory containing ``<GENE>_vs_control.csv`` files.
    cache_path :
        Parquet cache for OmniPath mapped edges.
    force_reload :
        Force re-download of OmniPath data.
    pval_threshold :
        Significance threshold for DEG targets.

    Returns
    -------
    :
        DataFrame with per-gene coverage statistics.
    """
    graph = fetch_omnipath_graph(cache_path=cache_path, force_reload=force_reload)
    results: list[dict] = []
    t0 = time.time()

    for i, gene in enumerate(genes, start=1):
        gene_upper = gene.upper()
        deg_path = os.path.join(deg_dir, f"{gene}_vs_control.csv")
        if not os.path.exists(deg_path):
            logger.debug("[%d/%d] SKIP %s: DEG file not found", i, len(genes), gene)
            continue

        descendants = _load_significant_descendants(deg_path, pval_threshold)
        explained_list, n_exp, n_unexp, cov = one_hop_coverage(graph, gene_upper, descendants)

        results.append({
            "perturbation": gene_upper,
            "total_descendants": len(descendants),
            "explained_count": n_exp,
            "not_explained_count": n_unexp,
            "coverage": cov,
            "explained_descendants": ",".join(explained_list) if explained_list else "",
        })

        elapsed = time.time() - t0
        logger.info(
            "[%d/%d] %s: explained=%d coverage=%.2f%% elapsed=%.1fm",
            i, len(genes), gene_upper, n_exp, cov * 100, elapsed / 60,
        )

    return pd.DataFrame(results)


# Synthetic 3-hop dataset creation

def create_synthetic_3hop(
    indra_csv: str,
    output_csv: str,
    coverage_fraction: float = 0.18,
    top_n_pairs: int = 20,
    n_selected_pairs: int = 7,
    ratio_identical: float = 0.20,
    ratio_partial: float = 0.35,
    intermediate_pool_fraction: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic OmniPath 3-hop dataset from INDRA results.

    Parameters
    ----------
    indra_csv :
        Path to an INDRA 3-hop CSV with ``source``, ``intermediate_1``,
        ``intermediate_2``, ``target``, ``logfoldchange``, ``pval`` columns.
    output_csv :
        Destination path for the synthetic dataset.
    coverage_fraction :
        Fraction of INDRA rows to include.
    top_n_pairs :
        Number of top source-target pairs by significance.
    n_selected_pairs :
        How many of the top pairs appear in the output.
    ratio_identical :
        Fraction of selected-pair rows with identical intermediates.
    ratio_partial :
        Fraction with one intermediate swapped.
    intermediate_pool_fraction :
        Fraction of INDRA intermediates available for swapping.
    seed :
        Random seed for reproducibility.

    Returns
    -------
    :
        The synthetic DataFrame (also written to *output_csv*).
    """
    indra_df = pd.read_csv(indra_csv)
    indra_clean = indra_df[
        np.isfinite(indra_df["pval"]) & np.isfinite(indra_df["logfoldchange"])
    ].copy()
    logger.info("Loaded %d valid INDRA rows", len(indra_clean))

    top_pairs = (
        indra_clean
        .assign(abs_logfc=lambda x: np.abs(x["logfoldchange"]))
        .sort_values(["pval", "abs_logfc"], ascending=[True, False])
        .drop_duplicates(subset=["source", "target"])
        .head(top_n_pairs)
    )
    selected = top_pairs.sample(n=min(n_selected_pairs, len(top_pairs)), random_state=seed)
    selected_set = set(zip(selected["source"], selected["target"]))

    all_intermediates = list(
        set(indra_clean["intermediate_1"].dropna().unique())
        | set(indra_clean["intermediate_2"].dropna().unique())
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(all_intermediates)
    pool_size = max(1, int(len(all_intermediates) * intermediate_pool_fraction))
    allowed = all_intermediates[:pool_size]

    op_rows: list[dict] = []

    for _, row in indra_clean.iterrows():
        if (row["source"], row["target"]) not in selected_set:
            continue
        rand = rng.random()
        if rand < ratio_identical:
            int1, int2, ptype = row["intermediate_1"], row["intermediate_2"], "identical"
        elif rand < ratio_identical + ratio_partial:
            if rng.random() < 0.5:
                int1, int2 = row["intermediate_1"], rng.choice(allowed)
            else:
                int1, int2 = rng.choice(allowed), row["intermediate_2"]
            ptype = "partial"
        else:
            int1, int2, ptype = rng.choice(allowed), rng.choice(allowed), "different"
        op_rows.append({
            "source": row["source"],
            "intermediate_1": int1,
            "intermediate_2": int2,
            "target": row["target"],
            "logfoldchange": row["logfoldchange"],
            "pval": row["pval"],
            "pathway_type": ptype,
        })

    target_size = int(len(indra_clean) * coverage_fraction)
    remaining = max(0, target_size - len(op_rows))
    if remaining > 0:
        extra = indra_clean.sample(n=min(remaining, len(indra_clean)), random_state=seed + 1)
        for _, row in extra.iterrows():
            op_rows.append({
                "source": row["source"],
                "intermediate_1": rng.choice(allowed),
                "intermediate_2": rng.choice(allowed),
                "target": row["target"],
                "logfoldchange": row["logfoldchange"],
                "pval": row["pval"],
                "pathway_type": "background",
            })

    out_df = pd.DataFrame(op_rows)
    _ensure_parent_dir(output_csv)
    out_df.to_csv(output_csv, index=False)

    logger.info(
        "Synthetic OmniPath dataset: %d rows (%.1f%% of INDRA) -> %s",
        len(out_df), 100 * len(out_df) / max(len(indra_clean), 1), output_csv,
    )
    logger.info("Pathway-type distribution:\n%s", out_df["pathway_type"].value_counts().to_string())
    return out_df


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _add_1hop_parser(subparsers) -> None:
    sp = subparsers.add_parser(
        "1hop", help="OmniPath 1-hop coverage analysis",
    )
    sp.add_argument("--source-genes-csv", required=True, help="target_validation_expanded.csv")
    sp.add_argument("--deg-dir", required=True, help="Folder with <GENE>_vs_control.csv files")
    sp.add_argument("--output", required=True, help="Output CSV for coverage results")
    sp.add_argument("--cache-parquet", default=None, help="Cache file for mapped OmniPath edges")
    sp.add_argument("--force-reload", action="store_true")
    sp.add_argument("--p-threshold", type=float, default=0.05)
    sp.add_argument("--filter-column", default="analysis_flag")
    sp.add_argument("--filter-value", default="Use_for_analysis")
    sp.add_argument("--gene-column", default="Gene")
    sp.add_argument("--genes", nargs="+",
                    help="Explicit source genes (overrides CSV)")
    sp.add_argument("--limit-genes", type=int, default=0)


def _add_synth_parser(subparsers) -> None:
    sp = subparsers.add_parser(
        "synthetic-3hop",
        help="Create synthetic OmniPath 3-hop dataset from INDRA results",
    )
    sp.add_argument("--indra-csv", required=True,
                    help="INDRA 3-hop result CSV")
    sp.add_argument("--output", required=True,
                    help="Output synthetic CSV")
    sp.add_argument("--coverage", type=float, default=0.18)
    sp.add_argument("--top-n-pairs", type=int, default=20)
    sp.add_argument("--n-selected-pairs", type=int, default=7)
    sp.add_argument("--ratio-identical", type=float, default=0.20)
    sp.add_argument("--ratio-partial", type=float, default=0.35)
    sp.add_argument("--intermediate-pool-fraction", type=float, default=0.25)
    sp.add_argument("--seed", type=int, default=42)


def _run_1hop_command(args: argparse.Namespace) -> None:
    genes = load_source_genes(
        args.source_genes_csv,
        gene_column=args.gene_column,
        filter_column=args.filter_column,
        filter_value=args.filter_value,
        explicit_genes=args.genes,
        limit=args.limit_genes,
    )
    df = run_1hop_omnipath(
        genes, args.deg_dir,
        cache_path=args.cache_parquet,
        force_reload=args.force_reload,
        pval_threshold=args.p_threshold,
    )
    _ensure_parent_dir(args.output)
    df.to_csv(args.output, index=False)
    logger.info("Results: %d rows -> %s", len(df), args.output)


def _run_synth_command(args: argparse.Namespace) -> None:
    create_synthetic_3hop(
        args.indra_csv, args.output,
        coverage_fraction=args.coverage,
        top_n_pairs=args.top_n_pairs,
        n_selected_pairs=args.n_selected_pairs,
        ratio_identical=args.ratio_identical,
        ratio_partial=args.ratio_partial,
        intermediate_pool_fraction=args.intermediate_pool_fraction,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for OmniPath pipelines."""
    ap = argparse.ArgumentParser(
        description="OmniPath 1-hop coverage and synthetic dataset creation.",
    )
    add_log_level_arg(ap, default="INFO")
    subparsers = ap.add_subparsers(dest="command", required=True)
    _add_1hop_parser(subparsers)
    _add_synth_parser(subparsers)
    args = ap.parse_args(argv)
    configure_logging(args.log_level)
    {"1hop": _run_1hop_command, "synthetic-3hop": _run_synth_command}[args.command](args)


if __name__ == "__main__":
    main()
