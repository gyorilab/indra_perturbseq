"""2-hop pathway extraction from an INDRA network export graph.
Pipeline:.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from indra_perturbseq.deg import load_deg_targets
from indra_perturbseq.evidence import enrich_evidence_2hop
from indra_perturbseq.gene_lists import load_gene_set
from indra_perturbseq.graph import is_hgnc_node, load_graph
from indra_perturbseq.hgnc import normalize_hgnc_symbol
from indra_perturbseq.mesh import annotate_mesh
from indra_perturbseq.pipelines.common import (
    load_sources_from_args,
    warn_deprecated_flags,
    write_split_outputs,
)
from indra_perturbseq.statements import best_statement, indra_html_url

logger = logging.getLogger(__name__)


def run_2hop_for_gene(
    graph,
    gene: str,
    deg_dir: str,
    p_threshold: float,
    prefer_fdr: bool,
    allowed_intermediates: set[str],
    limit_targets: int = 0,
) -> tuple[list[dict], str]:
    """Find all 2-hop paths for a single source gene."""
    targets, deg_map, err = load_deg_targets(
        deg_dir, gene, p_threshold, prefer_fdr,
    )
    if err:
        return [], f"SKIP {gene}: {err}"

    src = normalize_hgnc_symbol(gene)
    if not src or src not in graph or not is_hgnc_node(graph, src):
        return [], f"SKIP {gene}: source not in graph as HGNC"

    if limit_targets > 0:
        targets = targets[:limit_targets]

    target_set = {t for t in targets if t in graph and is_hgnc_node(graph, t)}
    rows: list[dict] = []

    for mid in graph.successors(src):
        if not is_hgnc_node(graph, mid) or mid not in allowed_intermediates:
            continue
        hop1 = best_statement(graph.get_edge_data(src, mid), require_incdec=False)
        if not hop1:
            continue

        for tgt in set(graph.successors(mid)) & target_set:
            hop2 = best_statement(graph.get_edge_data(mid, tgt), require_incdec=True)
            if not hop2:
                continue
            stats = deg_map.get(tgt, {})
            h1, h2 = hop1.get("stmt_hash"), hop2.get("stmt_hash")
            rows.append({
                "source": src,
                "intermediate": mid,
                "target": tgt,
                "stmt_type_1": hop1.get("stmt_type"),
                "stmt_type_2": hop2.get("stmt_type"),
                "belief_1": hop1.get("belief"),
                "belief_2": hop2.get("belief"),
                "evidence_1": hop1.get("evidence_count"),
                "evidence_2": hop2.get("evidence_count"),
                "logfoldchange": stats.get("logfoldchange"),
                "pval": stats.get("pval"),
                "hop1_hash": h1,
                "hop1_indra_url": indra_html_url(h1),
                "hop2_hash": h2,
                "hop2_indra_url": indra_html_url(h2),
            })

    return rows, f"{gene}: produced {len(rows)} rows"


def main(argv: list[str] | None = None) -> None:
    warn_deprecated_flags(
        argv,
        {
            "--genes-csv": "--source-genes-csv",
            "--out-csv-main": "--output-main",
            "--out-csv-self": "--output-self-targets",
            "--out-csv-self-targets": "--output-self-targets",
        },
        logger,
    )
    ap = argparse.ArgumentParser(
        description="2-hop pipeline using INDRA network export.",
    )
    ap.add_argument("--graph-pkl", required=True)
    ap.add_argument("--source-genes-csv", "--genes-csv", required=True,
                    help="target_validation_expanded.csv")
    ap.add_argument("--deg-dir", required=True)
    ap.add_argument("--endothelial-list", required=True,
                    help="CSV with 'gene' column for allowed intermediates")
    ap.add_argument("--mesh-reference", required=True,
                    help="MeSH reference CSV for term filtering")
    ap.add_argument("--output-main", "--out-csv-main", required=True)
    ap.add_argument(
        "--output-self-targets",
        "--out-csv-self",
        "--out-csv-self-targets",
        required=True,
    )

    ap.add_argument("--filter-column", default="analysis_flag")
    ap.add_argument("--filter-value", default="Use_for_analysis")
    ap.add_argument("--gene-column", default="Gene")
    ap.add_argument("--p-threshold", type=float, default=0.05)
    ap.add_argument("--prefer-fdr", action="store_true")
    ap.add_argument("--genes", nargs="+",
                    help="Explicit source genes (overrides CSV)")
    ap.add_argument("--limit-genes", type=int, default=0)
    ap.add_argument("--limit-targets", type=int, default=0)
    ap.add_argument("--path-workers", type=int, default=4)
    ap.add_argument("--evidence-workers", type=int, default=8)
    ap.add_argument("--mesh-batch-size", type=int, default=200)
    args = ap.parse_args(argv)

    graph, _ = load_graph(args.graph_pkl)
    allowed = load_gene_set(args.endothelial_list)

    genes = load_sources_from_args(args)

    all_rows: list[dict] = []
    logger.info("Running 2-hop extraction (parallel over genes)...")
    with ThreadPoolExecutor(max_workers=args.path_workers) as ex:
        futs = {
            ex.submit(
                run_2hop_for_gene, graph, g, args.deg_dir,
                args.p_threshold, args.prefer_fdr, allowed,
                args.limit_targets,
            ): g
            for g in genes
        }
        for fut in as_completed(futs):
            rows, msg = fut.result()
            logger.info(msg)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        logger.warning("No rows produced. Exiting.")
        return

    logger.info("Extraction complete: %d rows", len(df))
    logger.info("Enriching evidence + PMIDs...")
    df = enrich_evidence_2hop(df, max_workers=args.evidence_workers)
    logger.info("Annotating MeSH terms...")
    df = annotate_mesh(df, args.mesh_reference, mesh_batch_size=args.mesh_batch_size)

    write_split_outputs(df, args.output_main, args.output_self_targets, logger)


if __name__ == "__main__":
    main()
