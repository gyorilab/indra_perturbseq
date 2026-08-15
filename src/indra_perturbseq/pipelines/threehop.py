"""3-hop pathway extraction using INDRA network export + pathfinding.

Uses ``indra.explanation.pathfinding`` for shortest simple paths,
then enriches evidence via Neo4j and annotates MeSH terms.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx
import pandas as pd

from indra.explanation.pathfinding.pathfinding import shortest_simple_paths

from indra_perturbseq.deg import load_deg_targets
from indra_perturbseq.evidence import enrich_evidence_3hop
from indra_perturbseq.gene_lists import load_gene_set
from indra_perturbseq.graph import is_hgnc_node, load_graph
from indra_perturbseq.hgnc import normalize_hgnc_symbol
from indra_perturbseq.mesh import annotate_mesh
from indra_perturbseq.pipelines.common import (
    ensure_parent_dir,
    load_sources_from_args,
    warn_deprecated_flags,
    write_split_outputs,
)
from indra_perturbseq.statements import best_statement, indra_html_url

logger = logging.getLogger(__name__)


def run_3hop_for_gene(
    graph: nx.DiGraph,
    gene: str,
    deg_dir: str,
    p_threshold: float,
    prefer_fdr: bool,
    allowed_intermediates: set[str],
    limit_targets: int = 0,
    max_paths_per_pair: int = 1,
) -> tuple[list[dict], str]:
    """Find 3-hop paths: source -> mid1 -> mid2 -> target."""
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
    for tgt in target_set:
        try:
            path_gen = shortest_simple_paths(graph, src, tgt, weight=None)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        found = 0
        try:
            for path in path_gen:
                if len(path) < 4:
                    continue
                if len(path) > 4:
                    break

                _, mid1, mid2, _ = path
                if not (is_hgnc_node(graph, mid1) and is_hgnc_node(graph, mid2)):
                    continue
                if mid1 not in allowed_intermediates or mid2 not in allowed_intermediates:
                    continue

                h1s = best_statement(graph.get_edge_data(src, mid1), require_incdec=False)
                h2s = best_statement(graph.get_edge_data(mid1, mid2), require_incdec=False)
                h3s = best_statement(graph.get_edge_data(mid2, tgt), require_incdec=True)
                if not (h1s and h2s and h3s):
                    continue

                h1, h2, h3 = h1s.get("stmt_hash"), h2s.get("stmt_hash"), h3s.get("stmt_hash")
                stats = deg_map.get(tgt, {})
                rows.append({
                    "source": src,
                    "intermediate_1": mid1,
                    "intermediate_2": mid2,
                    "target": tgt,
                    "stmt_type_1": h1s.get("stmt_type"),
                    "stmt_type_2": h2s.get("stmt_type"),
                    "stmt_type_3": h3s.get("stmt_type"),
                    "belief_1": h1s.get("belief"),
                    "belief_2": h2s.get("belief"),
                    "belief_3": h3s.get("belief"),
                    "evidence_1": h1s.get("evidence_count"),
                    "evidence_2": h2s.get("evidence_count"),
                    "evidence_3": h3s.get("evidence_count"),
                    "logfoldchange": stats.get("logfoldchange"),
                    "pval": stats.get("pval"),
                    "hop1_hash": h1, "hop2_hash": h2, "hop3_hash": h3,
                    "hop1_indra_url": indra_html_url(h1),
                    "hop2_indra_url": indra_html_url(h2),
                    "hop3_indra_url": indra_html_url(h3),
                })
                found += 1
                if max_paths_per_pair > 0 and found >= max_paths_per_pair:
                    break
        except nx.NetworkXNoPath:
            pass

    return rows, f"{gene}: produced {len(rows)} rows"


_COLUMN_ORDER = [
    "source", "intermediate_1", "intermediate_2", "target",
    "stmt_type_1", "stmt_type_2", "stmt_type_3",
    "belief_1", "belief_2", "belief_3",
    "evidence_1", "evidence_2", "evidence_3",
    "logfoldchange", "pval",
    "evidence_text_hop1", "pmids_hop1", "Annotated MeSH terms hop1",
    "evidence_text_hop2", "pmids_hop2", "Annotated MeSH terms hop2",
    "evidence_text_hop3", "pmids_hop3", "Annotated MeSH terms hop3",
    "hop1_hash", "hop2_hash", "hop3_hash",
    "hop1_indra_url", "hop2_indra_url", "hop3_indra_url",
]


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in _COLUMN_ORDER if c in df.columns]
    extra = [c for c in df.columns if c not in keep]
    return df[keep + extra]


def main(argv: list[str] | None = None) -> None:
    warn_deprecated_flags(
        argv,
        {
            "--genes-csv": "--source-genes-csv",
            "--out-csv-raw": "--output-raw",
            "--out-csv-main": "--output-main",
            "--out-csv-self": "--output-self-targets",
            "--out-csv-self-targets": "--output-self-targets",
        },
        logger,
    )
    ap = argparse.ArgumentParser(
        description="3-hop pipeline using INDRA network export + Neo4j evidence.",
    )
    ap.add_argument("--graph-pkl", required=True)
    ap.add_argument("--source-genes-csv", "--genes-csv", required=True,
                    help="target_validation_expanded.csv")
    ap.add_argument("--deg-dir", required=True)
    ap.add_argument("--endothelial-list", required=True,
                    help="CSV with 'gene' column")
    ap.add_argument("--mesh-reference", required=True)

    ap.add_argument("--output-raw", "--out-csv-raw", required=True,
                    help="Raw 3-hop rows (pre-enrichment)")
    ap.add_argument("--output-main", "--out-csv-main", required=True,
                    help="Final enriched non-self rows")
    ap.add_argument(
        "--output-self-targets",
        "--out-csv-self",
        "--out-csv-self-targets",
        required=True,
        help="Final enriched self rows",
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
    ap.add_argument("--max-paths-per-pair", type=int, default=1)
    ap.add_argument("--path-workers", type=int, default=4)
    ap.add_argument("--neo4j-evidence-batch-size", type=int, default=2000)
    ap.add_argument("--mesh-batch-size", type=int, default=200)
    args = ap.parse_args(argv)

    graph, _ = load_graph(args.graph_pkl)
    allowed = load_gene_set(args.endothelial_list)

    genes = load_sources_from_args(args)

    all_rows: list[dict] = []
    logger.info("Running 3-hop extraction...")
    with ThreadPoolExecutor(max_workers=args.path_workers) as ex:
        futs = {
            ex.submit(
                run_3hop_for_gene, graph, g, args.deg_dir,
                args.p_threshold, args.prefer_fdr, allowed,
                args.limit_targets, args.max_paths_per_pair,
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

    ensure_parent_dir(args.output_raw)
    _reorder_columns(df.copy()).to_csv(args.output_raw, index=False)
    logger.info("Raw 3-hop rows saved: %s", args.output_raw)

    logger.info("Enriching evidence + PMIDs (Neo4j)...")
    df = enrich_evidence_3hop(df, neo4j_batch_size=args.neo4j_evidence_batch_size)
    logger.info("Annotating MeSH terms...")
    df = annotate_mesh(df, args.mesh_reference, mesh_batch_size=args.mesh_batch_size)

    df = _reorder_columns(df)
    write_split_outputs(df, args.output_main, args.output_self_targets, logger)


if __name__ == "__main__":
    main()
