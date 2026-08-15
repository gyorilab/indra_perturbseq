"""Permuted-network pathfinding for null-model TPR comparison.

Produces 1-hop and 2-hop path CSVs on a label-permuted INDRA graph.
Topology is unchanged; only HGNC node labels are shuffled.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from indra_perturbseq.deg import load_deg_targets
from indra_perturbseq.gene_lists import load_gene_set, load_source_genes
from indra_perturbseq.graph import is_hgnc_node, load_graph
from indra_perturbseq.hgnc import normalize_hgnc_symbol
from indra_perturbseq.permutation import PermutationView
from indra_perturbseq.statements import best_statement, iter_incdec_statements

logger = logging.getLogger(__name__)


def _run_1hop_permuted(graph, pv, src_label, targets, deg_map):
    rows = []
    src0 = pv.orig_for_label(src_label)
    if not src0 or src0 not in graph:
        return rows

    for tgt_label in targets:
        if tgt_label == src_label:
            continue
        tgt0 = pv.orig_for_label(tgt_label)
        if not tgt0 or tgt0 not in graph or not graph.has_edge(src0, tgt0):
            continue
        for s in iter_incdec_statements(graph, src0, tgt0):
            stats = deg_map.get(tgt_label, {})
            rows.append({
                "source": src_label,
                "target": tgt_label,
                "stmt_type": s.get("stmt_type"),
                "belief": s.get("belief"),
                "evidence_count": s.get("evidence_count"),
                "logfoldchange": stats.get("logfoldchange"),
                "pval": stats.get("pval"),
            })
    return rows


def _run_2hop_permuted(graph, pv, src_label, targets_set, deg_map,
                       allowed_labels):
    rows = []
    src0 = pv.orig_for_label(src_label)
    if not src0 or src0 not in graph:
        return rows

    for mid0 in graph.successors(src0):
        if not is_hgnc_node(graph, mid0):
            continue
        mid_label = pv.label_for_orig(mid0)
        if not mid_label:
            continue
        if allowed_labels is not None and mid_label not in allowed_labels:
            continue
        hop1 = best_statement(graph.get_edge_data(src0, mid0), require_incdec=False)
        if not hop1:
            continue

        for tgt0 in graph.successors(mid0):
            if not is_hgnc_node(graph, tgt0):
                continue
            tgt_label = pv.label_for_orig(tgt0)
            if not tgt_label or tgt_label not in targets_set or tgt_label == src_label:
                continue
            hop2 = best_statement(graph.get_edge_data(mid0, tgt0), require_incdec=True)
            if not hop2:
                continue
            stats = deg_map.get(tgt_label, {})
            rows.append({
                "source": src_label,
                "intermediate": mid_label,
                "target": tgt_label,
                "stmt_type_1": hop1.get("stmt_type"),
                "stmt_type_2": hop2.get("stmt_type"),
                "belief_1": hop1.get("belief"),
                "belief_2": hop2.get("belief"),
                "evidence_1": hop1.get("evidence_count"),
                "evidence_2": hop2.get("evidence_count"),
                "logfoldchange": stats.get("logfoldchange"),
                "pval": stats.get("pval"),
            })
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Permuted-network 1/2-hop path CSVs for TPR comparison.",
    )
    ap.add_argument("--graph-pkl", required=True)
    ap.add_argument("--source-genes-csv", required=True,
                    help="target_validation_expanded.csv")
    ap.add_argument("--deg-dir", required=True)
    ap.add_argument("--p-threshold", type=float, default=0.05)
    ap.add_argument("--prefer-fdr", action="store_true")
    ap.add_argument("--filter-column", default="analysis_flag")
    ap.add_argument("--filter-value", default="Use_for_analysis")
    ap.add_argument("--gene-column", default="Gene")
    ap.add_argument("--seed", type=int, default=42,
                    help="Permutation seed for HGNC label shuffling")
    ap.add_argument("--mode", choices=["1hop", "2hop", "both"], default="both")
    ap.add_argument("--allowed-intermediates-csv", default="")
    ap.add_argument("--allowed-intermediates-gene-col", default="gene")
    ap.add_argument("--output-1hop", default="permuted_1hop_paths.csv")
    ap.add_argument("--output-2hop", default="permuted_2hop_paths.csv")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    graph, _ = load_graph(args.graph_pkl)
    hgnc_nodes = [n for n in graph.nodes if is_hgnc_node(graph, n)]
    logger.info("HGNC nodes: %d", len(hgnc_nodes))

    pv = PermutationView(graph, hgnc_nodes, seed=args.seed)
    logger.info("Permutation seed: %d", args.seed)

    allowed = None
    if args.allowed_intermediates_csv:
        allowed = load_gene_set(
            args.allowed_intermediates_csv,
            gene_column=args.allowed_intermediates_gene_col,
        )

    sources_raw = load_source_genes(
        args.source_genes_csv,
        gene_column=args.gene_column,
        filter_column=args.filter_column,
        filter_value=args.filter_value,
    )

    jobs = []
    for raw_src in sources_raw:
        src = normalize_hgnc_symbol(raw_src)
        if not src or src not in graph or not is_hgnc_node(graph, src):
            continue
        targets, deg_map, err = load_deg_targets(
            args.deg_dir, raw_src, args.p_threshold, args.prefer_fdr,
        )
        if err:
            continue
        targets = [t for t in targets
                    if t in graph and is_hgnc_node(graph, t) and t != src]
        if not targets:
            continue
        deg_map = {t: deg_map.get(t, {}) for t in targets}
        jobs.append((src, targets, deg_map))

    logger.info("Jobs: %d sources", len(jobs))

    onehop_rows, twohop_rows = [], []

    def _job(job):
        src, tgts, dm = job
        o1 = _run_1hop_permuted(graph, pv, src, tgts, dm) if args.mode in ("1hop", "both") else []
        o2 = _run_2hop_permuted(graph, pv, src, set(tgts), dm, allowed) if args.mode in ("2hop", "both") else []
        return o1, o2

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            o1, o2 = fut.result()
            onehop_rows.extend(o1)
            twohop_rows.extend(o2)
            if i % 25 == 0 or i == len(futs):
                logger.info("Progress %d/%d | 1hop=%d 2hop=%d",
                            i, len(futs), len(onehop_rows), len(twohop_rows))

    if args.mode in ("1hop", "both"):
        df1 = pd.DataFrame(onehop_rows)
        if not df1.empty:
            df1 = df1[df1["source"] != df1["target"]].copy()
        df1.to_csv(args.output_1hop, index=False)
        logger.info("Wrote %d permuted 1-hop rows -> %s", len(df1), args.output_1hop)

    if args.mode in ("2hop", "both"):
        df2 = pd.DataFrame(twohop_rows)
        if not df2.empty:
            df2 = df2[df2["source"] != df2["target"]].copy()
        df2.to_csv(args.output_2hop, index=False)
        logger.info("Wrote %d permuted 2-hop rows -> %s", len(df2), args.output_2hop)


if __name__ == "__main__":
    main()
