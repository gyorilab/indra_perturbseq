"""Export false-positive paths using DEG-derived negative target sets.
Runs hop-based pathfinding and records supporting edge statement metadata."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from indra_perturbseq.deg import load_deg_universe
from indra_perturbseq.gene_lists import load_gene_set, load_source_genes
from indra_perturbseq.graph import is_hgnc_node, load_graph
from indra_perturbseq.hgnc import normalize_hgnc_symbol
from indra_perturbseq.permutation import PermutationView
from indra_perturbseq.statements import best_statement, iter_incdec_statements

logger = logging.getLogger(__name__)


def _run_1hop_fp(graph, src, neg_targets, stats_map, pv):
    rows = []
    src0 = src if pv is None else pv.orig_for_label(src)
    if not src0 or src0 not in graph:
        return rows
    for tgt in neg_targets:
        if tgt == src:
            continue
        tgt0 = tgt if pv is None else pv.orig_for_label(tgt)
        if not tgt0 or tgt0 not in graph or not graph.has_edge(src0, tgt0):
            continue
        for s in iter_incdec_statements(graph, src0, tgt0):
            st = stats_map.get(tgt, {})
            rows.append({
                "source": src, "target": tgt,
                "stmt_type": s.get("stmt_type"),
                "belief": s.get("belief"),
                "evidence_count": s.get("evidence_count"),
                "logfoldchange": st.get("logfoldchange"),
                "pval": st.get("pval"),
            })
    return rows


def _run_2hop_fp(graph, src, neg_set, allowed, stats_map, pv):
    rows = []
    src0 = src if pv is None else pv.orig_for_label(src)
    if not src0 or src0 not in graph:
        return rows
    for mid0 in graph.successors(src0):
        if not is_hgnc_node(graph, mid0):
            continue
        mid = mid0 if pv is None else pv.label_for_orig(mid0)
        if not mid:
            continue
        if allowed is not None and mid not in allowed:
            continue
        hop1 = best_statement(graph.get_edge_data(src0, mid0), require_incdec=False)
        if not hop1:
            continue
        for tgt0 in graph.successors(mid0):
            if not is_hgnc_node(graph, tgt0):
                continue
            tgt = tgt0 if pv is None else pv.label_for_orig(tgt0)
            if not tgt or tgt not in neg_set or tgt == src:
                continue
            hop2 = best_statement(graph.get_edge_data(mid0, tgt0), require_incdec=True)
            if not hop2:
                continue
            st = stats_map.get(tgt, {})
            rows.append({
                "source": src, "intermediate": mid, "target": tgt,
                "stmt_type_1": hop1.get("stmt_type"),
                "stmt_type_2": hop2.get("stmt_type"),
                "belief_1": hop1.get("belief"),
                "belief_2": hop2.get("belief"),
                "evidence_1": hop1.get("evidence_count"),
                "evidence_2": hop2.get("evidence_count"),
                "logfoldchange": st.get("logfoldchange"),
                "pval": st.get("pval"),
            })
    return rows


def _run_3hop_fp(graph, src, neg_set, allowed, stats_map, pv):
    rows = []
    src0 = src if pv is None else pv.orig_for_label(src)
    if not src0 or src0 not in graph:
        return rows
    for m10 in graph.successors(src0):
        if not is_hgnc_node(graph, m10):
            continue
        m1 = m10 if pv is None else pv.label_for_orig(m10)
        if not m1 or (allowed is not None and m1 not in allowed):
            continue
        h1 = best_statement(graph.get_edge_data(src0, m10), require_incdec=False)
        if not h1:
            continue
        for m20 in graph.successors(m10):
            if not is_hgnc_node(graph, m20):
                continue
            m2 = m20 if pv is None else pv.label_for_orig(m20)
            if not m2 or (allowed is not None and m2 not in allowed):
                continue
            h2 = best_statement(graph.get_edge_data(m10, m20), require_incdec=False)
            if not h2:
                continue
            for tgt0 in graph.successors(m20):
                if not is_hgnc_node(graph, tgt0):
                    continue
                tgt = tgt0 if pv is None else pv.label_for_orig(tgt0)
                if not tgt or tgt not in neg_set or tgt == src:
                    continue
                h3 = best_statement(graph.get_edge_data(m20, tgt0), require_incdec=True)
                if not h3:
                    continue
                st = stats_map.get(tgt, {})
                rows.append({
                    "source": src, "intermediate_1": m1,
                    "intermediate_2": m2, "target": tgt,
                    "stmt_type_1": h1.get("stmt_type"),
                    "stmt_type_2": h2.get("stmt_type"),
                    "stmt_type_3": h3.get("stmt_type"),
                    "belief_1": h1.get("belief"),
                    "belief_2": h2.get("belief"),
                    "belief_3": h3.get("belief"),
                    "evidence_1": h1.get("evidence_count"),
                    "evidence_2": h2.get("evidence_count"),
                    "evidence_3": h3.get("evidence_count"),
                    "logfoldchange": st.get("logfoldchange"),
                    "pval": st.get("pval"),
                })
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Export FP paths using per-source DEG universes.",
    )
    ap.add_argument("--graph-pkl", required=True)
    ap.add_argument("--source-genes-csv", required=True)
    ap.add_argument("--deg-dir", required=True)
    ap.add_argument("--allowed-intermediates-csv", default="")
    ap.add_argument("--allowed-intermediates-gene-col", default="gene")
    ap.add_argument("--p-threshold", type=float, default=0.05)
    ap.add_argument("--prefer-fdr", action="store_true")
    ap.add_argument("--filter-column", default="analysis_flag")
    ap.add_argument("--filter-value", default="Use_for_analysis")
    ap.add_argument("--gene-column", default="Gene")
    ap.add_argument("--mode", choices=["1hop", "2hop", "3hop", "both", "all"],
                    default="both")
    ap.add_argument("--limit-negatives-per-source", type=int, default=0)
    ap.add_argument("--sample-seed", type=int, default=1)
    ap.add_argument("--permute-seed", type=int, default=0,
                    help="If >0, evaluate on permuted network")
    ap.add_argument("--output-1hop", default="")
    ap.add_argument("--output-2hop", default="")
    ap.add_argument("--output-3hop", default="")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    graph, _ = load_graph(args.graph_pkl)
    hgnc_nodes = [n for n in graph.nodes if is_hgnc_node(graph, n)]

    allowed = None
    if args.allowed_intermediates_csv:
        allowed = load_gene_set(
            args.allowed_intermediates_csv,
            gene_column=args.allowed_intermediates_gene_col,
        )

    pv = None
    if args.permute_seed > 0:
        pv = PermutationView(graph, hgnc_nodes, seed=args.permute_seed)
        logger.info("Evaluating on PERMUTED labels (seed=%d)", args.permute_seed)

    sources_raw = load_source_genes(
        args.source_genes_csv,
        gene_column=args.gene_column,
        filter_column=args.filter_column,
        filter_value=args.filter_value,
    )

    rng = np.random.default_rng(args.sample_seed)
    jobs = []
    for raw_src in sources_raw:
        src = normalize_hgnc_symbol(raw_src)
        if not src or src not in graph or not is_hgnc_node(graph, src):
            continue
        all_tgts, pos_tgts, stats_map, err = load_deg_universe(
            args.deg_dir, raw_src, args.p_threshold, args.prefer_fdr,
        )
        if err:
            continue
        all_tgts = {t for t in all_tgts if t in graph and is_hgnc_node(graph, t)}
        pos_tgts = {t for t in pos_tgts if t in all_tgts}
        neg = sorted((all_tgts - pos_tgts) - {src})
        if not neg:
            continue
        lim = args.limit_negatives_per_source
        if lim > 0 and len(neg) > lim:
            neg = rng.choice(np.array(neg, dtype=object), size=lim, replace=False).tolist()
        jobs.append((src, neg, stats_map))

    logger.info("Jobs: %d sources", len(jobs))

    rows_1, rows_2, rows_3 = [], [], []

    def _job(j):
        src, neg, sm = j
        o1 = _run_1hop_fp(graph, src, neg, sm, pv) if args.mode in ("1hop", "both", "all") else []
        o2 = _run_2hop_fp(graph, src, set(neg), allowed, sm, pv) if args.mode in ("2hop", "both", "all") else []
        o3 = _run_3hop_fp(graph, src, set(neg), allowed, sm, pv) if args.mode in ("3hop", "all") else []
        return o1, o2, o3

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            o1, o2, o3 = fut.result()
            rows_1.extend(o1)
            rows_2.extend(o2)
            rows_3.extend(o3)
            if i % 25 == 0 or i == len(futs):
                logger.info("Progress %d/%d | 1hop=%d 2hop=%d 3hop=%d",
                            i, len(futs), len(rows_1), len(rows_2), len(rows_3))

    def _write(rows, path, label):
        if not path:
            return
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df[df["source"] != df["target"]].copy()
        df.to_csv(path, index=False)
        logger.info("Wrote %s -> %s | rows=%d", label, path, len(df))

    if args.mode in ("1hop", "both", "all"):
        _write(rows_1, args.output_1hop, "1-hop FP")
    if args.mode in ("2hop", "both", "all"):
        _write(rows_2, args.output_2hop, "2-hop FP")
    if args.mode in ("3hop", "all"):
        _write(rows_3, args.output_3hop, "3-hop FP")


if __name__ == "__main__":
    main()
