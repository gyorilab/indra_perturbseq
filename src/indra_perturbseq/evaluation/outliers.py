"""Outlier / hub gene evaluation.

Two entry points:
- ``recompute``: TPR/FPR with union of path CSVs (including outlier paths).
- ``run_outliers``: 1/2-hop pathfinding using DEG-based target universe
  (no p-value thresholding) for hub genes like TP53 / CDKN1A.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from indra_perturbseq.deg import build_pvalue_map, deg_path_for_source, pick_sig_column
from indra_perturbseq.gene_lists import load_gene_set, load_karen_sources
from indra_perturbseq.graph import is_hgnc_node, load_graph
from indra_perturbseq.hgnc import normalize_hgnc_symbol
from indra_perturbseq.statements import best_statement, iter_incdec_statements

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001]


# ------------------------------------------------------------------
# recompute TPR/FPR with outlier paths included
# ------------------------------------------------------------------

def _load_explained_pairs_union(
    paths: list[str],
    source_col: str,
    target_col: str,
) -> set[tuple[str, str]]:
    """Load and union (source, target) pairs from multiple path CSVs."""
    explained: set[tuple[str, str]] = set()
    for p in paths:
        df = pd.read_csv(p, low_memory=False)
        if not {source_col, target_col}.issubset(df.columns):
            raise ValueError(f"{p} missing {source_col}/{target_col}")
        s = df[source_col].astype(str).str.strip().map(normalize_hgnc_symbol)
        t = df[target_col].astype(str).str.strip().map(normalize_hgnc_symbol)
        dd = pd.DataFrame({"source": s, "target": t}).dropna()
        dd = dd[dd["source"] != dd["target"]]
        explained |= set(zip(dd["source"], dd["target"]))
    return explained


def recompute(argv: list[str] | None = None) -> None:
    """CLI: recompute TPR/FPR across thresholds with outlier paths."""
    ap = argparse.ArgumentParser(
        description="TPR/FPR over thresholds with union of path CSVs.",
    )
    ap.add_argument("--tv-path", required=True)
    ap.add_argument("--tv-source-col", default="Gene")
    ap.add_argument("--karen-flag-col", default="Karen_Flag")
    ap.add_argument("--karen-flag-value", default="Use_for_analysis")
    ap.add_argument("--de-dir", required=True)
    ap.add_argument("--paths", nargs="+", required=True,
                    help="Path CSVs to union (TP / FP / outlier)")
    ap.add_argument("--path-source-col", default="source")
    ap.add_argument("--path-target-col", default="target")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=DEFAULT_THRESHOLDS)
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--print-table", action="store_true")
    args = ap.parse_args(argv)

    sources = load_karen_sources(
        args.tv_path, args.tv_source_col,
        args.karen_flag_col, args.karen_flag_value,
    )
    pairs = _load_explained_pairs_union(
        args.paths, args.path_source_col, args.path_target_col,
    )
    logger.info("Sources: %d | Explained pairs: %d", len(sources), len(pairs))

    expl_by_src: dict[str, set[str]] = defaultdict(set)
    for s, t in pairs:
        expl_by_src[s].add(t)

    cache: dict[str, tuple[np.ndarray, dict]] = {}
    for src in sources:
        f = deg_path_for_source(args.de_dir, src)
        if not os.path.exists(f):
            continue
        pmap = build_pvalue_map(f, src)
        if pmap:
            cache[src] = (np.fromiter(pmap.values(), dtype=float), pmap)

    rows = []
    for thr in args.thresholds:
        tp_all = fp_all = pos_all = neg_all = 0
        for src, (pvals, pmap) in cache.items():
            pos_all += int(np.sum(pvals <= thr))
            neg_all += int(np.sum(pvals > thr))
            for t in expl_by_src.get(src, set()):
                p = pmap.get(t)
                if p is None:
                    continue
                if p <= thr:
                    tp_all += 1
                else:
                    fp_all += 1

        tpr = tp_all / pos_all if pos_all else float("nan")
        fpr = fp_all / neg_all if neg_all else float("nan")
        rows.append({
            "threshold": float(thr),
            "TP_total": tp_all, "FP_total": fp_all,
            "positives_total": pos_all, "negatives_total": neg_all,
            "TPR_overall": tpr, "FPR_overall": fpr,
        })
        logger.info("thr %g | TPR=%.4f FPR=%.4f", thr, tpr, fpr)

    out = pd.DataFrame(rows)
    if args.out_csv:
        out.to_csv(args.out_csv, index=False)
        logger.info("Wrote %s", args.out_csv)
    if args.print_table:
        print(out.to_string(index=False))


# ------------------------------------------------------------------
# run_outliers: pathfinding using DEG universe (no p-value threshold)
# ------------------------------------------------------------------

def _load_deg_universe(deg_csv: str) -> tuple[set[str], dict]:
    """Load all targets from a DEG CSV (no filtering by significance)."""
    df = pd.read_csv(deg_csv, low_memory=False)
    if "names" not in df.columns:
        raise ValueError(f"{deg_csv} missing 'names' column")

    pcol = next(
        (c for c in ("pvals", "pval", "p_value", "p_val",
                      "pvals_adj", "padj", "qval", "fdr", "p_adj")
         if c in df.columns),
        None,
    )
    if pcol:
        df[pcol] = pd.to_numeric(df[pcol], errors="coerce")
    if "logfoldchanges" in df.columns:
        df["logfoldchanges"] = pd.to_numeric(df["logfoldchanges"], errors="coerce")
    else:
        df["logfoldchanges"] = pd.NA

    universe: set[str] = set()
    stats: dict[str, dict] = {}
    for _, r in df.iterrows():
        t = normalize_hgnc_symbol(r["names"])
        if not t:
            continue
        universe.add(t)
        p = r[pcol] if pcol else None
        lfc = r.get("logfoldchanges")
        if t not in stats or (pcol and pd.notna(p) and
                               (stats[t]["pval"] is None or p < stats[t]["pval"])):
            stats[t] = {"logfoldchange": lfc, "pval": p}
    return universe, stats


def run_outliers(argv: list[str] | None = None) -> None:
    """CLI: 1/2-hop pathfinding for outlier genes on endothelial universe."""
    ap = argparse.ArgumentParser(
        description="Outlier gene pathfinding (DEG universe, no p-val threshold).",
    )
    ap.add_argument("--graph-pkl", required=True)
    ap.add_argument("--endothelial-list", required=True)
    ap.add_argument("--endothelial-gene-col", default="gene")
    ap.add_argument("--sources", nargs="+", required=True,
                    help="e.g. TP53 CDKN1A")
    ap.add_argument("--deg-csvs", nargs="+", required=True,
                    help="Corresponding <source>_vs_control.csv files")
    ap.add_argument("--out-1hop-csv", required=True)
    ap.add_argument("--out-2hop-csv", required=True)
    args = ap.parse_args(argv)

    if len(args.sources) != len(args.deg_csvs):
        raise SystemExit("--sources and --deg-csvs must have same length")

    graph, _ = load_graph(args.graph_pkl)
    endo = load_gene_set(args.endothelial_list, gene_col=args.endothelial_gene_col)
    endo = {g for g in endo if g in graph and is_hgnc_node(graph, g)}
    logger.info("Endothelial in-graph: %d", len(endo))

    rows_1, rows_2 = [], []
    for raw_src, deg_csv in zip(args.sources, args.deg_csvs):
        src = normalize_hgnc_symbol(raw_src)
        if not src or src not in graph or not is_hgnc_node(graph, src):
            logger.warning("SKIP %s: not in graph", raw_src)
            continue

        universe, stats = _load_deg_universe(deg_csv)
        targets = (universe & endo) - {src}
        targets = {t for t in targets if t in graph and is_hgnc_node(graph, t)}
        logger.info("%s: %d endothelial targets in DEG universe", src, len(targets))

        for tgt in sorted(targets):
            if not graph.has_edge(src, tgt):
                continue
            for s in iter_incdec_statements(graph, src, tgt):
                st = stats.get(tgt, {})
                rows_1.append({
                    "source": src, "target": tgt,
                    "stmt_type": s.get("stmt_type"),
                    "belief": s.get("belief"),
                    "evidence_count": s.get("evidence_count"),
                    "stmt_hash": s.get("stmt_hash"),
                    "logfoldchange": st.get("logfoldchange"),
                    "pval": st.get("pval"),
                })

        for mid in graph.successors(src):
            if mid not in endo:
                continue
            hop1 = best_statement(graph.get_edge_data(src, mid), require_incdec=False)
            if not hop1:
                continue
            for tgt in set(graph.successors(mid)) & targets:
                hop2 = best_statement(graph.get_edge_data(mid, tgt), require_incdec=True)
                if not hop2:
                    continue
                st = stats.get(tgt, {})
                rows_2.append({
                    "source": src, "intermediate": mid, "target": tgt,
                    "stmt_type_1": hop1.get("stmt_type"),
                    "stmt_type_2": hop2.get("stmt_type"),
                    "belief_1": hop1.get("belief"),
                    "belief_2": hop2.get("belief"),
                    "evidence_1": hop1.get("evidence_count"),
                    "evidence_2": hop2.get("evidence_count"),
                    "hop1_hash": hop1.get("stmt_hash"),
                    "hop2_hash": hop2.get("stmt_hash"),
                    "logfoldchange": st.get("logfoldchange"),
                    "pval": st.get("pval"),
                })

    pd.DataFrame(rows_1).to_csv(args.out_1hop_csv, index=False)
    pd.DataFrame(rows_2).to_csv(args.out_2hop_csv, index=False)
    logger.info("1-hop: %d -> %s", len(rows_1), args.out_1hop_csv)
    logger.info("2-hop: %d -> %s", len(rows_2), args.out_2hop_csv)


def main(argv: list[str] | None = None) -> None:
    """Dispatch to ``run_outliers`` (default CLI entry point)."""
    run_outliers(argv)


if __name__ == "__main__":
    main()
