"""Unified multi-hop path discovery CLI.

Discovers 1-hop, 2-hop, and/or 3-hop paths from source gene(s) to targets
within a gene whitelist, then annotates each target with DEG statistics.

No p-value threshold is applied during discovery -- DEG stats are annotations
only. Waterfall exclusion ensures targets explained at a lower hop are not
repeated at higher hops (disable with ``--no-waterfall``).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from indra_perturbseq.deg import load_deg_targets, pick_sig_column
from indra_perturbseq.gene_lists import load_gene_set
from indra_perturbseq.graph import is_hgnc_node, load_graph
from indra_perturbseq.hgnc import normalize_hgnc_symbol
from indra_perturbseq.statements import best_statement, indra_html_url

logger = logging.getLogger(__name__)

_COLUMN_ORDER = [
    "hop", "source", "intermediate_1", "intermediate_2", "target",
    "stmt_type_1", "stmt_type_2", "stmt_type_3",
    "belief_1", "belief_2", "belief_3",
    "evidence_count_1", "evidence_count_2", "evidence_count_3",
    "logfoldchange", "pval", "pvals_adj",
    "hop1_hash", "hop2_hash", "hop3_hash",
    "hop1_indra_url", "hop2_indra_url", "hop3_indra_url",
]


def _load_full_deg_map(deg_dir: str, raw_gene: str, prefer_fdr: bool) -> dict:
    """Load complete DEG map (all genes, no significance filter)."""
    path = os.path.join(deg_dir, f"{raw_gene}_vs_control.csv")
    if not os.path.exists(path):
        logger.warning("DEG file not found: %s", path)
        return {}
    df = pd.read_csv(path, low_memory=False)
    if "names" not in df.columns:
        logger.warning("DEG missing 'names': %s", path)
        return {}
    try:
        sig_col = pick_sig_column(df, prefer_fdr=prefer_fdr)
    except ValueError:
        sig_col = None

    import math
    for col in ("logfoldchanges", "pvals", "pvals_adj", sig_col):
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    deg_map: dict = {}
    for _, row in df.iterrows():
        tgt = normalize_hgnc_symbol(str(row["names"]))
        if not tgt:
            continue
        p = float(row.get(sig_col, float("nan"))) if sig_col else float("nan")
        padj = float(row.get("pvals_adj", float("nan")))
        lfc = float(row.get("logfoldchanges", float("nan")))
        if tgt not in deg_map or (not math.isnan(p) and p < deg_map[tgt]["pval"]):
            deg_map[tgt] = {"logfoldchange": lfc, "pval": p, "pvals_adj": padj}
    return deg_map


def _make_row(hop, src, tgt, deg_map, s1=None, s2=None, s3=None,
              mid1="", mid2=""):
    d = deg_map.get(tgt, {})
    return {
        "hop": hop, "source": src,
        "intermediate_1": mid1, "intermediate_2": mid2, "target": tgt,
        "stmt_type_1": s1.get("stmt_type") if s1 else pd.NA,
        "stmt_type_2": s2.get("stmt_type") if s2 else pd.NA,
        "stmt_type_3": s3.get("stmt_type") if s3 else pd.NA,
        "belief_1": s1.get("belief") if s1 else pd.NA,
        "belief_2": s2.get("belief") if s2 else pd.NA,
        "belief_3": s3.get("belief") if s3 else pd.NA,
        "evidence_count_1": s1.get("evidence_count") if s1 else pd.NA,
        "evidence_count_2": s2.get("evidence_count") if s2 else pd.NA,
        "evidence_count_3": s3.get("evidence_count") if s3 else pd.NA,
        "hop1_hash": s1.get("stmt_hash") if s1 else pd.NA,
        "hop2_hash": s2.get("stmt_hash") if s2 else pd.NA,
        "hop3_hash": s3.get("stmt_hash") if s3 else pd.NA,
        "hop1_indra_url": indra_html_url(s1.get("stmt_hash")) if s1 else "",
        "hop2_indra_url": indra_html_url(s2.get("stmt_hash")) if s2 else "",
        "hop3_indra_url": indra_html_url(s3.get("stmt_hash")) if s3 else "",
        "logfoldchange": d.get("logfoldchange", float("nan")),
        "pval": d.get("pval", float("nan")),
        "pvals_adj": d.get("pvals_adj", float("nan")),
    }


def _run_1hop(graph, src, whitelist, deg_map):
    rows, found = [], set()
    for tgt in graph.successors(src):
        if not is_hgnc_node(graph, tgt) or tgt not in whitelist or tgt == src:
            continue
        s1 = best_statement(graph.get_edge_data(src, tgt), require_incdec=True)
        if not s1:
            continue
        rows.append(_make_row(1, src, tgt, deg_map, s1=s1))
        found.add(tgt)
    return rows, found


def _run_2hop(graph, src, whitelist, deg_map, excluded):
    rows, found = [], set()
    for mid in graph.successors(src):
        if not is_hgnc_node(graph, mid) or mid not in whitelist or mid == src:
            continue
        s1 = best_statement(graph.get_edge_data(src, mid), require_incdec=False)
        if not s1:
            continue
        for tgt in graph.successors(mid):
            if not is_hgnc_node(graph, tgt) or tgt not in whitelist:
                continue
            if tgt in (src, mid) or tgt in excluded:
                continue
            s2 = best_statement(graph.get_edge_data(mid, tgt), require_incdec=True)
            if not s2:
                continue
            rows.append(_make_row(2, src, tgt, deg_map, s1=s1, s2=s2, mid1=mid))
            found.add(tgt)
    return rows, found


def _run_3hop(graph, src, whitelist, deg_map, excluded, max_per_pair):
    candidates: dict[str, list[tuple[float, dict]]] = {}
    for mid1 in graph.successors(src):
        if not is_hgnc_node(graph, mid1) or mid1 not in whitelist or mid1 == src:
            continue
        s1 = best_statement(graph.get_edge_data(src, mid1), require_incdec=False)
        if not s1:
            continue
        for mid2 in graph.successors(mid1):
            if not is_hgnc_node(graph, mid2) or mid2 not in whitelist:
                continue
            if mid2 in (src, mid1):
                continue
            s2 = best_statement(graph.get_edge_data(mid1, mid2), require_incdec=False)
            if not s2:
                continue
            for tgt in graph.successors(mid2):
                if not is_hgnc_node(graph, tgt) or tgt not in whitelist:
                    continue
                if tgt in (src, mid1, mid2) or tgt in excluded:
                    continue
                s3 = best_statement(graph.get_edge_data(mid2, tgt), require_incdec=True)
                if not s3:
                    continue
                row = _make_row(3, src, tgt, deg_map,
                                s1=s1, s2=s2, s3=s3, mid1=mid1, mid2=mid2)
                candidates.setdefault(tgt, []).append(
                    (float(s3.get("belief", 0.0)), row),
                )

    rows = []
    for items in candidates.values():
        items.sort(key=lambda x: x[0], reverse=True)
        keep = items if max_per_pair == 0 else items[:max_per_pair]
        rows.extend(r for _, r in keep)
    return rows


def _process_gene(raw_gene, gene, graph, whitelist, args):
    if not gene or gene not in graph or not is_hgnc_node(graph, gene):
        logger.warning("[%s] SKIP -- not in graph as HGNC", raw_gene)
        return None

    deg_map = _load_full_deg_map(args.deg_dir, raw_gene, args.prefer_fdr)
    waterfall = not args.no_waterfall
    excluded: set = set()
    all_rows: list[dict] = []

    if 1 in args.hops:
        rows, found = _run_1hop(graph, gene, whitelist, deg_map)
        logger.info("[%s] 1-hop: %d paths, %d targets", raw_gene, len(rows), len(found))
        all_rows.extend(rows)
        if waterfall:
            excluded |= found

    if 2 in args.hops:
        rows, found = _run_2hop(graph, gene, whitelist, deg_map, excluded)
        logger.info("[%s] 2-hop: %d paths, %d new targets", raw_gene, len(rows), len(found))
        all_rows.extend(rows)
        if waterfall:
            excluded |= found

    if 3 in args.hops:
        rows = _run_3hop(graph, gene, whitelist, deg_map, excluded, args.max_paths_3hop)
        logger.info("[%s] 3-hop: %d paths", raw_gene, len(rows))
        all_rows.extend(rows)

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    for col in _COLUMN_ORDER:
        if col not in df.columns:
            df[col] = pd.NA
    return df[_COLUMN_ORDER]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="INDRA multi-hop path discovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--graph-pkl", required=True)
    ap.add_argument("--endo-list", required=True,
                    help="Gene whitelist CSV")
    ap.add_argument("--deg-dir", required=True,
                    help="Folder with <GENE>_vs_control.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--genes", nargs="+")
    ap.add_argument("--genes-csv", default=None)
    ap.add_argument("--gene-col", default="Gene")
    ap.add_argument("--flag-col", default=None)
    ap.add_argument("--flag-val", default=None)
    ap.add_argument("--endo-col", default="gene")
    ap.add_argument("--hops", nargs="+", type=int, choices=[1, 2, 3],
                    default=[1, 2, 3])
    ap.add_argument("--no-waterfall", action="store_true")
    ap.add_argument("--max-paths-3hop", type=int, default=1)
    ap.add_argument("--prefer-fdr", action="store_true")
    ap.add_argument("--combine-output", action="store_true")
    ap.add_argument("--out-filename", default="all_genes_all_hops.csv")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.genes and not args.genes_csv:
        logger.error("Supply either --genes or --genes-csv")
        sys.exit(1)

    for label, path in [("--graph-pkl", args.graph_pkl),
                        ("--endo-list", args.endo_list),
                        ("--deg-dir", args.deg_dir)]:
        if not os.path.exists(path):
            logger.error("%s not found: %s", label, path)
            sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    graph, _ = load_graph(args.graph_pkl)
    whitelist = load_gene_set(args.endo_list, gene_col=args.endo_col)
    whitelist = {g for g in whitelist if g in graph and is_hgnc_node(graph, g)}
    logger.info("Whitelist in-graph: %d", len(whitelist))

    if args.genes:
        raw_genes = [g.strip() for g in args.genes if g.strip()]
    else:
        df = pd.read_csv(args.genes_csv, low_memory=False)
        if args.flag_col and args.flag_val and args.flag_col in df.columns:
            df = df[df[args.flag_col] == args.flag_val]
        raw_genes = [
            s.strip() for s in df[args.gene_col].dropna().astype(str) if s.strip()
        ]

    gene_pairs = [(r, normalize_hgnc_symbol(r)) for r in raw_genes]
    logger.info("Source genes: %s", [p[0] for p in gene_pairs])

    results: dict[str, pd.DataFrame] = {}

    def _job(raw, norm):
        return raw, _process_gene(raw, norm, graph, whitelist, args)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_job, r, n): r for r, n in gene_pairs}
            for fut in as_completed(futs):
                raw, df = fut.result()
                if df is not None:
                    results[raw] = df
    else:
        for raw, norm in gene_pairs:
            _, df = _job(raw, norm)
            if df is not None:
                results[raw] = df

    if not results:
        logger.error("No results produced. Exiting.")
        sys.exit(1)

    if args.combine_output:
        combined = pd.concat(list(results.values()), ignore_index=True)
        out = os.path.join(args.out_dir, args.out_filename)
        combined.to_csv(out, index=False)
        logger.info("Combined: %d rows -> %s", len(combined), out)
    else:
        for raw, df in results.items():
            out = os.path.join(args.out_dir, f"{raw}_all_hops.csv")
            df.to_csv(out, index=False)
            logger.info("%s: %d rows -> %s", raw, len(df), out)


if __name__ == "__main__":
    main()
