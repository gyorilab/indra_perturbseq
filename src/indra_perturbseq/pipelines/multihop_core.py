"""Core functions for multi-hop path discovery.
This module provides pipeline execution and command-line workflow orchestration.
"""

from __future__ import annotations

import os
import re

import networkx as nx
import pandas as pd
from indra.explanation.pathfinding.pathfinding import shortest_simple_paths

from indra_perturbseq.deg import pick_sig_column
from indra_perturbseq.graph import is_hgnc_node
from indra_perturbseq.hgnc import normalize_hgnc_symbol
from indra_perturbseq.statements import best_statement, indra_html_url


def column_order(max_hop: int) -> list[str]:
    cols = ["hop", "source"]
    cols.extend([f"intermediate_{i}" for i in range(1, max_hop)])
    cols.append("target")
    for prefix in ("stmt_type", "belief", "evidence_count"):
        cols.extend([f"{prefix}_{i}" for i in range(1, max_hop + 1)])
    cols.extend(["logfoldchange", "pval", "pvals_adj"])
    cols.extend([f"hop{i}_hash" for i in range(1, max_hop + 1)])
    cols.extend([f"hop{i}_indra_url" for i in range(1, max_hop + 1)])
    return cols


def _load_full_deg_map(deg_dir: str, raw_gene: str, prefer_fdr: bool, logger) -> dict:
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

    deg_map: dict[str, dict] = {}
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


def _read_gene_tokens_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    return [t.strip() for t in re.split(r"[\s,;]+", text) if t.strip()]


def _load_genes_from_csv(
    csv_path: str,
    gene_column: str,
    filter_column: str | None = None,
    filter_value: str | None = None,
) -> list[str]:
    df = pd.read_csv(csv_path, low_memory=False)
    if gene_column not in df.columns:
        raise ValueError(
            f"CSV missing gene column '{gene_column}': {csv_path}. "
            f"Columns={df.columns.tolist()}"
        )
    if filter_column and filter_value and filter_column in df.columns:
        df = df[df[filter_column] == filter_value].copy()
    return [s.strip() for s in df[gene_column].dropna().astype(str) if s.strip()]


def _dedup_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _normalize_gene_set(values: list[str]) -> set[str]:
    out: set[str] = set()
    for v in values:
        n = normalize_hgnc_symbol(v)
        if n:
            out.add(n)
    return out


def load_sources_from_args(args) -> list[str]:
    if args.genes:
        raw_genes = [g.strip() for g in args.genes if g.strip()]
    elif args.source_genes_file:
        raw_genes = _read_gene_tokens_file(args.source_genes_file)
    elif args.source_genes_csv:
        raw_genes = _load_genes_from_csv(
            args.source_genes_csv,
            gene_column=args.gene_column,
            filter_column=args.filter_column,
            filter_value=args.filter_value,
        )
    else:
        raise ValueError("Supply one of --genes, --source-genes-file, --source-genes-csv")

    raw_genes = _dedup_preserve_order(raw_genes)
    if args.limit_genes > 0:
        raw_genes = raw_genes[:args.limit_genes]
    return raw_genes


def load_targets_from_args(args) -> set[str] | None:
    raw: list[str] = []
    if args.target_genes:
        raw.extend([g.strip() for g in args.target_genes if g.strip()])
    if args.target_genes_file:
        raw.extend(_read_gene_tokens_file(args.target_genes_file))
    if args.target_genes_csv:
        raw.extend(_load_genes_from_csv(
            args.target_genes_csv,
            gene_column=args.target_column,
            filter_column=args.target_filter_column,
            filter_value=args.target_filter_value,
        ))
    if not raw:
        return None
    return _normalize_gene_set(_dedup_preserve_order(raw))


def load_intermediates_from_args(args) -> set[str]:
    raw: list[str] = []
    if args.intermediate_genes:
        raw.extend([g.strip() for g in args.intermediate_genes if g.strip()])
    if args.intermediate_genes_file:
        raw.extend(_read_gene_tokens_file(args.intermediate_genes_file))
    if args.intermediate_genes_csv:
        raw.extend(_load_genes_from_csv(
            args.intermediate_genes_csv,
            gene_column=args.endothelial_column,
        ))
    raw = _dedup_preserve_order(raw)
    if not raw:
        raise ValueError(
            "Supply intermediates via --intermediate-genes, "
            "--intermediate-genes-file, or --intermediate-genes-csv "
            "(legacy alias: --endothelial-list)."
        )
    return _normalize_gene_set(raw)


def _make_row_from_path(
    hop: int,
    path: list[str],
    statements: list[dict],
    deg_map: dict[str, dict],
) -> dict:
    src = path[0]
    tgt = path[-1]
    d = deg_map.get(tgt, {})
    row = {
        "hop": hop,
        "source": src,
        "target": tgt,
        "logfoldchange": d.get("logfoldchange", float("nan")),
        "pval": d.get("pval", float("nan")),
        "pvals_adj": d.get("pvals_adj", float("nan")),
    }
    for i, mid in enumerate(path[1:-1], start=1):
        row[f"intermediate_{i}"] = mid
    for i, s in enumerate(statements, start=1):
        stmt_hash = s.get("stmt_hash")
        row[f"stmt_type_{i}"] = s.get("stmt_type")
        row[f"belief_{i}"] = s.get("belief")
        row[f"evidence_count_{i}"] = s.get("evidence_count")
        row[f"hop{i}_hash"] = stmt_hash
        row[f"hop{i}_indra_url"] = indra_html_url(stmt_hash)
    return row


def _run_1hop(graph, src, target_set, deg_map):
    rows, found = [], set()
    for tgt in graph.successors(src):
        if not is_hgnc_node(graph, tgt) or tgt not in target_set or tgt == src:
            continue
        s1 = best_statement(graph.get_edge_data(src, tgt), require_incdec=True)
        if not s1:
            continue
        rows.append(_make_row_from_path(1, [src, tgt], [s1], deg_map))
        found.add(tgt)
    return rows, found


def _run_2hop(graph, src, intermediate_set, target_set, deg_map, excluded):
    rows, found = [], set()
    for mid in graph.successors(src):
        if not is_hgnc_node(graph, mid) or mid not in intermediate_set or mid == src:
            continue
        s1 = best_statement(graph.get_edge_data(src, mid), require_incdec=False)
        if not s1:
            continue
        for tgt in graph.successors(mid):
            if not is_hgnc_node(graph, tgt) or tgt not in target_set:
                continue
            if tgt in (src, mid) or tgt in excluded:
                continue
            s2 = best_statement(graph.get_edge_data(mid, tgt), require_incdec=True)
            if not s2:
                continue
            rows.append(_make_row_from_path(2, [src, mid, tgt], [s1, s2], deg_map))
            found.add(tgt)
    return rows, found


def _run_3hop(graph, src, intermediate_set, target_set, deg_map, excluded, max_per_pair):
    candidates: dict[str, list[tuple[float, dict]]] = {}
    for mid1 in graph.successors(src):
        if not is_hgnc_node(graph, mid1) or mid1 not in intermediate_set or mid1 == src:
            continue
        s1 = best_statement(graph.get_edge_data(src, mid1), require_incdec=False)
        if not s1:
            continue
        for mid2 in graph.successors(mid1):
            if not is_hgnc_node(graph, mid2) or mid2 not in intermediate_set:
                continue
            if mid2 in (src, mid1):
                continue
            s2 = best_statement(graph.get_edge_data(mid1, mid2), require_incdec=False)
            if not s2:
                continue
            for tgt in graph.successors(mid2):
                if not is_hgnc_node(graph, tgt) or tgt not in target_set:
                    continue
                if tgt in (src, mid1, mid2) or tgt in excluded:
                    continue
                s3 = best_statement(graph.get_edge_data(mid2, tgt), require_incdec=True)
                if not s3:
                    continue
                row = _make_row_from_path(3, [src, mid1, mid2, tgt], [s1, s2, s3], deg_map)
                candidates.setdefault(tgt, []).append((float(s3.get("belief", 0.0)), row))

    rows = []
    for items in candidates.values():
        items.sort(key=lambda x: x[0], reverse=True)
        keep = items if max_per_pair == 0 else items[:max_per_pair]
        rows.extend(r for _, r in keep)
    return rows


def _run_nhop_pathfinding(
    graph: nx.DiGraph,
    src: str,
    intermediate_set: set[str],
    target_set: set[str],
    deg_map: dict[str, dict],
    excluded: set[str],
    hop: int,
    max_per_pair: int,
) -> list[dict]:
    expected_len = hop + 1
    rows: list[dict] = []

    for tgt in sorted(target_set):
        if tgt == src or tgt in excluded:
            continue
        try:
            path_gen = shortest_simple_paths(graph, src, tgt, weight=None)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        candidates: list[tuple[float, dict]] = []
        try:
            for path in path_gen:
                if len(path) < expected_len:
                    continue
                if len(path) > expected_len:
                    break

                mids = path[1:-1]
                if any((not is_hgnc_node(graph, n) or n not in intermediate_set) for n in mids):
                    continue

                statements: list[dict] = []
                ok = True
                for i in range(len(path) - 1):
                    require_incdec = i == (len(path) - 2)
                    s = best_statement(
                        graph.get_edge_data(path[i], path[i + 1]),
                        require_incdec=require_incdec,
                    )
                    if not s:
                        ok = False
                        break
                    statements.append(s)
                if not ok:
                    continue

                row = _make_row_from_path(hop, path, statements, deg_map)
                score = float(statements[-1].get("belief", 0.0))
                candidates.append((score, row))
        except nx.NetworkXNoPath:
            pass

        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0], reverse=True)
        keep = candidates if max_per_pair == 0 else candidates[:max_per_pair]
        rows.extend(r for _, r in keep)
    return rows


def process_gene(
    raw_gene: str,
    gene: str | None,
    graph,
    intermediate_set: set[str],
    target_set: set[str],
    args,
    max_hop: int,
    logger,
):
    if not gene or gene not in graph or not is_hgnc_node(graph, gene):
        logger.warning("[%s] SKIP -- not in graph as HGNC", raw_gene)
        return None

    deg_map = _load_full_deg_map(args.deg_dir, raw_gene, args.prefer_fdr, logger)
    waterfall = not args.no_waterfall
    excluded: set[str] = set()
    all_rows: list[dict] = []

    for hop in args.hops:
        if hop == 1:
            rows, found = _run_1hop(graph, gene, target_set, deg_map)
        elif hop == 2:
            rows, found = _run_2hop(
                graph, gene, intermediate_set, target_set, deg_map, excluded,
            )
        elif hop == 3:
            rows = _run_3hop(
                graph,
                gene,
                intermediate_set,
                target_set,
                deg_map,
                excluded,
                args.max_paths_3hop,
            )
            found = {r["target"] for r in rows}
        else:
            rows = _run_nhop_pathfinding(
                graph,
                gene,
                intermediate_set,
                target_set,
                deg_map,
                excluded,
                hop,
                args.max_paths_per_pair,
            )
            found = {r["target"] for r in rows}

        logger.info(
            "[%s] %d-hop: %d paths, %d targets",
            raw_gene,
            hop,
            len(rows),
            len(found),
        )
        all_rows.extend(rows)
        if waterfall:
            excluded |= found

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    cols = column_order(max_hop)
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[cols]
