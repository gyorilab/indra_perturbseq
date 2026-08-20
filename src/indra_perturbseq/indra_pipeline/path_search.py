"""INDRA graph path search for pipeline 1-hop and 2-hop outputs."""

from __future__ import annotations

import json
import math
from typing import Iterable

import pandas as pd

from indra_perturbseq.graph import is_hgnc_node
from indra_perturbseq.indra_pipeline.inputs import SourceRecord, TargetData
from indra_perturbseq.statements import indra_html_url


def _jsonish(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _stmt_sort_key(stmt: dict) -> tuple[float, int]:
    try:
        belief = float(stmt.get("belief"))
    except (TypeError, ValueError):
        belief = -math.inf
    try:
        ev_count = int(stmt.get("evidence_count") or 0)
    except (TypeError, ValueError):
        ev_count = 0
    return belief, ev_count


def iter_edge_statements(
    edge_data: dict | None,
    stmt_types: Iterable[str],
    representative_only: bool = False,
) -> list[dict]:
    """Return all configured statements on an edge, or one representative."""
    allowed = set(stmt_types)
    statements = (edge_data or {}).get("statements", [])
    if not isinstance(statements, list):
        return []
    kept = [
        s for s in statements
        if isinstance(s, dict) and s.get("stmt_type") in allowed
    ]
    if representative_only and kept:
        return [max(kept, key=_stmt_sort_key)]
    return kept


def _target_records(target_data: TargetData) -> dict[str, dict]:
    if target_data.targets.empty:
        return {}
    return target_data.targets.set_index("target").to_dict("index")


def run_1hop(
    graph,
    sources: list[SourceRecord],
    targets_by_source: dict[str, TargetData],
    stmt_types: list[str],
    representative_only: bool = False,
) -> pd.DataFrame:
    """Find source -> target paths for all sources."""
    rows: list[dict] = []
    for source in sources:
        if not source.found or not source.node:
            continue
        target_data = targets_by_source.get(source.raw)
        if target_data is None:
            continue
        records = _target_records(target_data)
        for target, meta in records.items():
            if target not in graph or not is_hgnc_node(graph, target):
                continue
            if not graph.has_edge(source.node, target):
                continue
            for stmt in iter_edge_statements(
                graph.get_edge_data(source.node, target),
                stmt_types,
                representative_only,
            ):
                stmt_hash = stmt.get("stmt_hash")
                rows.append({
                    "source": source.node,
                    "target": target,
                    "stmt_type": stmt.get("stmt_type"),
                    "DEGs-group": meta.get("DEGs-group"),
                    "logfoldchange": meta.get("logfoldchange"),
                    "pval": meta.get("pval"),
                    "belief": stmt.get("belief"),
                    "evidence_count": stmt.get("evidence_count"),
                    "source_counts": _jsonish(stmt.get("source_counts")),
                    "stmt_hash": stmt_hash,
                    "indra_url": indra_html_url(stmt_hash),
                })
    return pd.DataFrame(rows)


def run_2hop(
    graph,
    sources: list[SourceRecord],
    targets_by_source: dict[str, TargetData],
    intermediates: set[str],
    stmt_types: list[str],
    representative_only: bool = False,
) -> pd.DataFrame:
    """Find source -> intermediate -> target paths for all sources."""
    rows: list[dict] = []
    for source in sources:
        if not source.found or not source.node or source.node not in graph:
            continue
        target_data = targets_by_source.get(source.raw)
        if target_data is None:
            continue
        records = _target_records(target_data)
        graph_targets = {
            t for t in records
            if t in graph and is_hgnc_node(graph, t)
        }
        if not graph_targets:
            continue

        for mid in graph.successors(source.node):
            if mid not in intermediates or not is_hgnc_node(graph, mid):
                continue
            hop1_statements = iter_edge_statements(
                graph.get_edge_data(source.node, mid),
                stmt_types,
                representative_only,
            )
            if not hop1_statements:
                continue
            for target in set(graph.successors(mid)) & graph_targets:
                hop2_statements = iter_edge_statements(
                    graph.get_edge_data(mid, target),
                    stmt_types,
                    representative_only,
                )
                if not hop2_statements:
                    continue
                meta = records[target]
                for stmt1 in hop1_statements:
                    for stmt2 in hop2_statements:
                        h1 = stmt1.get("stmt_hash")
                        h2 = stmt2.get("stmt_hash")
                        rows.append({
                            "source": source.node,
                            "intermediate": mid,
                            "target": target,
                            "stmt_type_1": stmt1.get("stmt_type"),
                            "stmt_type_2": stmt2.get("stmt_type"),
                            "DEGs-group": meta.get("DEGs-group"),
                            "logfoldchange": meta.get("logfoldchange"),
                            "pval": meta.get("pval"),
                            "belief_1": stmt1.get("belief"),
                            "belief_2": stmt2.get("belief"),
                            "evidence_count_1": stmt1.get("evidence_count"),
                            "evidence_count_2": stmt2.get("evidence_count"),
                            "source_counts_1": _jsonish(stmt1.get("source_counts")),
                            "source_counts_2": _jsonish(stmt2.get("source_counts")),
                            "stmt_hash_1": h1,
                            "stmt_hash_2": h2,
                            "indra_url_1": indra_html_url(h1),
                            "indra_url_2": indra_html_url(h2),
                        })
    return pd.DataFrame(rows)


def _metadata_from_row(row: dict) -> dict:
    reserved = {
        "source_raw", "target_raw", "source_node", "target_node",
        "source_ns", "target_ns", "source_id", "target_id",
        "source_found", "target_found",
        "pvalue", "logfoldchange",
    }
    return {k: v for k, v in row.items() if k not in reserved}


def _pair_rows(source_targets: pd.DataFrame) -> dict[tuple[str, str], list[dict]]:
    pairs: dict[tuple[str, str], list[dict]] = {}
    for row in source_targets.to_dict("records"):
        src, tgt = row.get("source_node"), row.get("target_node")
        if src and tgt:
            pairs.setdefault((src, tgt), []).append(row)
    return pairs


def run_1hop_source_target_table(
    graph,
    source_targets: pd.DataFrame,
    stmt_types: list[str],
    representative_only: bool = False,
) -> pd.DataFrame:
    """Find source -> target paths for paired input rows."""
    rows: list[dict] = []
    if source_targets.empty:
        return pd.DataFrame()
    for (src, tgt), metadata_rows in _pair_rows(source_targets).items():
        if not src or not tgt or src not in graph or tgt not in graph:
            continue
        if not graph.has_edge(src, tgt):
            continue
        for stmt in iter_edge_statements(
            graph.get_edge_data(src, tgt),
            stmt_types,
            representative_only,
        ):
            stmt_hash = stmt.get("stmt_hash")
            for row in metadata_rows:
                out = {
                    "source": src,
                    "target": tgt,
                    "source_raw": row.get("source_raw"),
                    "target_raw": row.get("target_raw"),
                    "stmt_type": stmt.get("stmt_type"),
                    "logfoldchange": row.get("logfoldchange"),
                    "pval": row.get("pvalue"),
                    "belief": stmt.get("belief"),
                    "evidence_count": stmt.get("evidence_count"),
                    "source_counts": _jsonish(stmt.get("source_counts")),
                    "stmt_hash": stmt_hash,
                    "indra_url": indra_html_url(stmt_hash),
                }
                out.update(_metadata_from_row(row))
                rows.append(out)
    return pd.DataFrame(rows)


def run_2hop_source_target_table(
    graph,
    source_targets: pd.DataFrame,
    intermediates: set[str],
    stmt_types: list[str],
    representative_only: bool = False,
) -> pd.DataFrame:
    """Find source -> intermediate -> target paths for paired input rows."""
    rows: list[dict] = []
    if source_targets.empty:
        return pd.DataFrame()
    pair_rows = _pair_rows(source_targets)
    targets_by_source: dict[str, set[str]] = {}
    for src, tgt in pair_rows:
        if src in graph and tgt in graph:
            targets_by_source.setdefault(src, set()).add(tgt)

    for src, target_set in targets_by_source.items():
        if src not in graph:
            continue
        for mid in graph.successors(src):
            if mid not in intermediates:
                continue
            hop1_statements = iter_edge_statements(
                graph.get_edge_data(src, mid),
                stmt_types,
                representative_only,
            )
            if not hop1_statements:
                continue
            for tgt in set(graph.successors(mid)) & target_set:
                hop2_statements = iter_edge_statements(
                    graph.get_edge_data(mid, tgt),
                    stmt_types,
                    representative_only,
                )
                for stmt1 in hop1_statements:
                    for stmt2 in hop2_statements:
                        h1 = stmt1.get("stmt_hash")
                        h2 = stmt2.get("stmt_hash")
                        for row in pair_rows[(src, tgt)]:
                            out = {
                                "source": src,
                                "intermediate": mid,
                                "target": tgt,
                                "source_raw": row.get("source_raw"),
                                "target_raw": row.get("target_raw"),
                                "stmt_type_1": stmt1.get("stmt_type"),
                                "stmt_type_2": stmt2.get("stmt_type"),
                                "logfoldchange": row.get("logfoldchange"),
                                "pval": row.get("pvalue"),
                                "belief_1": stmt1.get("belief"),
                                "belief_2": stmt2.get("belief"),
                                "evidence_count_1": stmt1.get("evidence_count"),
                                "evidence_count_2": stmt2.get("evidence_count"),
                                "source_counts_1": _jsonish(stmt1.get("source_counts")),
                                "source_counts_2": _jsonish(stmt2.get("source_counts")),
                                "stmt_hash_1": h1,
                                "stmt_hash_2": h2,
                                "indra_url_1": indra_html_url(h1),
                                "indra_url_2": indra_html_url(h2),
                            }
                            out.update(_metadata_from_row(row))
                            rows.append(out)
    return pd.DataFrame(rows)
