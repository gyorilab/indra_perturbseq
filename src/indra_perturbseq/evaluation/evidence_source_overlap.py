"""Compute pair-level evidence-source overlap counts for pathways.

Pathway classes (mutually exclusive):
1) DB-only: every edge has at least one DB source.
2) Reader-only: no edge has DB evidence.
3) Mixed (DB+reader): at least one DB-supported edge and at least one reader-only edge.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

from indra_perturbseq.graph import load_graph
from indra_perturbseq.utils.evidence_overlap_helpers import (
    build_db_only_vs_db_reader_counts,
    build_priority_db_only_vs_db_reader_counts,
)
from indra_perturbseq.utils.selected_statement_cache import (
    load_selected_statement_cache,
    stmt_has_db_from_source_counts,
)

logger = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)

DB_SOURCES = {
    "acsn",
    "bel",
    "bel_lc",
    "biogrid",
    "biopax",
    "cbn",
    "conib",
    "creeds",
    "crog",
    "ctd",
    "dgi",
    "dip",
    "drugbank",
    "hprd",
    "intact",
    "kegg",
    "minerva",
    "mint",
    "msigdb",
    "omnipath",
    "pathwaycommons",
    "pc",
    "phosphosite",
    "phosphoelm",
    "pid",
    "reactome",
    "signor",
    "tas",
    "trrust",
    "ubibrowser",
    "virhostnet",
    "wormbase",
}

ALIASES = {"bel_lc": "bel"}
URL_HASH_RE = re.compile(r"/from_hash/(-?\d+)")
INTEGER_RE = re.compile(r"^-?\d+$")


def _norm_gene(value: object) -> str | None:
    s = str(value).strip().upper()
    return s or None


def _norm_source_name(name: object) -> str:
    s = str(name).strip().lower()
    if ":" in s:
        s = s.split(":", 1)[0]
    return ALIASES.get(s, s)


def _parse_hash_literal(value: object) -> int | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or not INTEGER_RE.match(s):
        return None
    try:
        return int(s)
    except Exception:
        return None


def _parse_hash_from_url(url: object) -> int | None:
    if url is None:
        return None
    s = str(url).strip()
    if not s:
        return None
    m = URL_HASH_RE.search(s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _choose_hash(row: dict[str, str], hash_col: str, url_col: str) -> int | None:
    h = _parse_hash_from_url(row.get(url_col))
    if h is not None:
        return h
    return _parse_hash_literal(row.get(hash_col))


def _stmt_has_db(stmt: dict) -> bool:
    sc = stmt.get("source_counts")
    if not isinstance(sc, dict) or not sc:
        return False
    return any(_norm_source_name(k) in DB_SOURCES for k in sc.keys())


def _row_source_counts_has_db(row: dict[str, str]) -> bool | None:
    raw = row.get("source_counts")
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    return stmt_has_db_from_source_counts(s)


def _build_hash_db_map(graph) -> dict[int, bool]:
    logger.info("Building stmt_hash -> edge DB support map...")
    hash_has_db: dict[int, bool] = {}
    for _, _, data in graph.edges(data=True):
        stmts = (data or {}).get("statements", [])
        if not isinstance(stmts, list):
            continue
        for stmt in stmts:
            if not isinstance(stmt, dict):
                continue
            h = _parse_hash_literal(stmt.get("stmt_hash"))
            if h is None:
                continue
            has_db = _stmt_has_db(stmt)
            prev = hash_has_db.get(h)
            hash_has_db[h] = bool(has_db) if prev is None else (prev or has_db)
    logger.info("Mapped %d unique statement hashes", len(hash_has_db))
    return hash_has_db


def _classify_pathway(edge_db_flags: list[bool]) -> str:
    if all(edge_db_flags):
        return "db_only"
    if not any(edge_db_flags):
        return "reader_only"
    return "mixed"


def _row_is_explained(row: dict[str, str]) -> bool:
    label = str(row.get("directional consistency", "")).strip()
    if not label:
        return True
    return label != "Inconsistent"


def _venn_counts(a: set[tuple[str, str]], b: set[tuple[str, str]], c: set[tuple[str, str]]) -> dict[str, int]:
    only_a = len(a - b - c)
    only_b = len(b - a - c)
    only_c = len(c - a - b)
    ab_only = len((a & b) - c)
    ac_only = len((a & c) - b)
    bc_only = len((b & c) - a)
    abc = len(a & b & c)
    return {
        "100_db_only": only_a,
        "010_mixed": only_b,
        "001_reader_only": only_c,
        "110_db_and_mixed": ab_only,
        "101_db_and_reader": ac_only,
        "011_mixed_and_reader": bc_only,
        "111_all_three": abc,
        "union_pairs": len(a | b | c),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Build pair-level DB-only / Mixed / Reader-only sets and Venn counts.",
    )
    ap.add_argument("--graph-pkl", default=None)
    ap.add_argument(
        "--selected-edge-cache",
        default=None,
        help="Optional selected-edge cache CSV. Used before falling back to graph lookups.",
    )
    ap.add_argument("--hop1-csv", required=True)
    ap.add_argument("--hop2-csv", required=True)
    ap.add_argument("--hop3-csv", required=True)
    ap.add_argument(
        "--output-json",
        default="outputs/tables/db_mixed_reader_venn_counts.json",
    )
    ap.add_argument(
        "--output-hop-csv",
        default=None,
        help=(
            "Optional per-hop overlap CSV. "
            "If omitted, per-hop stats are logged but no CSV is written."
        ),
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    selected_edge_cache = load_selected_statement_cache(args.selected_edge_cache)
    hash_has_db: dict[int, bool] | None = None
    if args.graph_pkl:
        graph, _ = load_graph(args.graph_pkl)
        hash_has_db = _build_hash_db_map(graph)
    elif selected_edge_cache is None:
        raise SystemExit("Provide either --graph-pkl or --selected-edge-cache")

    hop_specs = {
        "1-hop": (
            Path(args.hop1_csv),
            [("source", "target", True, "stmt_hash", "hop1_indra_url", "source_counts")],
        ),
        "2-hop": (
            Path(args.hop2_csv),
            [
                ("source", "intermediate", False, "hop1_hash", "hop1_indra_url", None),
                ("intermediate", "target", True, "hop2_hash", "hop2_indra_url", None),
            ],
        ),
        "3-hop": (
            Path(args.hop3_csv),
            [
                ("source", "intermediate_1", False, "hop1_hash", "hop1_indra_url", None),
                ("intermediate_1", "intermediate_2", False, "hop2_hash", "hop2_indra_url", None),
                ("intermediate_2", "target", True, "hop3_hash", "hop3_indra_url", None),
            ],
        ),
    }

    # Pair-level pathway class support sets (combined and per-hop).
    combined_sets = {"db_only": set(), "mixed": set(), "reader_only": set()}
    per_hop_sets: dict[str, dict[str, set[tuple[str, str]]]] = {}

    unresolved_edges = 0
    for hop, (csv_path, edge_specs) in hop_specs.items():
        hop_sets = {"db_only": set(), "mixed": set(), "reader_only": set()}
        logger.info("Reading %s", csv_path)
        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if not _row_is_explained(row):
                    continue
                src = _norm_gene(row.get("source"))
                tgt = _norm_gene(row.get("target"))
                if not src or not tgt or src == tgt:
                    continue
                pair = (src, tgt)

                edge_db_flags: list[bool] = []
                for src_col, tgt_col, require_incdec, hash_col, url_col, source_counts_col in edge_specs:
                    has_db: bool | None = None
                    if source_counts_col:
                        has_db = _row_source_counts_has_db(row)
                    if has_db is None and selected_edge_cache is not None:
                        rec = selected_edge_cache.get(
                            row.get(src_col),
                            row.get(tgt_col),
                            require_incdec,
                        )
                        if rec is not None:
                            has_db = bool(rec.get("has_db"))
                    if has_db is None and hash_has_db is not None:
                        h = _choose_hash(row, hash_col, url_col)
                        if h is not None:
                            has_db = bool(hash_has_db.get(h, False))
                    if has_db is None:
                        unresolved_edges += 1
                        has_db = False
                    edge_db_flags.append(has_db)

                cls = _classify_pathway(edge_db_flags)
                hop_sets[cls].add(pair)
                combined_sets[cls].add(pair)

        per_hop_sets[hop] = hop_sets

    combined_counts = _venn_counts(
        combined_sets["db_only"],
        combined_sets["mixed"],
        combined_sets["reader_only"],
    )
    # Per-hop count rows for optional table.
    hop_rows: list[dict[str, int | str]] = []
    per_hop_counts: dict[str, dict[str, int]] = {}
    for hop in ("1-hop", "2-hop", "3-hop"):
        counts = _venn_counts(
            per_hop_sets[hop]["db_only"],
            per_hop_sets[hop]["mixed"],
            per_hop_sets[hop]["reader_only"],
        )
        per_hop_counts[hop] = counts
        hop_rows.append({"hop": hop, **counts})

    # Log per-hop summary rows regardless of CSV export.
    for row in hop_rows:
        logger.info("Per-hop overlap [%s]: %s", row["hop"], row)

    # Save per-hop CSV only when explicitly requested.
    if args.output_hop_csv:
        out_hop_csv = Path(args.output_hop_csv)
        out_hop_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["hop", "100_db_only", "010_mixed", "001_reader_only", "110_db_and_mixed", "101_db_and_reader", "011_mixed_and_reader", "111_all_three", "union_pairs"]
        with out_hop_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in hop_rows:
                writer.writerow(row)
        logger.info("Saved per-hop overlap CSV: %s", out_hop_csv)

    out = {
        "definitions": {
            "db_only": "all edges DB-supported",
            "mixed": "at least one DB-supported edge and at least one reader-only edge",
            "reader_only": "no edge DB-supported",
        },
        "combined_all_hops": combined_counts,
        "combined_db_only_vs_db_plus_reader_overlap": build_db_only_vs_db_reader_counts(combined_counts),
        "combined_db_only_vs_db_plus_reader_priority": build_priority_db_only_vs_db_reader_counts(combined_counts),
        "per_hop": per_hop_counts,
        "per_hop_db_only_vs_db_plus_reader_priority": {
            hop: build_priority_db_only_vs_db_reader_counts(counts)
            for hop, counts in per_hop_counts.items()
        },
        "set_sizes_combined": {
            "db_only_pairs": len(combined_sets["db_only"]),
            "mixed_pairs": len(combined_sets["mixed"]),
            "reader_only_pairs": len(combined_sets["reader_only"]),
        },
        "unresolved_edges": unresolved_edges,
        "selected_edge_cache": str(args.selected_edge_cache) if args.selected_edge_cache else None,
        "graph_pkl": str(args.graph_pkl) if args.graph_pkl else None,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    logger.info("Saved overlap JSON: %s", out_json)
    logger.info("Combined counts: %s", combined_counts)


if __name__ == "__main__":
    main()
