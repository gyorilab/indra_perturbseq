"""Backfill a selected-edge cache from existing hop CSVs.

This is mainly useful for legacy pathway exports created before cache
persisting was added to extraction. It reconstructs edge-level selected
statement metadata from the exported rows and optionally supplements DB support
from the graph when multihop rows do not carry ``source_counts`` directly.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

from indra_perturbseq.graph import load_graph
from indra_perturbseq.utils.selected_statement_cache import (
    SelectedStatementCache,
    stmt_has_db_from_source_counts,
)

logger = logging.getLogger(__name__)
csv.field_size_limit(sys.maxsize)

INTEGER_RE = re.compile(r"^-?\d+$")
URL_HASH_RE = re.compile(r"/from_hash/(-?\d+)")


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
    s = str(url or "").strip()
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


def _norm_source_name(name: object) -> str:
    s = str(name or "").strip().lower()
    if ":" in s:
        s = s.split(":", 1)[0]
    return {"bel_lc": "bel"}.get(s, s)


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


def _stmt_has_db(stmt: dict) -> bool:
    sc = stmt.get("source_counts")
    if not isinstance(sc, dict) or not sc:
        return False
    return any(_norm_source_name(k) in DB_SOURCES for k in sc.keys())


def _build_hash_db_map(graph) -> dict[int, bool]:
    logger.info("Building stmt_hash -> DB support map from graph...")
    out: dict[int, bool] = {}
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
            prev = out.get(h)
            has_db = _stmt_has_db(stmt)
            out[h] = bool(has_db) if prev is None else (prev or has_db)
    logger.info("Mapped %d unique statement hashes", len(out))
    return out


def _ingest_csv(
    cache: SelectedStatementCache,
    csv_path: Path,
    edge_specs: list[tuple[str, str, bool, str, str, str, str, str | None]],
    hash_has_db: dict[int, bool] | None,
) -> None:
    logger.info("Reading %s", csv_path)
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for src_col, tgt_col, require_incdec, hash_col, url_col, stmt_col, belief_col, ev_col in edge_specs:
                h = _choose_hash(row, hash_col, url_col)
                has_db: bool | None = None
                if "source_counts" in row and row.get("source_counts"):
                    has_db = stmt_has_db_from_source_counts(row.get("source_counts"))
                elif h is not None and hash_has_db is not None:
                    has_db = bool(hash_has_db.get(h, False))
                cache.record_fields(
                    source=row.get(src_col),
                    target=row.get(tgt_col),
                    require_incdec=require_incdec,
                    stmt_hash="" if h is None else h,
                    stmt_type=row.get(stmt_col),
                    belief=row.get(belief_col),
                    evidence_count=row.get(ev_col),
                    has_db=has_db,
                )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build a selected-edge cache from hop CSVs.")
    ap.add_argument("--hop1-csv", required=True)
    ap.add_argument("--hop2-csv", required=True)
    ap.add_argument("--hop3-csv", required=True)
    ap.add_argument("--graph-pkl", default=None, help="Optional graph for backfilling DB support on multihop rows.")
    ap.add_argument("--output-cache", required=True, help="Output cache CSV path.")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    hash_has_db: dict[int, bool] | None = None
    if args.graph_pkl:
        graph, _ = load_graph(args.graph_pkl)
        hash_has_db = _build_hash_db_map(graph)

    cache = SelectedStatementCache()
    _ingest_csv(
        cache,
        Path(args.hop1_csv),
        [("source", "target", True, "stmt_hash", "hop1_indra_url", "stmt_type", "belief", "evidence_count")],
        hash_has_db,
    )
    _ingest_csv(
        cache,
        Path(args.hop2_csv),
        [
            ("source", "intermediate", False, "hop1_hash", "hop1_indra_url", "stmt_type_1", "belief_1", "evidence_1"),
            ("intermediate", "target", True, "hop2_hash", "hop2_indra_url", "stmt_type_2", "belief_2", "evidence_2"),
        ],
        hash_has_db,
    )
    _ingest_csv(
        cache,
        Path(args.hop3_csv),
        [
            ("source", "intermediate_1", False, "hop1_hash", "hop1_indra_url", "stmt_type_1", "belief_1", "evidence_1"),
            ("intermediate_1", "intermediate_2", False, "hop2_hash", "hop2_indra_url", "stmt_type_2", "belief_2", "evidence_2"),
            ("intermediate_2", "target", True, "hop3_hash", "hop3_indra_url", "stmt_type_3", "belief_3", "evidence_3"),
        ],
        hash_has_db,
    )
    cache.write_csv(args.output_cache)
    logger.info("Selected edge cache: %d records -> %s", len(cache), args.output_cache)


if __name__ == "__main__":
    main()
