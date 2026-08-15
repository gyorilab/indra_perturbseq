"""Compute additional reader-added causal paths per (source, target) pair.

For each pair, this computes:
    delta_paths = N_all_paths - N_db_only_paths

where ``N_all_paths`` is the number of 1/2/3-hop pathway rows for the pair and
``N_db_only_paths`` is the subset where every edge has at least one DB source.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

from indra_perturbseq.graph import load_graph
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
    logger.info("Building stmt_hash -> DB-only-edge support map...")
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


def _row_is_explained(row: dict[str, str]) -> bool:
    label = str(row.get("directional consistency", "")).strip()
    if not label:
        return True
    return label != "Inconsistent"


def _compute_pair_path_counts(
    hash_has_db: dict[int, bool] | None,
    selected_edge_cache,
    hop1_csv: Path,
    hop2_csv: Path,
    hop3_csv: Path,
) -> dict[tuple[str, str], dict[str, int]]:
    hop_specs = [
        (hop1_csv, [("source", "target", True, "stmt_hash", "hop1_indra_url", "source_counts")]),
        (
            hop2_csv,
            [
                ("source", "intermediate", False, "hop1_hash", "hop1_indra_url", None),
                ("intermediate", "target", True, "hop2_hash", "hop2_indra_url", None),
            ],
        ),
        (
            hop3_csv,
            [
                ("source", "intermediate_1", False, "hop1_hash", "hop1_indra_url", None),
                ("intermediate_1", "intermediate_2", False, "hop2_hash", "hop2_indra_url", None),
                ("intermediate_2", "target", True, "hop3_hash", "hop3_indra_url", None),
            ],
        ),
    ]

    pair_counts: dict[tuple[str, str], dict[str, int]] = {}
    for csv_path, edge_specs in hop_specs:
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
                    edge_db_flags.append(bool(has_db) if has_db is not None else False)
                path_db_only = all(edge_db_flags)

                rec = pair_counts.setdefault(pair, {"all_paths": 0, "db_only_paths": 0})
                rec["all_paths"] += 1
                if path_db_only:
                    rec["db_only_paths"] += 1

    return pair_counts


def _write_pair_table(path: Path, pair_counts: dict[tuple[str, str], dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (src, tgt), rec in pair_counts.items():
        delta = rec["all_paths"] - rec["db_only_paths"]
        rows.append((delta, rec["all_paths"], rec["db_only_paths"], src, tgt))
    rows.sort(reverse=True)

    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source", "target", "all_paths", "db_only_paths", "additional_paths"])
        for delta, all_paths, db_only_paths, src, tgt in rows:
            writer.writerow([src, tgt, all_paths, db_only_paths, delta])
    logger.info("Wrote pair delta table: %s", path)


def _load_pair_table(path: Path) -> dict[tuple[str, str], dict[str, int]]:
    pair_counts: dict[tuple[str, str], dict[str, int]] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"source", "target", "all_paths", "db_only_paths"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"{path} must include columns {sorted(required)}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            src = _norm_gene(row.get("source"))
            tgt = _norm_gene(row.get("target"))
            if not src or not tgt or src == tgt:
                continue
            try:
                all_paths = int(row.get("all_paths", "0"))
                db_only_paths = int(row.get("db_only_paths", "0"))
            except Exception as exc:
                raise ValueError(f"Failed to parse counts in {path}: {row}") from exc
            pair_counts[(src, tgt)] = {
                "all_paths": all_paths,
                "db_only_paths": db_only_paths,
            }
    logger.info("Loaded pair delta table: %s (%d pairs)", path, len(pair_counts))
    return pair_counts


def summarize_deltas(deltas: list[int]) -> dict[str, float | dict[str, float]]:
    """Return summary statistics for additional-mechanism deltas."""
    arr = np.asarray(deltas, dtype=int)
    if len(arr) == 0:
        return {
            "n_pairs": 0,
            "min_delta": 0,
            "max_delta": 0,
            "mean_delta": float("nan"),
            "median_delta": float("nan"),
            "n_zero_delta_pairs": 0,
            "n_positive_delta_pairs": 0,
            "tail_focus": {
                "n_pairs_total": 0,
                "max_delta": 0,
                "p95": float("nan"),
                "p99": float("nan"),
                "tail_start": 10,
                "tail_marker": 0,
                "n_pairs_delta_ge_tail_start": 0,
                "pct_pairs_delta_ge_tail_start": float("nan"),
                "n_pairs_delta_ge_tail_marker": 0,
                "pct_pairs_delta_ge_tail_marker": float("nan"),
            },
        }

    tail_start = 10
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    marker = int(round(p99))
    n_tail_start = int(np.sum(arr >= tail_start))
    n_marker = int(np.sum(arr >= marker))

    return {
        "n_pairs": int(len(arr)),
        "min_delta": int(arr.min()),
        "max_delta": int(arr.max()),
        "mean_delta": float(arr.mean()),
        "median_delta": float(np.median(arr)),
        "n_zero_delta_pairs": int(np.sum(arr == 0)),
        "n_positive_delta_pairs": int(np.sum(arr > 0)),
        "tail_focus": {
            "n_pairs_total": int(len(arr)),
            "max_delta": int(arr.max()),
            "p95": p95,
            "p99": p99,
            "tail_start": tail_start,
            "tail_marker": marker,
            "n_pairs_delta_ge_tail_start": n_tail_start,
            "pct_pairs_delta_ge_tail_start": float(n_tail_start / len(arr) * 100.0),
            "n_pairs_delta_ge_tail_marker": n_marker,
            "pct_pairs_delta_ge_tail_marker": float(n_marker / len(arr) * 100.0),
        },
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compute per-pair additional-mechanism deltas, where "
            "Δ = (#paths with DB+reader) - (#DB-only paths)."
        ),
    )
    ap.add_argument("--graph-pkl", default=None, help="INDRA network export .pkl")
    ap.add_argument(
        "--selected-edge-cache",
        default=None,
        help="Optional selected-edge cache CSV. Used before falling back to graph lookups.",
    )
    ap.add_argument("--hop1-csv", required=True, help="1-hop pathway CSV")
    ap.add_argument("--hop2-csv", required=True, help="2-hop pathway CSV")
    ap.add_argument("--hop3-csv", required=True, help="3-hop pathway CSV")
    ap.add_argument(
        "--output-pair-csv",
        default="outputs/tables/additional_paths_per_pair.csv",
        help="Output per-pair delta table CSV",
    )
    ap.add_argument(
        "--input-pair-csv",
        default=None,
        help=(
            "Optional precomputed pair table with columns: source,target,all_paths,db_only_paths. "
            "If provided, graph + hop CSV computation is skipped."
        ),
    )
    ap.add_argument(
        "--output-summary-json",
        default="outputs/tables/additional_paths_hist_summary.json",
        help="Output summary JSON",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.input_pair_csv:
        pair_counts = _load_pair_table(Path(args.input_pair_csv))
    else:
        selected_edge_cache = load_selected_statement_cache(args.selected_edge_cache)
        hash_has_db: dict[int, bool] | None = None
        if args.graph_pkl:
            graph, _ = load_graph(args.graph_pkl)
            hash_has_db = _build_hash_db_map(graph)
        elif selected_edge_cache is None:
            raise SystemExit("Provide either --graph-pkl or --selected-edge-cache")
        pair_counts = _compute_pair_path_counts(
            hash_has_db=hash_has_db,
            selected_edge_cache=selected_edge_cache,
            hop1_csv=Path(args.hop1_csv),
            hop2_csv=Path(args.hop2_csv),
            hop3_csv=Path(args.hop3_csv),
        )
        _write_pair_table(Path(args.output_pair_csv), pair_counts)

    deltas = [rec["all_paths"] - rec["db_only_paths"] for rec in pair_counts.values()]
    summary = summarize_deltas(deltas)

    summary_path = Path(args.output_summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote summary JSON: %s", summary_path)
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
