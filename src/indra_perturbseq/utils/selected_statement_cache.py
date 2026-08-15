"""Selected edge-statement cache helpers.

This module stores the chosen statement metadata for a directed gene-gene edge
under a specific ``require_incdec`` selection mode. It is designed as a small
bridge between extraction and downstream enrichment/overlap analyses.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from threading import Lock

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


def _norm_gene(value: object) -> str:
    return str(value or "").strip().upper()


def _norm_source_name(name: object) -> str:
    s = str(name or "").strip().lower()
    if ":" in s:
        s = s.split(":", 1)[0]
    return ALIASES.get(s, s)


def _normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value or "").strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _normalize_source_counts(source_counts: object) -> dict[str, int]:
    if isinstance(source_counts, dict):
        out: dict[str, int] = {}
        for key, value in source_counts.items():
            try:
                out[str(key)] = int(value)
            except Exception:
                continue
        return out
    if source_counts is None:
        return {}
    s = str(source_counts).strip()
    if not s:
        return {}
    try:
        parsed = json.loads(s)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return _normalize_source_counts(parsed)


def stmt_has_db_from_source_counts(source_counts: object) -> bool:
    sc = _normalize_source_counts(source_counts)
    if not sc:
        return False
    return any(_norm_source_name(k) in DB_SOURCES for k in sc.keys())


def cache_key(source: object, target: object, require_incdec: object) -> tuple[str, str, bool]:
    return (_norm_gene(source), _norm_gene(target), bool(require_incdec))


def _stmt_rank(stmt_hash: object, belief: object, evidence_count: object) -> tuple[float, int, int]:
    try:
        bf = float(belief) if belief is not None else float("-inf")
    except Exception:
        bf = float("-inf")
    if not math.isfinite(bf):
        bf = float("-inf")
    try:
        ev = int(evidence_count) if evidence_count is not None else 0
    except Exception:
        ev = 0
    try:
        sh = int(stmt_hash) if stmt_hash is not None and str(stmt_hash).strip() else -1
    except Exception:
        sh = -1
    return (bf, ev, sh)


class SelectedStatementCache:
    """Thread-safe in-memory selected-edge cache with CSV persistence."""

    FIELDNAMES = [
        "source",
        "target",
        "require_incdec",
        "stmt_hash",
        "stmt_type",
        "belief",
        "evidence_count",
        "has_db",
        "source_counts",
    ]

    def __init__(self) -> None:
        self._lock = Lock()
        self._records: dict[tuple[str, str, bool], dict[str, object]] = {}

    def __len__(self) -> int:
        return len(self._records)

    def get(self, source: object, target: object, require_incdec: object) -> dict[str, object] | None:
        return self._records.get(cache_key(source, target, require_incdec))

    def record(
        self,
        source: object,
        target: object,
        require_incdec: bool,
        stmt: dict | None,
    ) -> None:
        if not stmt:
            return
        self.record_fields(
            source=source,
            target=target,
            require_incdec=require_incdec,
            stmt_hash=stmt.get("stmt_hash"),
            stmt_type=stmt.get("stmt_type"),
            belief=stmt.get("belief"),
            evidence_count=stmt.get("evidence_count"),
            source_counts=stmt.get("source_counts"),
            has_db=stmt_has_db_from_source_counts(stmt.get("source_counts")),
        )

    def record_fields(
        self,
        *,
        source: object,
        target: object,
        require_incdec: bool,
        stmt_hash: object,
        stmt_type: object,
        belief: object,
        evidence_count: object,
        source_counts: object = None,
        has_db: bool | None = None,
    ) -> None:
        key = cache_key(source, target, require_incdec)
        if not key[0] or not key[1]:
            return
        normalized_source_counts = _normalize_source_counts(source_counts)
        record = {
            "source": key[0],
            "target": key[1],
            "require_incdec": bool(require_incdec),
            "stmt_hash": stmt_hash,
            "stmt_type": stmt_type,
            "belief": belief,
            "evidence_count": evidence_count,
            "has_db": stmt_has_db_from_source_counts(normalized_source_counts) if has_db is None else bool(has_db),
            "source_counts": normalized_source_counts,
        }
        with self._lock:
            prev = self._records.get(key)
            if prev is None or _stmt_rank(
                record["stmt_hash"],
                record["belief"],
                record["evidence_count"],
            ) >= _stmt_rank(
                prev.get("stmt_hash"),
                prev.get("belief"),
                prev.get("evidence_count"),
            ):
                self._records[key] = record

    def write_csv(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(self._records.values(), key=lambda r: (r["source"], r["target"], r["require_incdec"]))
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "source": row["source"],
                        "target": row["target"],
                        "require_incdec": "1" if row["require_incdec"] else "0",
                        "stmt_hash": "" if row["stmt_hash"] is None else row["stmt_hash"],
                        "stmt_type": row.get("stmt_type", "") or "",
                        "belief": "" if row["belief"] is None else row["belief"],
                        "evidence_count": "" if row["evidence_count"] is None else row["evidence_count"],
                        "has_db": "1" if row.get("has_db") else "0",
                        "source_counts": json.dumps(row.get("source_counts") or {}, sort_keys=True),
                    }
                )

    @classmethod
    def load_csv(cls, path: str | Path) -> "SelectedStatementCache":
        cache = cls()
        src = Path(path)
        with src.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = cache_key(row.get("source"), row.get("target"), _normalize_bool(row.get("require_incdec")))
                if not key[0] or not key[1]:
                    continue
                cache._records[key] = {
                    "source": key[0],
                    "target": key[1],
                    "require_incdec": key[2],
                    "stmt_hash": str(row.get("stmt_hash", "")).strip(),
                    "stmt_type": str(row.get("stmt_type", "")).strip(),
                    "belief": str(row.get("belief", "")).strip(),
                    "evidence_count": str(row.get("evidence_count", "")).strip(),
                    "has_db": _normalize_bool(row.get("has_db")),
                    "source_counts": _normalize_source_counts(row.get("source_counts")),
                }
        return cache


def load_selected_statement_cache(path: str | Path | None) -> SelectedStatementCache | None:
    if not path:
        return None
    src = Path(path)
    if not src.exists():
        return None
    return SelectedStatementCache.load_csv(src)
