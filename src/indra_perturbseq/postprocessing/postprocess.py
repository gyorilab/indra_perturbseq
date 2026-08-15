"""Post-process one-stop 2-hop output.

1. Filter MeSH terms to a reference set.
2. Enforce (source, target) uniqueness with quality-based row selection.
"""

from __future__ import annotations

import argparse
import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


def _is_nonempty(x) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and pd.isna(x):
        return False
    s = str(x).strip()
    return s != "" and s.lower() not in {"nan", "none"}


def _nonempty_count(series: pd.Series) -> int:
    return series.apply(_is_nonempty).sum()


def _looks_structured(text: str) -> bool:
    if not _is_nonempty(text):
        return False
    return bool(re.search(r"(^|\n)\s*1\)\s+", str(text).strip()))


def _is_placeholder(text: str) -> bool:
    if not _is_nonempty(text):
        return True
    s = str(text).strip()
    return s.startswith(("No evidence found", "Evidence from:",
                         "Database evidence only", "Error"))


def _valid_indra_url(url: str) -> bool:
    if not _is_nonempty(url):
        return False
    return bool(re.fullmatch(
        r"https://db\.indra\.bio/statements/from_hash/-?\d+\?format=html",
        str(url).strip(),
    ))


def _safe_float(x, default=None):
    try:
        v = float(x)
        return default if pd.isna(v) else v
    except Exception:
        return default


def _safe_int(x, default=0):
    try:
        return default if pd.isna(x) else int(float(x))
    except Exception:
        return default


def _load_valid_mesh_ids(reference_csv: str) -> set[str]:
    ref = pd.read_csv(
        reference_csv, encoding="utf-8-sig",
        on_bad_lines="skip", low_memory=False,
    )
    if "mesh_id" not in ref.columns:
        raise ValueError(f"Reference CSV must have 'mesh_id'. Columns: {ref.columns.tolist()}")
    ids = (ref["mesh_id"].astype(str)
           .str.replace(r"\s+", "", regex=True)
           .str.replace(r"[^A-Za-z0-9]", "", regex=True))
    return set(ids[ids.str.startswith("D")])


def _filter_mesh_cell(text: str, valid_ids: set[str]) -> str:
    if not _is_nonempty(text):
        return ""
    pairs = re.findall(r"[^,]+?\(D\d{5,10}\)", str(text))
    kept = []
    for p in pairs:
        m = re.search(r"\((D\d{5,10})\)", p)
        if m and m.group(1) in valid_ids:
            kept.append(p.strip())
    return ", ".join(kept)


def apply_mesh_filter(df: pd.DataFrame, reference_csv: str) -> pd.DataFrame:
    """Filter MeSH terms to those in the reference set."""
    df = df.copy()
    valid = _load_valid_mesh_ids(reference_csv)
    logger.info("Valid MeSH IDs from reference: %d", len(valid))

    for col in ("Annotated MeSH terms hop1", "Annotated MeSH terms hop2"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].apply(lambda x: _filter_mesh_cell(x, valid))
    return df


def _row_quality_score(row: pd.Series) -> float:
    score = 0.0
    score += 10.0 * sum(
        _valid_indra_url(row.get(c, ""))
        for c in ("hop1_indra_url", "hop2_indra_url")
    )
    score += 5.0 * sum(
        _is_nonempty(row.get(c, ""))
        for c in ("hop1_hash", "hop2_hash")
    )
    score += 6.0 * sum(
        _looks_structured(row.get(c, ""))
        for c in ("evidence_text_hop1", "evidence_text_hop2")
    )
    score -= 6.0 * sum(
        _is_placeholder(row.get(c, ""))
        for c in ("evidence_text_hop1", "evidence_text_hop2")
    )
    score += 3.0 * sum(
        _is_nonempty(row.get(c, ""))
        for c in ("pmids_hop1", "pmids_hop2")
    )
    required = [
        "source", "intermediate", "target",
        "stmt_type_1", "stmt_type_2",
        "belief_1", "belief_2",
        "evidence_1", "evidence_2",
        "logfoldchange", "pval",
        "Annotated MeSH terms hop1", "Annotated MeSH terms hop2",
    ]
    score += sum(_is_nonempty(row.get(f, "")) for f in required)
    ev = _safe_int(row.get("evidence_1", 0)) + _safe_int(row.get("evidence_2", 0))
    belief = (_safe_float(row.get("belief_1"), 0) or 0) + (_safe_float(row.get("belief_2"), 0) or 0)
    lfc = abs(_safe_float(row.get("logfoldchange"), 0) or 0)
    score += 0.01 * ev + 0.5 * belief + 0.1 * lfc
    return score


def deduplicate_by_source_target(df: pd.DataFrame) -> pd.DataFrame:
    """Keep best row per (source, target) by quality score."""
    df = df.copy()
    df["_score"] = df.apply(_row_quality_score, axis=1)
    df.sort_values("_score", ascending=False, kind="mergesort", inplace=True)
    before = len(df)
    df = df.drop_duplicates(subset=["source", "target"], keep="first")
    df.drop(columns=["_score"], inplace=True, errors="ignore")
    logger.info("Dedup: %d -> %d rows (-%d)", before, len(df), before - len(df))
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Post-process one-stop 2-hop output.",
    )
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--mesh-reference", required=True)
    ap.add_argument("--output-mesh-filtered", required=True)
    ap.add_argument("--output-unique", required=True)
    args = ap.parse_args(argv)

    df = pd.read_csv(args.input_csv, low_memory=False)
    logger.info("Loaded %d rows from %s", len(df), args.input_csv)

    df_mesh = apply_mesh_filter(df, args.mesh_reference)
    df_mesh.to_csv(args.output_mesh_filtered, index=False)
    logger.info("MeSH-filtered -> %s", args.output_mesh_filtered)

    df_unique = deduplicate_by_source_target(df_mesh)
    df_unique.to_csv(args.output_unique, index=False)
    logger.info("Unique (source,target) -> %s", args.output_unique)


if __name__ == "__main__":
    main()
