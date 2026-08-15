"""Combine, deduplicate, and segregate GWAS pathway datasets.
Provides two subcommands:.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys

import pandas as pd
from indra_perturbseq.preprocessing.gwas_gene_sets import (
    GENES_8,
    GENES_17,
    GENES_420,
    GWAS_GENES,
    URL_RE,
)
from indra_perturbseq.runtime import add_log_level_arg, configure_logging

logger = logging.getLogger(__name__)


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _warn_deprecated_flags(argv: list[str] | None) -> None:
    tokens = list(argv) if argv is not None else sys.argv[1:]
    replacements = {
        "--input-primary": "--primary-csv",
        "--input-secondary": "--secondary-csv",
        "--output-main": "--output-combined",
        "--input-csv": "--input",
    }
    seen = set(tokens)
    for old, new in replacements.items():
        if old in seen:
            logger.warning(
                "Flag '%s' is deprecated and will be removed in a future "
                "release; use '%s' instead.",
                old,
                new,
            )


# ---------------------------------------------------------------------------
# Quality helpers
# ---------------------------------------------------------------------------
def _is_nonempty(x: object) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and pd.isna(x):
        return False
    s = str(x).strip()
    return s != "" and s.lower() not in {"nan", "none"}


def _evidence_ok_strict(text: object) -> bool:
    if not _is_nonempty(text):
        return False
    s = str(text).strip()
    return bool(re.search(r"(^|\n)\s*1\)\s+", s)) or s.startswith("Evidence from:")


def _is_valid_indra_url(url: object) -> bool:
    if not _is_nonempty(url):
        return False
    return bool(URL_RE.fullmatch(str(url).strip()))


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
        return default if pd.isna(v) else v
    except (TypeError, ValueError):
        return default


def _compute_gwas_in_path(row: pd.Series) -> str:
    hits = []
    for col in ("source", "intermediate", "target"):
        val = str(row.get(col, "")).strip()
        if val in GWAS_GENES:
            hits.append(val)
    return ", ".join(sorted(set(hits)))


# ---------------------------------------------------------------------------
# Dedup subcommand
# ---------------------------------------------------------------------------
def _ensure_schema(df: pd.DataFrame, template_cols: list[str]) -> pd.DataFrame:
    """Ensure *df* has at least the columns present in *template_cols*."""
    df = df.copy()
    df = df.drop(columns=["pval_col_used"], errors="ignore")

    for col in template_cols:
        if col not in df.columns:
            df[col] = ""

    if "logfoldchange" in df.columns:
        df["logfoldchange"] = pd.to_numeric(df["logfoldchange"], errors="coerce")

    if "GWAS_genes_in_path" not in df.columns:
        df["GWAS_genes_in_path"] = ""
    missing = ~df["GWAS_genes_in_path"].apply(_is_nonempty)
    if missing.any():
        df.loc[missing, "GWAS_genes_in_path"] = df.loc[missing].apply(
            _compute_gwas_in_path, axis=1,
        )
    return df


def _hop2_strict_ok(row: pd.Series, require_pmids: bool) -> bool:
    if not _is_nonempty(row.get("hop2_hash", "")):
        return False
    if not _evidence_ok_strict(row.get("evidence_text_hop2", "")):
        return False
    if require_pmids and not _is_nonempty(row.get("pmids_hop2", "")):
        return False
    return True


def _row_quality_score(row: pd.Series) -> float:
    url1 = _is_valid_indra_url(row.get("hop1_indra_url", ""))
    url2 = _is_valid_indra_url(row.get("hop2_indra_url", ""))

    ev1_ok = _evidence_ok_strict(row.get("evidence_text_hop1", ""))
    ev2_ok = _evidence_ok_strict(row.get("evidence_text_hop2", ""))

    pm1 = _is_nonempty(row.get("pmids_hop1", ""))
    pm2 = _is_nonempty(row.get("pmids_hop2", ""))

    completeness = sum(1 for c in row.index if _is_nonempty(row.get(c, "")))
    abs_lfc = abs(_safe_float(row.get("logfoldchange", 0.0)))

    score = 0.0
    score += 20.0 * (url1 + url2)
    score += 10.0 * (ev1_ok + ev2_ok)
    score += 4.0 * (pm1 + pm2)
    score += 0.5 * completeness
    score += 1.0 * abs_lfc
    return score


def deduplicate_gwas(
    primary_csv: str,
    secondary_csv: str,
    *,
    require_hop2_pmids: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combine two GWAS pathway CSVs and deduplicate on ``(source, target)``.

    Parameters
    ----------
    primary_csv
        Path to the primary (GWAS-URL-enriched) CSV.
    secondary_csv
        Path to the secondary (one-stop) CSV.
    require_hop2_pmids
        If ``True``, rows without hop-2 PMIDs are dropped.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(combined_deduped, gwas_sources, gwas_targets)``.
    """
    df_a = pd.read_csv(primary_csv, low_memory=False)
    df_b = pd.read_csv(secondary_csv, low_memory=False)

    df_a = df_a.drop(columns=["pval_col_used"], errors="ignore")
    df_b = df_b.drop(columns=["pval_col_used"], errors="ignore")
    a_cols = list(df_a.columns)

    logger.info("Primary CSV: %d rows, %d cols", len(df_a), len(df_a.columns))
    logger.info("Secondary CSV: %d rows, %d cols", len(df_b), len(df_b.columns))

    df_a = _ensure_schema(df_a, a_cols)
    df_b = _ensure_schema(df_b, a_cols)

    combined = pd.concat([df_a, df_b], ignore_index=True)
    before = len(combined)

    combined["_hop2_ok"] = combined.apply(
        lambda r: _hop2_strict_ok(r, require_hop2_pmids), axis=1,
    )
    combined = combined[combined["_hop2_ok"]].copy()
    logger.info(
        "Strict hop2 filter: %d -> %d rows (%.1f%%)",
        before, len(combined), len(combined) / max(before, 1) * 100,
    )

    if combined.empty:
        raise RuntimeError(
            "No rows survived strict hop2 filtering. "
            "Consider disabling --require-hop2-pmids."
        )

    combined["_score"] = combined.apply(_row_quality_score, axis=1)
    combined["_abs_lfc"] = combined["logfoldchange"].apply(
        lambda x: abs(_safe_float(x)),
    )
    combined = combined.sort_values(
        ["_score", "_abs_lfc"], ascending=[False, False], kind="mergesort",
    )
    combined = combined.drop_duplicates(subset=["source", "target"], keep="first")
    combined = combined.drop(columns=["_hop2_ok", "_score", "_abs_lfc"], errors="ignore")

    keep = [c for c in a_cols if c in combined.columns]
    combined = combined[keep].copy()
    logger.info("Deduplicated to %d unique (source, target) pairs", len(combined))

    gwas_src = combined[combined["source"].astype(str).isin(GWAS_GENES)].copy()
    gwas_tgt = combined[combined["target"].astype(str).isin(GWAS_GENES)].copy()
    logger.info("GWAS-as-source: %d rows", len(gwas_src))
    logger.info("GWAS-as-target: %d rows", len(gwas_tgt))

    return combined, gwas_src, gwas_tgt


# ---------------------------------------------------------------------------
# Segregate subcommand
# ---------------------------------------------------------------------------
def _genes_in_path(row: pd.Series, gene_set: set[str]) -> str:
    hits: list[str] = []
    for col in ("source", "intermediate", "target"):
        val = str(row.get(col, "")).strip()
        if val in gene_set:
            hits.append(val)
    return ", ".join(sorted(set(hits)))


def segregate_gwas(
    input_csv: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a GWAS CSV by 8-gene, 17-gene, and residual sets.

    Parameters
    ----------
    input_csv
        Path to the input GWAS pathway CSV.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(df_8gene, df_17gene, df_residual)``.
    """
    df = pd.read_csv(input_csv, low_memory=False)
    logger.info("Loaded %d rows from %s", len(df), input_csv)

    mask_8 = (
        df["source"].isin(GENES_8)
        | df["intermediate"].isin(GENES_8)
        | df["target"].isin(GENES_8)
    )
    mask_17 = (
        df["source"].isin(GENES_17)
        | df["intermediate"].isin(GENES_17)
        | df["target"].isin(GENES_17)
    )
    mask_residual = ~(mask_8 | mask_17)

    df_8 = df[mask_8].copy()
    df_8["Genes_in_path_8gene_set"] = df_8.apply(
        lambda r: _genes_in_path(r, GENES_8), axis=1,
    )
    df_8["Genes_in_path_420gene_set"] = df_8.apply(
        lambda r: _genes_in_path(r, GENES_420), axis=1,
    )

    df_17 = df[mask_17].copy()
    df_17["Genes_in_path_17gene_set"] = df_17.apply(
        lambda r: _genes_in_path(r, GENES_17), axis=1,
    )
    df_17["Genes_in_path_420gene_set"] = df_17.apply(
        lambda r: _genes_in_path(r, GENES_420), axis=1,
    )

    df_res = df[mask_residual].copy()
    df_res["Genes_in_path_420gene_set"] = df_res.apply(
        lambda r: _genes_in_path(r, GENES_420), axis=1,
    )

    logger.info(
        "Segregation: 8-gene=%d, 17-gene=%d, residual=%d",
        len(df_8), len(df_17), len(df_res),
    )
    return df_8, df_17, df_res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    """CLI entry point with ``dedup`` and ``segregate`` subcommands."""
    _warn_deprecated_flags(argv)
    parser = argparse.ArgumentParser(
        description="Combine, deduplicate, and segregate GWAS pathway datasets.",
    )
    add_log_level_arg(parser, default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- dedup --
    dd = sub.add_parser("dedup", help="Combine + strict deduplicate two GWAS CSVs.")
    dd.add_argument(
        "--primary-csv", "--input-primary", required=True,
        help="Primary GWAS CSV (e.g. gwas_endothelial_paths_url_enriched.csv).",
    )
    dd.add_argument(
        "--secondary-csv", "--input-secondary", required=True,
        help="Secondary CSV (e.g. one-stop 2hop output).",
    )
    dd.add_argument(
        "--output-combined",
        "--output-main",
        required=True,
        help="Combined deduped output.",
    )
    dd.add_argument("--output-gwas-sources", required=True, help="GWAS-as-source subset.")
    dd.add_argument("--output-gwas-targets", required=True, help="GWAS-as-target subset.")
    dd.add_argument(
        "--require-hop2-pmids", action="store_true",
        help="Also require pmids_hop2 to be non-empty.",
    )

    # -- segregate --
    seg = sub.add_parser("segregate", help="Split GWAS CSV by gene groups.")
    seg.add_argument("--input", "--input-csv", required=True, help="Input GWAS CSV.")
    seg.add_argument("--output-8gene", required=True, help="8-gene subset output.")
    seg.add_argument("--output-17gene", required=True, help="17-gene subset output.")
    seg.add_argument("--output-residual", required=True, help="Residual subset output.")

    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    if args.command == "dedup":
        combined, gwas_src, gwas_tgt = deduplicate_gwas(
            primary_csv=args.primary_csv,
            secondary_csv=args.secondary_csv,
            require_hop2_pmids=args.require_hop2_pmids,
        )
        _ensure_parent_dir(args.output_combined)
        _ensure_parent_dir(args.output_gwas_sources)
        _ensure_parent_dir(args.output_gwas_targets)
        combined.to_csv(args.output_combined, index=False)
        gwas_src.to_csv(args.output_gwas_sources, index=False)
        gwas_tgt.to_csv(args.output_gwas_targets, index=False)
        logger.info("Saved dedup outputs")

    elif args.command == "segregate":
        df_8, df_17, df_res = segregate_gwas(args.input)
        _ensure_parent_dir(args.output_8gene)
        _ensure_parent_dir(args.output_17gene)
        _ensure_parent_dir(args.output_residual)
        df_8.to_csv(args.output_8gene, index=False)
        df_17.to_csv(args.output_17gene, index=False)
        df_res.to_csv(args.output_residual, index=False)
        logger.info("Saved segregation outputs")


if __name__ == "__main__":
    main()
