"""DEG output normalization."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

RAW_SINGLE_CELL_KINDS = {"raw_perturbseq", "raw_scrna"}
RAW_BULK_KINDS = {"raw_bulk_rna"}
RAW_INPUT_KINDS = RAW_SINGLE_CELL_KINDS | RAW_BULK_KINDS

CANONICAL_DEG_COLUMNS = ("names", "logfoldchanges", "pvals", "pvals_adj")


def safe_source_name(label: object) -> str:
    """Return a filesystem-safe source label for ``<SOURCE>_vs_control.csv``."""
    text = str(label).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.-]", "_", text)
    return text.strip("._") or "source"


def empty_deg_frame() -> pd.DataFrame:
    """Return an empty DEG table with pipeline columns."""
    return pd.DataFrame({col: [] for col in CANONICAL_DEG_COLUMNS})


def write_deg_csv(df: pd.DataFrame, output_dir: str | Path, source: str) -> Path:
    """Write one DEG CSV and return its path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{safe_source_name(source)}_vs_control.csv"
    df.to_csv(path, index=False)
    return path


def sources_from_deg_dir(deg_dir: str | Path) -> list[str]:
    """Infer source names from ``<SOURCE>_vs_control.csv`` files."""
    path = Path(deg_dir)
    sources = []
    for csv_path in sorted(path.glob("*_vs_control.csv")):
        sources.append(csv_path.name.removesuffix("_vs_control.csv"))
    return sources


def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def standardize_scanpy_deg(
    df: pd.DataFrame,
    *,
    source: str | None = None,
) -> pd.DataFrame:
    """Map Scanpy result columns to the pipeline DEG schema."""
    out = df.copy()
    rename = {
        "name": "names",
        "gene": "names",
        "gene_symbol": "names",
        "logfoldchange": "logfoldchanges",
        "log2FoldChange": "logfoldchanges",
        "pval": "pvals",
        "pvalue": "pvals",
        "padj": "pvals_adj",
        "qval": "pvals_adj",
        "fdr": "pvals_adj",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    if "names" not in out.columns:
        raise ValueError(
            "Scanpy DEG output missing gene-name column: "
            f"{df.columns.tolist()}"
        )
    if "pvals" not in out.columns:
        raise ValueError(
            "Scanpy DEG output missing p-value column: "
            f"{df.columns.tolist()}"
        )
    if "logfoldchanges" not in out.columns:
        out["logfoldchanges"] = pd.NA
    if "pvals_adj" not in out.columns:
        out["pvals_adj"] = pd.NA
    if source is not None and "source" not in out.columns:
        out["source"] = source
    return _order_deg_columns(out)


def standardize_pydeseq2_deg(
    df: pd.DataFrame,
    *,
    source: str | None = None,
) -> pd.DataFrame:
    """Map PyDESeq2 result columns to the pipeline DEG schema."""
    out = df.copy()
    if "target" not in out.columns and "names" not in out.columns:
        out = out.reset_index()
        out = out.rename(columns={out.columns[0]: "target"})
    rename = {
        "target": "names",
        "gene": "names",
        "gene_symbol": "names",
        "log2FoldChange": "logfoldchanges",
        "pvalue": "pvals",
        "padj": "pvals_adj",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    if "names" not in out.columns:
        raise ValueError(
            "PyDESeq2 output missing gene-name column: "
            f"{df.columns.tolist()}"
        )
    if "pvals" not in out.columns:
        raise ValueError(
            "PyDESeq2 output missing p-value column: "
            f"{df.columns.tolist()}"
        )
    if "logfoldchanges" not in out.columns:
        out["logfoldchanges"] = pd.NA
    if "pvals_adj" not in out.columns:
        out["pvals_adj"] = pd.NA
    if source is not None and "source" not in out.columns:
        out["source"] = source
    if "deg_method" not in out.columns:
        out["deg_method"] = "PyDESeq2"
    return _order_deg_columns(out)


def _order_deg_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CANONICAL_DEG_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    extras = [col for col in out.columns if col not in CANONICAL_DEG_COLUMNS]
    return out[list(CANONICAL_DEG_COLUMNS) + extras]
