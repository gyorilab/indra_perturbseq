"""DEG table helpers used by the pipeline."""

from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

_FDR_CANDIDATES = ("pvals_adj", "padj", "qval", "fdr", "p_adj")
_P_CANDIDATES = ("pvals", "pval", "p_value", "p_val")


def pick_sig_column(df: pd.DataFrame, prefer_fdr: bool = False) -> str:
    """Choose the p-value or FDR column to use for significance."""
    fdr_col = next((c for c in _FDR_CANDIDATES if c in df.columns), None)
    p_col = next((c for c in _P_CANDIDATES if c in df.columns), None)
    if prefer_fdr and fdr_col:
        return fdr_col
    if p_col:
        return p_col
    if fdr_col:
        return fdr_col
    raise ValueError(
        f"DEG file missing p-value columns. Columns: {df.columns.tolist()}"
    )


def deg_path_for_source(deg_dir: str, source: str) -> str:
    """Return the expected DEG CSV path for a source gene."""
    return os.path.join(deg_dir, f"{source}_vs_control.csv")
