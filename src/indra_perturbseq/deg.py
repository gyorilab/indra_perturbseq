"""DEG (differentially expressed gene) file helpers."""

from __future__ import annotations

import logging
import os

import pandas as pd

from indra_perturbseq.hgnc import normalize_hgnc_symbol

logger = logging.getLogger(__name__)

_FDR_CANDIDATES = ("pvals_adj", "padj", "qval", "fdr", "p_adj")
_P_CANDIDATES = ("pvals", "pval", "p_value", "p_val")


def pick_sig_column(df: pd.DataFrame, prefer_fdr: bool = False) -> str:
    """Return the name of the best p-value / FDR column in *df*.

    Parameters
    ----------
    df :
        DEG DataFrame.
    prefer_fdr :
        If ``True`` and an FDR column exists, prefer it over raw p-values.

    Raises
    ------
    ValueError
        If no suitable column is found.
    """
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


def deg_path_for_source(de_dir: str, source: str) -> str:
    """Return the expected DEG CSV path for a source gene."""
    return os.path.join(de_dir, f"{source}_vs_control.csv")


def load_deg_targets(
    de_dir: str,
    raw_gene: str,
    p_threshold: float,
    prefer_fdr: bool = False,
) -> tuple[list[str], dict[str, dict], str | None]:
    """Load significant DEG targets for a source gene.

    Returns
    -------
    targets :
        Normalized HGNC symbols of significant targets.
    deg_map :
        ``{target: {"logfoldchange": ..., "pval": ...}}`` with the
        most-significant p-value per target.
    error :
        ``None`` on success, otherwise a short error message.
    """
    path = deg_path_for_source(de_dir, raw_gene)
    if not os.path.exists(path):
        return [], {}, f"missing DEG file: {path}"

    df = pd.read_csv(path, low_memory=False)
    if "names" not in df.columns:
        return [], {}, "DEG missing 'names' column"

    sig_col = pick_sig_column(df, prefer_fdr=prefer_fdr)
    df[sig_col] = pd.to_numeric(df[sig_col], errors="coerce")
    df = df[df[sig_col] < p_threshold].copy()
    if df.empty:
        return [], {}, "no significant targets"

    if "logfoldchanges" in df.columns:
        df["logfoldchanges"] = pd.to_numeric(df["logfoldchanges"], errors="coerce")
    else:
        df["logfoldchanges"] = pd.NA

    df["names"] = df["names"].astype(str)
    targets = [t for t in
               (normalize_hgnc_symbol(x) for x in df["names"].dropna().unique())
               if t]

    deg_map: dict[str, dict] = {}
    for _, row in df.iterrows():
        t = normalize_hgnc_symbol(row["names"])
        if not t:
            continue
        p = row.get(sig_col)
        lfc = row.get("logfoldchanges")
        if t not in deg_map or (p is not None and p < deg_map[t]["pval"]):
            deg_map[t] = {"logfoldchange": lfc, "pval": p}

    return targets, deg_map, None


def load_deg_universe(
    de_dir: str,
    raw_gene: str,
    p_threshold: float,
    prefer_fdr: bool = False,
) -> tuple[set[str] | None, set[str] | None, dict | None, str | None]:
    """Load the full DEG universe and positive subset for *raw_gene*.

    Returns ``(all_targets, pos_targets, stats_map, error)``.
    """
    path = deg_path_for_source(de_dir, raw_gene)
    if not os.path.exists(path):
        return None, None, None, f"missing DEG file: {path}"

    df = pd.read_csv(path, low_memory=False)
    if "names" not in df.columns:
        return None, None, None, "DEG missing 'names' column"

    sig_col = pick_sig_column(df, prefer_fdr=prefer_fdr)
    df[sig_col] = pd.to_numeric(df[sig_col], errors="coerce")

    if "logfoldchanges" in df.columns:
        df["logfoldchanges"] = pd.to_numeric(df["logfoldchanges"], errors="coerce")
    else:
        df["logfoldchanges"] = pd.NA

    stats_map: dict[str, dict] = {}
    all_targets: set[str] = set()
    pos_targets: set[str] = set()

    for _, row in df.iterrows():
        t = normalize_hgnc_symbol(row["names"])
        if not t:
            continue
        p = row.get(sig_col)
        lfc = row.get("logfoldchanges")
        all_targets.add(t)
        if t not in stats_map or (p is not None and (
                stats_map[t]["pval"] is None or p < stats_map[t]["pval"])):
            stats_map[t] = {"logfoldchange": lfc, "pval": p}
        if p is not None and p < p_threshold:
            pos_targets.add(t)

    return all_targets, pos_targets, stats_map, None


def build_pvalue_map(deg_csv: str, source: str) -> dict[str, float]:
    """Build ``{target: best_pval}`` from a DEG CSV, excluding *source*."""
    df = pd.read_csv(deg_csv, low_memory=False)
    if "names" not in df.columns:
        raise ValueError(f"{deg_csv} missing 'names' column")
    pcol = pick_sig_column(df)
    df[pcol] = pd.to_numeric(df[pcol], errors="coerce")

    pmap: dict[str, float] = {}
    for _, row in df.iterrows():
        p = row[pcol]
        if pd.isna(p):
            continue
        t = normalize_hgnc_symbol(row["names"])
        if not t or t == source:
            continue
        p = float(p)
        if t not in pmap or p < pmap[t]:
            pmap[t] = p
    return pmap
