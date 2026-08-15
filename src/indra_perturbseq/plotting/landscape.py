"""Fold-change vs p-value landscape scatter plots across hop distances.
This module provides plotting utilities and CLI entry points for analysis outputs.
"""

from __future__ import annotations

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HOP_COLORS: dict[str, str] = {
    "1-hop": "#2563EB",
    "2-hop": "#059669",
    "3-hop": "#D97706",
    "3+": "#DC2626",
}
HOP_ORDER: list[str] = ["1-hop", "2-hop", "3-hop", "3+"]

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "savefig.dpi": 300,
})


def _load_pairs(path: str) -> set[tuple[str, str]]:
    """Load source-target pairs from a hop CSV.

    Parameters
    ----------
    path :
        CSV with ``source`` and ``target`` columns.

    Returns
    -------
    set[tuple[str, str]]
        Unique (source, target) pairs, uppercased and stripped.
    """
    df = pd.read_csv(path, usecols=["source", "target"]).astype(str)
    df["source"] = df["source"].str.strip().str.upper()
    df["target"] = df["target"].str.strip().str.upper()
    return set(map(tuple, df.values))


def _load_deg_pairs(
    sources: set[str],
    deg_dir: str,
    pval_cutoff: float = 0.05,
) -> pd.DataFrame:
    """Build a DataFrame of DEG pairs with log-FC and p-values.

    Parameters
    ----------
    sources :
        Source gene symbols to scan.
    deg_dir :
        Directory containing ``<GENE>_vs_control.csv`` files.
    pval_cutoff :
        Maximum raw p-value to include.

    Returns
    -------
    pd.DataFrame
        Columns: ``source``, ``target``, ``logfc``, ``pval``, ``neglogp``.
    """
    rows: list[dict[str, object]] = []
    for src in sorted(sources):
        f = os.path.join(deg_dir, f"{src}_vs_control.csv")
        if not os.path.exists(f):
            continue
        deg = pd.read_csv(f, usecols=["names", "logfoldchanges", "pvals"]).dropna()
        deg = deg[deg["pvals"] < pval_cutoff]
        for _, r in deg.iterrows():
            rows.append({
                "source": src.upper(),
                "target": str(r["names"]).strip().upper(),
                "logfc": float(r["logfoldchanges"]),
                "pval": float(r["pvals"]),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.loc[df.groupby(["source", "target"])["pval"].idxmin()].copy()
    df["neglogp"] = -np.log10(df["pval"].clip(lower=1e-300))
    return df


def _assign_hop_labels(
    df: pd.DataFrame,
    h1: set[tuple[str, str]],
    h2_only: set[tuple[str, str]],
    h3_only: set[tuple[str, str]],
) -> pd.DataFrame:
    """Add a ``hop`` column based on pair membership in hop sets."""

    def _label(row: pd.Series) -> str:
        pair = (row["source"], row["target"])
        if pair in h1:
            return "1-hop"
        if pair in h2_only:
            return "2-hop"
        if pair in h3_only:
            return "3-hop"
        return "3+"

    df["hop"] = df.apply(_label, axis=1)
    return df


def _trim_outliers(
    df: pd.DataFrame,
    trim_pct: float,
    groups: list[str] | None = None,
) -> pd.DataFrame:
    """Remove the top *trim_pct*% of ``neglogp`` from selected hop groups.

    Parameters
    ----------
    df :
        DataFrame with ``hop`` and ``neglogp`` columns.
    trim_pct :
        Percentage of most-significant values to remove per group.
    groups :
        Hop labels to trim. Defaults to ``["2-hop", "3-hop", "3+"]``.
    """
    if groups is None:
        groups = ["2-hop", "3-hop", "3+"]

    cutoff_q = 1.0 - trim_pct / 100.0
    parts = [df[~df["hop"].isin(groups)].copy()]

    for grp in groups:
        sub = df[df["hop"] == grp].copy()
        if sub.empty:
            continue
        threshold = sub["neglogp"].quantile(cutoff_q)
        before = len(sub)
        sub = sub[sub["neglogp"] <= threshold]
        logger.info(
            "%s: removed %d / %d rows (top %.0f%% outliers)",
            grp, before - len(sub), before, trim_pct,
        )
        parts.append(sub)

    return pd.concat(parts, ignore_index=True)


def plot_fc_pval_landscape(
    df: pd.DataFrame,
    trim_pct: float,
    out_path: str,
) -> None:
    """Create a 2x2 hexbin + scatter plot of FC vs -log10(p) by hop.

    Parameters
    ----------
    df :
        DataFrame with ``logfc``, ``neglogp``, and ``hop`` columns.
    trim_pct :
        Trim percentage used (for annotation only).
    out_path :
        Output PNG path.
    """
    x, y = df["logfc"].values, df["neglogp"].values
    xmin, xmax = np.percentile(x, [0.5, 99.5])
    ymax = np.percentile(y, 99.5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.flatten()

    for i, hop in enumerate(HOP_ORDER):
        ax = axes_flat[i]
        hb = ax.hexbin(x, y, gridsize=50, cmap="YlOrRd", mincnt=1, alpha=0.85)
        sub = df[df["hop"] == hop]
        ax.scatter(
            sub["logfc"], sub["neglogp"],
            s=20, c=HOP_COLORS[hop], alpha=0.8,
            edgecolors="black", linewidths=0.3,
            label=f"{hop} (n={len(sub):,})",
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Log fold-change")
        ax.set_ylabel(r"$-\log_{10}$(p-value)")
        ax.set_title(f"Highlighting {hop}")
        ax.axvline(0, ls="--", lw=1, c="gray")
        ax.axhline(-np.log10(0.05), ls="--", lw=1, c="gray")
        ax.legend(loc="upper right", fontsize=8)
        plt.colorbar(hb, ax=ax, label="Count")

    fig.suptitle(
        f"FC vs p-value landscape (top {trim_pct:.0f}% outliers removed from 2/3-hop & 3+)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Landscape plot saved: %s", out_path)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for fold-change / p-value landscape plots."""
    ap = argparse.ArgumentParser(
        description="Plot fold-change / p-value landscape by hop distance.",
    )
    ap.add_argument("--hop1", required=True, help="1-hop network export CSV")
    ap.add_argument("--hop2", required=True, help="2-hop network export CSV")
    ap.add_argument("--hop3", required=True, help="3-hop network export CSV")
    ap.add_argument("--target-validation", required=True, help="target_validation_expanded.csv")
    ap.add_argument("--deg-dir", required=True, help="DEG results per-gene directory")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument(
        "--trim-top-pct", type=float, default=10.0,
        help="Remove top N%% of smallest p-values per group (default: 10)",
    )
    ap.add_argument(
        "--exclude-sources", nargs="*", default=["TP53", "CDKN1A"],
        help="Source genes to exclude (default: TP53 CDKN1A)",
    )
    args = ap.parse_args(argv)

    h1 = _load_pairs(args.hop1)
    h2 = _load_pairs(args.hop2)
    h3 = _load_pairs(args.hop3)
    h2_only = h2 - h1
    h3_only = h3 - h1 - h2

    tv = pd.read_csv(args.target_validation, usecols=["Gene", "analysis_flag"])
    sources = set(
        tv.loc[tv["analysis_flag"] == "Use_for_analysis", "Gene"]
        .astype(str).str.strip()
    )
    sources -= set(args.exclude_sources or [])

    df = _load_deg_pairs(sources, args.deg_dir)
    if df.empty:
        logger.warning("No DEG pairs found; nothing to plot.")
        return

    df = _assign_hop_labels(df, h1, h2_only, h3_only)
    df = _trim_outliers(df, args.trim_top_pct)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_fc_pval_landscape(df, args.trim_top_pct, args.out)


if __name__ == "__main__":
    main()
