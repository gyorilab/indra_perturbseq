"""Fold-change landscape analysis binned by hop distance.

Produces stacked and per-hop bar charts showing what percentage of each
fold-change bin is explained by 1-hop, 2-hop, and 3-hop pathways.
"""

from __future__ import annotations

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HOP_LABELS = ("1-hop", "2-hop", "3-hop")
HOP_COLORS = ("skyblue", "lightcoral", "lightgreen")

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "savefig.dpi": 300,
})


def bin_fc_analysis(
    hop_dfs: dict[str, pd.DataFrame],
    n_bins: int = 10,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Bin fold-change values and compute per-hop percentages.

    Parameters
    ----------
    hop_dfs :
        Mapping from hop label (e.g. ``"1-hop"``) to its DataFrame,
        each containing a ``logfoldchange`` column.
    n_bins :
        Number of equal-width bins.

    Returns
    -------
    results :
        DataFrame with bin ranges, counts, and percentages per hop.
    bin_edges :
        Array of bin edge values.
    """
    all_fc = np.concatenate([
        df["logfoldchange"].dropna().values for df in hop_dfs.values()
    ])
    _, bin_edges = pd.cut(all_fc, bins=n_bins, retbins=True)

    rows: list[dict[str, object]] = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        counts = {}
        for label, df in hop_dfs.items():
            fc = df["logfoldchange"]
            counts[label] = int(((fc >= lo) & (fc < hi)).sum())

        total = sum(counts.values())
        row: dict[str, object] = {
            "bin": i + 1,
            "fc_range": f"[{lo:.2f}, {hi:.2f}]",
            "total": total,
        }
        for label in hop_dfs:
            pct = 100.0 * counts[label] / total if total > 0 else 0.0
            row[f"{label}_count"] = counts[label]
            row[f"{label}_pct"] = pct
        rows.append(row)

    return pd.DataFrame(rows), bin_edges


def plot_stacked_bars(
    results: pd.DataFrame,
    bin_edges: np.ndarray,
    out_path: str,
    hop_labels: tuple[str, ...] = HOP_LABELS,
) -> None:
    """Plot a stacked bar chart of hop-type percentages per FC bin.

    Parameters
    ----------
    results :
        Output of :func:`bin_fc_analysis`.
    bin_edges :
        Bin edge values.
    out_path :
        Output PNG path.
    hop_labels :
        Ordered hop labels matching columns in *results*.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(results))
    bottom = np.zeros(len(results))

    for label, color in zip(hop_labels, HOP_COLORS):
        col = f"{label}_pct"
        vals = results[col].values
        ax.bar(x, vals, bottom=bottom, label=label, alpha=0.7, color=color)
        bottom += vals

    bin_labels = [
        f"Bin {i + 1}\n[{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f}]"
        for i in range(len(bin_edges) - 1)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.set_xlabel("Fold-change bins")
    ax.set_ylabel("Percentage")
    ax.set_title("Percentage of each FC bin explained by hop distance")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Stacked bar chart saved: %s", out_path)


def plot_individual_bars(
    results: pd.DataFrame,
    bin_edges: np.ndarray,
    out_path: str,
    hop_labels: tuple[str, ...] = HOP_LABELS,
) -> None:
    """Plot separate bar charts for each hop type.

    Parameters
    ----------
    results :
        Output of :func:`bin_fc_analysis`.
    bin_edges :
        Bin edge values.
    out_path :
        Output PNG path.
    hop_labels :
        Ordered hop labels matching columns in *results*.
    """
    n_hops = len(hop_labels)
    fig, axes = plt.subplots(1, n_hops, figsize=(7 * n_hops, 6))
    if n_hops == 1:
        axes = [axes]

    x = np.arange(len(results))
    bin_labels = [
        f"Bin {i + 1}\n[{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f}]"
        for i in range(len(bin_edges) - 1)
    ]

    for i, (label, color) in enumerate(zip(hop_labels, HOP_COLORS)):
        ax = axes[i]
        pct_col = f"{label}_pct"
        cnt_col = f"{label}_count"
        pcts = results[pct_col].values
        counts = results[cnt_col].values

        ax.bar(x, pcts, color=color, alpha=0.7)
        ax.set_xlabel("Fold-change bins")
        ax.set_ylabel("Percentage of bin")
        ax.set_title(f"{label}: percentage of each FC bin")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=8)

        max_h = max(pcts) if max(pcts) > 0 else 1.0
        for j, (pct, cnt) in enumerate(zip(pcts, counts)):
            if pct > 0:
                ax.text(j, pct + max_h * 0.02, str(cnt),
                        ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max_h * 1.15)

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.15, top=0.9, wspace=0.35)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Individual bar charts saved: %s", out_path)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for fold-change landscape analysis."""
    ap = argparse.ArgumentParser(
        description="Fold-change landscape binned analysis by hop distance.",
    )
    ap.add_argument("--hop1-csv", required=True, help="1-hop results CSV")
    ap.add_argument("--hop2-csv", required=True, help="2-hop results CSV")
    ap.add_argument("--hop3-csv", required=True, help="3-hop results CSV")
    ap.add_argument("--n-bins", type=int, default=10,
                    help="Number of FC bins (default: 10)")
    ap.add_argument("--output-stacked", default="fc_bin_combined.png",
                    help="Output path for stacked bar chart")
    ap.add_argument("--output-individual", default="fc_bin_individual.png",
                    help="Output path for individual bar charts")
    ap.add_argument("--output-csv", default="fc_bin_results.csv",
                    help="Output path for results CSV")
    args = ap.parse_args(argv)

    hop_dfs: dict[str, pd.DataFrame] = {}
    for label, path in [("1-hop", args.hop1_csv), ("2-hop", args.hop2_csv),
                        ("3-hop", args.hop3_csv)]:
        df = pd.read_csv(path, low_memory=False)
        df["logfoldchange"] = pd.to_numeric(df["logfoldchange"], errors="coerce")
        hop_dfs[label] = df
        logger.info("Loaded %s: %d rows from %s", label, len(df), path)

    results, bin_edges = bin_fc_analysis(hop_dfs, n_bins=args.n_bins)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    results.to_csv(args.output_csv, index=False)
    logger.info("Results CSV saved: %s", args.output_csv)

    plot_stacked_bars(results, bin_edges, args.output_stacked)
    plot_individual_bars(results, bin_edges, args.output_individual)


if __name__ == "__main__":
    main()
