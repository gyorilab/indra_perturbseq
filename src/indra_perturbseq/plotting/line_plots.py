"""Overlaid TPR/FPR line plots across thresholds."""

from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def _load_stats(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"threshold", "TPR_overall", "FPR_overall"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=list(required)).sort_values("threshold")


def _plot_overlay(datasets, metric_col, ylabel, title, out_png):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for label, df, style, color in datasets:
        ax.plot(df["threshold"].values, df[metric_col].values,
                style, color=color, marker="o", markersize=6,
                linewidth=2, alpha=0.9, label=label)
    ax.set_xlabel("p-value threshold")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out_png)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Overlaid TPR/FPR line plots.",
    )
    ap.add_argument("--onehop-csv", required=True)
    ap.add_argument("--twohop-csv", required=True)
    ap.add_argument("--out-prefix", required=True,
                    help="Output file prefix (e.g. output/overlay)")
    ap.add_argument("--title-suffix", default="")
    args = ap.parse_args(argv)

    df1 = _load_stats(args.onehop_csv)
    df2 = _load_stats(args.twohop_csv)

    datasets = [
        ("1-hop", df1, "-", "#1f77b4"),
        ("<=2-hop", df2, "--", "#d62728"),
    ]

    _plot_overlay(datasets, "TPR_overall",
                  "TPR (overall)",
                  f"TPR vs threshold {args.title_suffix}".strip(),
                  f"{args.out_prefix}_tpr.png")
    _plot_overlay(datasets, "FPR_overall",
                  "FPR (overall)",
                  f"FPR vs threshold {args.title_suffix}".strip(),
                  f"{args.out_prefix}_fpr.png")


if __name__ == "__main__":
    main()
