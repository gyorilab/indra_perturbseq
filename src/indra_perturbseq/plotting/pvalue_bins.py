"""P-value bin distribution plot by pathway type.

For each pathway category (1-hop, 2-hop, 3-hop, unexplained) reports what
percentage of that category's pairs fall in each p-value bin.  Produces a
grouped bar chart so the shape of the p-value distribution is directly
comparable across categories.

Typical usage
-------------
indra-ps-plot-pval-bins \\
    --input-csv  final_dataset_for_boxplots.csv \\
    --output-png pvalue_bins_no_hubs.png \\
    --title-suffix "without hubs"

indra-ps-plot-pval-bins \\
    --input-csv  final_dataset_for_boxplots_with_hubs.csv \\
    --output-png pvalue_bins_with_hubs.png \\
    --title-suffix "with hubs"
"""

from __future__ import annotations

import argparse
import logging
import warnings
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

PATHWAY_ORDER = ["1-hop", "2-hop", "3-hop", "unexplained"]
PALETTE = {
    "1-hop": "#1f77b4",      # blue
    "2-hop": "#2ca02c",      # green
    "3-hop": "#ff7f0e",      # orange
    "unexplained": "#d62728",  # red
}

DEFAULT_BIN_EDGES = [0.0, 0.001, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0]


def _load_dataset(path: str, max_pval: float | None = None) -> pd.DataFrame:
    """Load the combined hop dataset and coerce pval to numeric.

    If *max_pval* is given, only rows with pval <= max_pval are kept.
    This ensures the denominator (total pairs per category) is restricted
    to significant pairs, making hub and non-hub datasets directly comparable.
    """
    df = pd.read_csv(path, low_memory=False)
    if "pval" not in df.columns:
        raise ValueError(f"'pval' column not found in {path}. Columns: {df.columns.tolist()}")
    if "pathway_type" not in df.columns:
        raise ValueError(
            f"'pathway_type' column not found in {path}. Columns: {df.columns.tolist()}"
        )
    df["pval"] = pd.to_numeric(df["pval"], errors="coerce")
    df = df.dropna(subset=["pval"])
    if max_pval is not None:
        before = len(df)
        df = df[df["pval"] <= max_pval]
        logger.info("Filtered to pval <= %g: %d -> %d rows", max_pval, before, len(df))
    return df


def _compute_bin_percentages(
    df: pd.DataFrame,
    bin_edges: Sequence[float],
    bin_labels: Sequence[str],
) -> pd.DataFrame:
    """Return a DataFrame (index=bin_labels, columns=pathway_order) of percentages.

    Each cell is the percentage of that category's total pairs that fall in
    the given bin.  Rows therefore do NOT sum to 100; columns sum to 100.
    """
    df = df.copy()
    df["bin"] = pd.cut(
        df["pval"],
        bins=list(bin_edges),
        labels=list(bin_labels),
        right=False,
        include_lowest=True,
    )

    rows: dict[str, dict[str, float]] = {}
    for cat in PATHWAY_ORDER:
        sub = df[df["pathway_type"] == cat]
        total = len(sub)
        counts = sub["bin"].value_counts()
        rows[cat] = {
            lbl: (counts.get(lbl, 0) / total * 100 if total > 0 else 0.0)
            for lbl in bin_labels
        }

    result = pd.DataFrame(rows, index=list(bin_labels))
    return result


def _plot_bins(
    pct_df: pd.DataFrame,
    out_path: str,
    title_suffix: str = "",
) -> None:
    """Grouped bar chart: x = p-value bins, bars = pathway categories."""
    bin_labels = list(pct_df.index)
    categories = [c for c in PATHWAY_ORDER if c in pct_df.columns]
    n_bins = len(bin_labels)
    n_cats = len(categories)

    bar_width = 0.18
    x = np.arange(n_bins)

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, cat in enumerate(categories):
        offsets = x + (i - n_cats / 2 + 0.5) * bar_width
        heights = pct_df[cat].values
        ax.bar(
            offsets,
            heights,
            width=bar_width,
            label=cat,
            color=PALETTE[cat],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=25, ha="right", fontsize=10)
    ax.set_xlabel("P-value bin", fontsize=12)
    ax.set_ylabel("% of category in bin", fontsize=12)

    title = "P-value Bin Distribution by Pathway Type"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title, fontsize=13)

    ax.axvline(x=3.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
               label="p=0.05 boundary")

    ax.legend(title="Pathway type", fontsize=10, title_fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Grouped bar chart: for each p-value bin, show the %% of each "
            "pathway category (1-hop, 2-hop, 3-hop, unexplained) that falls in it."
        ),
    )
    ap.add_argument(
        "--input-csv", required=True,
        help="Combined hop dataset CSV (e.g. final_dataset_for_boxplots.csv).",
    )
    ap.add_argument(
        "--output-png", default="pvalue_bins.png",
        help="Output PNG path.",
    )
    ap.add_argument(
        "--bin-edges", nargs="+", type=float, default=DEFAULT_BIN_EDGES,
        help="Bin edges as space-separated floats (default: 0 0.001 0.01 0.025 0.05 0.1 0.5 1.0).",
    )
    ap.add_argument(
        "--bin-labels", nargs="+", type=str, default=None,
        help=(
            "Labels for each bin (one fewer than --bin-edges). "
            "Auto-generated from edges if omitted."
        ),
    )
    ap.add_argument(
        "--max-pval", type=float, default=None,
        help=(
            "If set, only pairs with pval <= this threshold are included "
            "(both for computing bin percentages and as the denominator). "
            "Use 0.05 to restrict to significant pairs only."
        ),
    )
    ap.add_argument(
        "--title-suffix", default="",
        help="Optional suffix appended to the plot title (e.g. 'with hubs').",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    edges = sorted(args.bin_edges)
    if args.bin_labels:
        labels = args.bin_labels
        if len(labels) != len(edges) - 1:
            raise SystemExit(
                f"--bin-labels must have {len(edges) - 1} entries "
                f"(one per bin), got {len(labels)}."
            )
    else:
        labels = [f"{edges[i]}–{edges[i + 1]}" for i in range(len(edges) - 1)]

    df = _load_dataset(args.input_csv, max_pval=args.max_pval)
    logger.info(
        "Loaded %d rows. Pathway counts:\n%s",
        len(df),
        df["pathway_type"].value_counts().to_string(),
    )

    pct_df = _compute_bin_percentages(df, edges, labels)
    logger.info("Bin percentages:\n%s", pct_df.round(2).to_string())

    _plot_bins(pct_df, args.output_png, title_suffix=args.title_suffix)
    logger.info("Done.")


if __name__ == "__main__":
    main()
