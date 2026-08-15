"""P-value and log fold-change bin distribution plots by pathway type.

Produces grouped bar charts showing what percentage of each pathway category
falls in each bin. Two modes:

- **pval**: Bins p-values; categories are 1-hop, 2-hop, 3-hop, unexplained.
- **lfc**: Bins |log fold-change|; categories are 1-hop vs all others.

Typical usage
-------------
indra-ps-plot-pval-bins \\
    --input-csv  final_dataset_for_boxplots.csv \\
    --output-png pvalue_bins_no_hubs.png \\
    --max-pval 0.05 --title-suffix "without hubs"

indra-ps-plot-lfc-bins \\
    --input-csv  final_dataset_for_boxplots.csv \\
    --output-png lfc_bins_no_hubs.png \\
    --max-pval 0.05 --title-suffix "without hubs"

indra-ps-plot-lfc-line \\
    --input-csv  final_dataset_for_boxplots.csv \\
    --output-png lfc_line_no_hubs.png \\
    --max-pval 0.05 --title-suffix "without hubs"

indra-ps-plot-pval-line \\
    --input-csv  final_dataset_for_boxplots_with_hubs.csv \\
    --output-png pval_line_with_hubs.png \\
    --max-pval 0.05 --title-suffix "with hubs"
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

# P-value mode: four pathway categories
PATHWAY_ORDER = ["1-hop", "2-hop", "3-hop", "unexplained"]
PATHWAY_PALETTE = {
    "1-hop": "#1f77b4",
    "2-hop": "#2ca02c",
    "3-hop": "#ff7f0e",
    "unexplained": "#d62728",
}

# LFC mode: 1-hop vs all others
LFC_ORDER = ["1-hop", "all others"]
LFC_PALETTE = {
    "1-hop": "#1f77b4",
    "all others": "#d62728",
}

DEFAULT_PVAL_EDGES = [0.0, 0.001, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0]
DEFAULT_LFC_EDGES = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 20.0]
DEFAULT_LFC_LINE_POINTS = 300
DEFAULT_PVAL_LINE_POINTS = 300
DEFAULT_LFC_WINDOW_WIDTH = 0.25
DEFAULT_LFC_WINDOW_STEP = 0.05


def _load_dataset(path: str, max_pval: float | None = None) -> pd.DataFrame:
    """Load the combined hop dataset and coerce pval to numeric.

    If *max_pval* is given, only rows with pval <= max_pval are kept.
    """
    df = pd.read_csv(path, low_memory=False)
    if "pval" not in df.columns:
        raise ValueError(
            f"'pval' column not found in {path}. Columns: {df.columns.tolist()}"
        )
    if "pathway_type" not in df.columns:
        raise ValueError(
            f"'pathway_type' column not found in {path}. "
            f"Columns: {df.columns.tolist()}"
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
    bin_col: str,
    bin_edges: Sequence[float],
    bin_labels: Sequence[str],
    category_col: str,
    categories: Sequence[str],
) -> pd.DataFrame:
    """Return a DataFrame (index=bin_labels, columns=categories) of percentages.

    Each cell is the percentage of that category's total pairs that fall in
    the given bin. Columns sum to 100.
    """
    df = df.copy()
    df["_bin"] = pd.cut(
        df[bin_col],
        bins=list(bin_edges),
        labels=list(bin_labels),
        right=False,
        include_lowest=True,
    )

    rows: dict[str, dict[str, float]] = {}
    for cat in categories:
        sub = df[df[category_col] == cat]
        total = len(sub)
        counts = sub["_bin"].value_counts()
        rows[cat] = {
            lbl: (counts.get(lbl, 0) / total * 100 if total > 0 else 0.0)
            for lbl in bin_labels
        }

    return pd.DataFrame(rows, index=list(bin_labels))


def _plot_grouped_bins(
    pct_df: pd.DataFrame,
    out_path: str,
    categories: Sequence[str],
    palette: dict[str, str],
    xlabel: str,
    title: str,
    title_suffix: str = "",
    ref_line_x: float | None = None,
    ref_line_label: str | None = None,
) -> None:
    """Grouped bar chart: x = bins, bars = categories."""
    bin_labels = list(pct_df.index)
    cats = [c for c in categories if c in pct_df.columns]
    n_bins = len(bin_labels)
    n_cats = len(cats)

    bar_width = 0.35 if n_cats == 2 else 0.18
    x = np.arange(n_bins)

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, cat in enumerate(cats):
        offsets = x + (i - n_cats / 2 + 0.5) * bar_width
        heights = pct_df[cat].values
        ax.bar(
            offsets,
            heights,
            width=bar_width,
            label=cat,
            color=palette.get(cat, "#333333"),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=25, ha="right", fontsize=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("% of category in bin", fontsize=12)

    full_title = title
    if title_suffix:
        full_title += f" — {title_suffix}"
    ax.set_title(full_title, fontsize=13)

    if ref_line_x is not None and ref_line_label:
        ax.axvline(
            x=ref_line_x,
            color="grey",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
            label=ref_line_label,
        )

    ax.legend(title="Pathway type", fontsize=10, title_fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def _plot_lfc_distribution_line(
    df: pd.DataFrame,
    out_path: str,
    title_suffix: str = "",
    n_points: int = DEFAULT_LFC_LINE_POINTS,
    window_width: float = DEFAULT_LFC_WINDOW_WIDTH,
    window_step: float = DEFAULT_LFC_WINDOW_STEP,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    """Continuous line plot of % distribution by |logFC| for each group.

    Uses overlapping sliding windows for smoother, more stable curves.
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    xmin = float(df["abs_logfc"].min())
    xmax = float(df["abs_logfc"].max())
    if np.isclose(xmin, xmax):
        xmax = xmin + 1e-6
    window_width = max(window_width, 1e-9)
    window_step = max(window_step, 1e-9)

    for cat in LFC_ORDER:
        sub = df[df["comparison_group"] == cat]["abs_logfc"].dropna().to_numpy()
        if sub.size == 0:
            continue
        left = xmin
        right = xmax
        if right - left < window_width:
            centers = np.array([(left + right) / 2.0])
        else:
            centers = np.arange(
                left + window_width / 2.0,
                right - window_width / 2.0 + window_step / 2.0,
                window_step,
            )

        y = np.empty_like(centers, dtype=float)
        for i, center in enumerate(centers):
            lo = center - window_width / 2.0
            hi = center + window_width / 2.0
            y[i] = np.mean((sub >= lo) & (sub < hi)) * 100.0

        ax.plot(
            centers,
            y,
            label=cat,
            color=LFC_PALETTE.get(cat, "#333333"),
            linewidth=2.0,
        )

    title = "Continuous |Log Fold-Change| Distribution: 1-hop vs All Others"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("|Log fold-change|", fontsize=12)
    ax.set_ylabel("% of category", fontsize=12)
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Pathway type", fontsize=10, title_fontsize=10)
    fig.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def _plot_lfc_cdf_line(
    df: pd.DataFrame,
    out_path: str,
    title_suffix: str = "",
    n_points: int = DEFAULT_LFC_LINE_POINTS,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = 0.0,
    y_max: float | None = 100.0,
) -> None:
    """Cumulative |logFC| line plot (% of each group up to x)."""
    fig, ax = plt.subplots(figsize=(13, 6))

    xmin = float(df["abs_logfc"].min())
    xmax = float(df["abs_logfc"].max())
    if np.isclose(xmin, xmax):
        xmax = xmin + 1e-6
    xgrid = np.linspace(xmin, xmax, max(int(n_points), 2))

    for cat in LFC_ORDER:
        sub = df[df["comparison_group"] == cat]["abs_logfc"].dropna().to_numpy()
        if sub.size == 0:
            continue
        s = np.sort(sub)
        y = np.searchsorted(s, xgrid, side="right") / s.size * 100.0
        ax.plot(
            xgrid,
            y,
            label=cat,
            color=LFC_PALETTE.get(cat, "#333333"),
            linewidth=2.0,
        )

    title = "Cumulative |Log Fold-Change| Distribution: 1-hop vs All Others"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("|Log fold-change|", fontsize=12)
    ax.set_ylabel("Cumulative % of category", fontsize=12)
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Pathway type", fontsize=10, title_fontsize=10)
    fig.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def _plot_lfc_binshare_line(
    df: pd.DataFrame,
    out_path: str,
    title_suffix: str = "",
    window_width: float = DEFAULT_LFC_WINDOW_WIDTH,
    window_step: float = DEFAULT_LFC_WINDOW_STEP,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    """Line plot of FC-window composition: % 1-hop vs % all-others in each window."""
    fig, ax = plt.subplots(figsize=(13, 6))

    xmin = float(df["abs_logfc"].min())
    xmax = float(df["abs_logfc"].max())
    if np.isclose(xmin, xmax):
        xmax = xmin + 1e-6
    window_width = max(window_width, 1e-9)
    window_step = max(window_step, 1e-9)

    left = xmin
    right = xmax
    if right - left < window_width:
        centers = np.array([(left + right) / 2.0])
    else:
        centers = np.arange(
            left + window_width / 2.0,
            right - window_width / 2.0 + window_step / 2.0,
            window_step,
        )

    s1 = df.loc[df["comparison_group"] == "1-hop", "abs_logfc"].dropna().to_numpy()
    s2 = (
        df.loc[df["comparison_group"] == "all others", "abs_logfc"]
        .dropna()
        .to_numpy()
    )
    y1 = np.empty_like(centers, dtype=float)
    y2 = np.empty_like(centers, dtype=float)
    for i, center in enumerate(centers):
        lo = center - window_width / 2.0
        hi = center + window_width / 2.0
        c1 = np.sum((s1 >= lo) & (s1 < hi))
        c2 = np.sum((s2 >= lo) & (s2 < hi))
        total = c1 + c2
        if total == 0:
            y1[i] = np.nan
            y2[i] = np.nan
        else:
            y1[i] = c1 / total * 100.0
            y2[i] = c2 / total * 100.0

    ax.plot(
        centers,
        y1,
        label="1-hop share in FC window",
        color=LFC_PALETTE.get("1-hop", "#333333"),
        linewidth=2.0,
    )
    ax.plot(
        centers,
        y2,
        label="all others share in FC window",
        color=LFC_PALETTE.get("all others", "#333333"),
        linewidth=2.0,
    )

    title = "Within-Window |Log Fold-Change| Composition: 1-hop vs All Others"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("|Log fold-change|", fontsize=12)
    ax.set_ylabel("% of genes in FC window", fontsize=12)
    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)
    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Pathway type", fontsize=10, title_fontsize=10)
    fig.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def _plot_pval_distribution_line(
    df: pd.DataFrame,
    out_path: str,
    title_suffix: str = "",
    n_points: int = DEFAULT_PVAL_LINE_POINTS,
) -> None:
    """Continuous line plot of % distribution by p-value for each pathway type."""
    fig, ax = plt.subplots(figsize=(13, 6))

    xmin = float(df["pval"].min())
    xmax = float(df["pval"].max())
    if np.isclose(xmin, xmax):
        xmax = xmin + 1e-6

    for cat in PATHWAY_ORDER:
        sub = df[df["pathway_type"] == cat]["pval"].dropna().to_numpy()
        if sub.size == 0:
            continue
        counts, edges = np.histogram(sub, bins=n_points, range=(xmin, xmax))
        centers = (edges[:-1] + edges[1:]) / 2.0
        y = counts / counts.sum() * 100.0
        ax.plot(
            centers,
            y,
            label=cat,
            color=PATHWAY_PALETTE.get(cat, "#333333"),
            linewidth=2.0,
        )

    title = "Continuous P-value Distribution by Pathway Type"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("P-value", fontsize=12)
    ax.set_ylabel("% of category", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Pathway type", fontsize=10, title_fontsize=10)
    fig.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def _run_pval_mode(args: argparse.Namespace) -> None:
    """Run p-value bin distribution plot."""
    edges = sorted(args.bin_edges)
    labels = (
        args.bin_labels
        if args.bin_labels
        else [f"{edges[i]}–{edges[i + 1]}" for i in range(len(edges) - 1)]
    )
    if args.bin_labels and len(labels) != len(edges) - 1:
        raise SystemExit(
            f"--bin-labels must have {len(edges) - 1} entries, got {len(labels)}."
        )

    df = _load_dataset(args.input_csv, max_pval=args.max_pval)
    logger.info(
        "Loaded %d rows. Pathway counts:\n%s",
        len(df),
        df["pathway_type"].value_counts().to_string(),
    )

    pct_df = _compute_bin_percentages(
        df, "pval", edges, labels, "pathway_type", PATHWAY_ORDER
    )
    logger.info("Bin percentages:\n%s", pct_df.round(2).to_string())

    _plot_grouped_bins(
        pct_df,
        args.output_png,
        PATHWAY_ORDER,
        PATHWAY_PALETTE,
        xlabel="P-value bin",
        title="P-value Bin Distribution by Pathway Type",
        title_suffix=args.title_suffix,
        ref_line_x=3.5,
        ref_line_label="p=0.05 boundary",
    )


def _run_lfc_mode(args: argparse.Namespace) -> None:
    """Run |log fold-change| bin distribution plot (1-hop vs all others)."""
    df = _load_dataset(args.input_csv, max_pval=args.max_pval)
    if "logfoldchange" not in df.columns:
        raise ValueError(
            f"'logfoldchange' column not found in {args.input_csv}. "
            f"Columns: {df.columns.tolist()}"
        )

    df["abs_logfc"] = pd.to_numeric(df["logfoldchange"], errors="coerce").abs()
    df = df.dropna(subset=["abs_logfc"])
    df["comparison_group"] = df["pathway_type"].apply(
        lambda x: "1-hop" if x == "1-hop" else "all others"
    )

    edges = sorted(args.bin_edges)
    labels = (
        args.bin_labels
        if args.bin_labels
        else [f"{edges[i]}–{edges[i + 1]}" for i in range(len(edges) - 1)]
    )
    if args.bin_labels and len(labels) != len(edges) - 1:
        raise SystemExit(
            f"--bin-labels must have {len(edges) - 1} entries, got {len(labels)}."
        )

    logger.info(
        "Loaded %d rows. Comparison counts:\n%s",
        len(df),
        df["comparison_group"].value_counts().to_string(),
    )

    pct_df = _compute_bin_percentages(
        df, "abs_logfc", edges, labels, "comparison_group", LFC_ORDER
    )
    logger.info("Bin percentages:\n%s", pct_df.round(2).to_string())

    _plot_grouped_bins(
        pct_df,
        args.output_png,
        LFC_ORDER,
        LFC_PALETTE,
        xlabel="|Log fold-change| bin",
        title="|Log Fold-Change| Bin Distribution: 1-hop vs All Others",
        title_suffix=args.title_suffix,
    )


def _run_lfc_line_mode(args: argparse.Namespace) -> None:
    """Run continuous |logFC| line plot for 1-hop vs all others."""
    df = _load_dataset(args.input_csv, max_pval=args.max_pval)
    if "logfoldchange" not in df.columns:
        raise ValueError(
            f"'logfoldchange' column not found in {args.input_csv}. "
            f"Columns: {df.columns.tolist()}"
        )

    df["abs_logfc"] = pd.to_numeric(df["logfoldchange"], errors="coerce").abs()
    df = df.dropna(subset=["abs_logfc"])
    df["comparison_group"] = df["pathway_type"].apply(
        lambda x: "1-hop" if x == "1-hop" else "all others"
    )

    logger.info(
        "Loaded %d rows. Comparison counts:\n%s",
        len(df),
        df["comparison_group"].value_counts().to_string(),
    )
    if args.line_mode == "cdf":
        _plot_lfc_cdf_line(
            df,
            args.output_png,
            title_suffix=args.title_suffix,
            n_points=args.n_points,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
        )
    else:
        _plot_lfc_distribution_line(
            df,
            args.output_png,
            title_suffix=args.title_suffix,
            n_points=args.n_points,
            window_width=args.window_width,
            window_step=args.window_step,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
        )


def _run_lfc_binshare_line_mode(args: argparse.Namespace) -> None:
    """Run FC-window composition line plot for 1-hop vs all others."""
    df = _load_dataset(args.input_csv, max_pval=args.max_pval)
    if "logfoldchange" not in df.columns:
        raise ValueError(
            f"'logfoldchange' column not found in {args.input_csv}. "
            f"Columns: {df.columns.tolist()}"
        )

    df["abs_logfc"] = pd.to_numeric(df["logfoldchange"], errors="coerce").abs()
    df = df.dropna(subset=["abs_logfc"])
    df["comparison_group"] = df["pathway_type"].apply(
        lambda x: "1-hop" if x == "1-hop" else "all others"
    )

    logger.info(
        "Loaded %d rows. Comparison counts:\n%s",
        len(df),
        df["comparison_group"].value_counts().to_string(),
    )
    _plot_lfc_binshare_line(
        df,
        args.output_png,
        title_suffix=args.title_suffix,
        window_width=args.window_width,
        window_step=args.window_step,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
    )


def _run_pval_line_mode(args: argparse.Namespace) -> None:
    """Run continuous p-value line plot for pathway categories."""
    df = _load_dataset(args.input_csv, max_pval=args.max_pval)
    logger.info(
        "Loaded %d rows. Pathway counts:\n%s",
        len(df),
        df["pathway_type"].value_counts().to_string(),
    )
    _plot_pval_distribution_line(
        df,
        args.output_png,
        title_suffix=args.title_suffix,
        n_points=args.n_points,
    )


def _add_common_args(ap: argparse.ArgumentParser) -> None:
    """Add arguments shared by both pval and lfc modes."""
    ap.add_argument(
        "--input-csv",
        required=True,
        help="Combined hop dataset CSV (e.g. final_dataset_for_boxplots.csv).",
    )
    ap.add_argument(
        "--output-png",
        default=None,
        help="Output PNG path (default depends on mode).",
    )
    ap.add_argument(
        "--max-pval",
        type=float,
        default=None,
        help=(
            "Only pairs with pval <= this threshold. "
            "Use 0.05 for significant pairs only (recommended)."
        ),
    )
    ap.add_argument(
        "--title-suffix",
        default="",
        help="Optional suffix appended to the plot title (e.g. 'with hubs').",
    )


def main(argv: list[str] | None = None) -> None:
    """P-value bin distribution by pathway type."""
    ap = argparse.ArgumentParser(
        description=(
            "Grouped bar chart: for each p-value bin, show the %% of each "
            "pathway category (1-hop, 2-hop, 3-hop, unexplained) that falls in it."
        ),
    )
    _add_common_args(ap)
    ap.add_argument(
        "--bin-edges",
        nargs="+",
        type=float,
        default=DEFAULT_PVAL_EDGES,
        help="P-value bin edges (default: 0 0.001 0.01 0.025 0.05 0.1 0.5 1.0).",
    )
    ap.add_argument(
        "--bin-labels",
        nargs="+",
        type=str,
        default=None,
        help="Labels for each bin (one fewer than --bin-edges).",
    )
    args = ap.parse_args(argv)
    if args.output_png is None:
        args.output_png = "pvalue_bins.png"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _run_pval_mode(args)
    logger.info("Done.")


def main_lfc(argv: list[str] | None = None) -> None:
    """|Log fold-change| bin distribution: 1-hop vs all others."""
    ap = argparse.ArgumentParser(
        description=(
            "Grouped bar chart: for each |logFC| bin, show the %% of "
            "1-hop vs all others (2-hop + 3-hop + unexplained) that falls in it."
        ),
    )
    _add_common_args(ap)
    ap.add_argument(
        "--bin-edges",
        nargs="+",
        type=float,
        default=DEFAULT_LFC_EDGES,
        help="|logFC| bin edges (default: 0 0.25 0.5 1 2 5 20).",
    )
    ap.add_argument(
        "--bin-labels",
        nargs="+",
        type=str,
        default=None,
        help="Labels for each bin (one fewer than --bin-edges).",
    )
    args = ap.parse_args(argv)
    if args.output_png is None:
        args.output_png = "lfc_bins.png"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _run_lfc_mode(args)
    logger.info("Done.")


def main_lfc_line(argv: list[str] | None = None) -> None:
    """Continuous |logFC| line plot: % distribution for 1-hop vs all others."""
    ap = argparse.ArgumentParser(
        description=(
            "Continuous line plot: x is |logFC| and y is % of each group "
            "(1-hop vs all others) across fine-grained x bins."
        ),
    )
    _add_common_args(ap)
    ap.add_argument(
        "--n-points",
        type=int,
        default=DEFAULT_LFC_LINE_POINTS,
        help="X-grid density for line plots.",
    )
    ap.add_argument(
        "--line-mode",
        choices=["pdf", "cdf"],
        default="pdf",
        help="`pdf` = overlapping-window density line, `cdf` = cumulative line.",
    )
    ap.add_argument(
        "--window-width",
        type=float,
        default=DEFAULT_LFC_WINDOW_WIDTH,
        help="Sliding window width on |logFC| axis for smoothing (default: 0.25).",
    )
    ap.add_argument(
        "--window-step",
        type=float,
        default=DEFAULT_LFC_WINDOW_STEP,
        help="Step size between window centers (default: 0.05).",
    )
    ap.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Optional lower x-axis bound for display.",
    )
    ap.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional upper x-axis bound for display.",
    )
    ap.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Optional lower y-axis bound for display.",
    )
    ap.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional upper y-axis bound for display.",
    )
    args = ap.parse_args(argv)
    if args.output_png is None:
        args.output_png = "lfc_line.png"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _run_lfc_line_mode(args)
    logger.info("Done.")


def main_lfc_binshare_line(argv: list[str] | None = None) -> None:
    """Continuous |logFC| line plot: within-window % 1-hop vs all-others."""
    ap = argparse.ArgumentParser(
        description=(
            "Continuous line plot: x is |logFC| and y is the composition of each "
            "FC window (% 1-hop vs % all others within that window)."
        ),
    )
    _add_common_args(ap)
    ap.add_argument(
        "--window-width",
        type=float,
        default=DEFAULT_LFC_WINDOW_WIDTH,
        help="Sliding window width on |logFC| axis (default: 0.25).",
    )
    ap.add_argument(
        "--window-step",
        type=float,
        default=DEFAULT_LFC_WINDOW_STEP,
        help="Step size between window centers (default: 0.05).",
    )
    ap.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Optional lower x-axis bound for display.",
    )
    ap.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional upper x-axis bound for display.",
    )
    ap.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Optional lower y-axis bound for display.",
    )
    ap.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional upper y-axis bound for display.",
    )
    args = ap.parse_args(argv)
    if args.output_png is None:
        args.output_png = "lfc_binshare_line.png"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _run_lfc_binshare_line_mode(args)
    logger.info("Done.")


def main_pval_line(argv: list[str] | None = None) -> None:
    """Continuous p-value line plot: % distribution by pathway type."""
    ap = argparse.ArgumentParser(
        description=(
            "Continuous line plot: x is p-value and y is % of each pathway "
            "category across fine-grained x bins."
        ),
    )
    _add_common_args(ap)
    ap.add_argument(
        "--n-points",
        type=int,
        default=DEFAULT_PVAL_LINE_POINTS,
        help="Number of x-grid points used for the line plot.",
    )
    args = ap.parse_args(argv)
    if args.output_png is None:
        args.output_png = "pvalue_line.png"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _run_pval_line_mode(args)
    logger.info("Done.")


if __name__ == "__main__":
    main()
