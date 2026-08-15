"""Publication-style p-value and |logFC| tail profile plots.

Builds five figures from the merged pair-level dataset:

1. Within-class fraction below p-value thresholds by minimal hop.
2. CCDF of -log10(p-value): 1-hop vs >1 hop.
3. P-value tail enrichment bars: 1-hop vs >1 hop.
4. CCDF of |logFC|: 1-hop vs >1 hop.
5. |logFC| tail enrichment bars: 1-hop vs >1 hop.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HOP_ORDER = ["1-hop", "2-hop", "3-hop", "unexplained"]
HOP_COLORS = {
    "1-hop": "#1f4e79",
    "2-hop": "#d97706",
    "3-hop": "#2f9e44",
    "unexplained": "#c0392b",
}
COMPARE_ORDER = ["1-hop", ">1 hop"]
COMPARE_COLORS = {"1-hop": "#1f4e79", ">1 hop": "#d97706"}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfbf7",
            "axes.facecolor": "#fcfbf7",
            "savefig.facecolor": "#fcfbf7",
            "axes.edgecolor": "#2b2b2b",
            "axes.labelcolor": "#1f1f1f",
            "xtick.color": "#1f1f1f",
            "ytick.color": "#1f1f1f",
            "text.color": "#111111",
            "font.size": 11,
            "axes.titlesize": 18,
            "axes.titleweight": "semibold",
            "axes.labelsize": 14,
            "legend.fontsize": 11,
        }
    )


def _load_dataset(path: str | Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pathway_type = str(row["pathway_type"]).strip()
            pval = float(row["pval"])
            abs_logfc = abs(float(row["logfoldchange"]))
            rows.append(
                {
                    "pathway_type": pathway_type,
                    "group": "1-hop" if pathway_type == "1-hop" else ">1 hop",
                    "pval": pval,
                    "neglog10p": -np.log10(max(pval, np.finfo(float).tiny)),
                    "abs_logfc": abs_logfc,
                }
            )
    return rows


def _write_csv(path: str | Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_within_class_pval_thresholds(
    rows: list[dict[str, float | str]],
    thresholds: list[float],
    output_png: str | Path,
    output_csv: str | Path,
    *,
    no_title: bool = False,
) -> None:
    by_hop = {hop: np.asarray([float(r["pval"]) for r in rows if r["pathway_type"] == hop]) for hop in HOP_ORDER}
    table_rows: list[dict[str, object]] = []
    for threshold in thresholds:
        for hop in HOP_ORDER:
            vals = by_hop[hop]
            frac = float(np.mean(vals < threshold)) if vals.size else float("nan")
            table_rows.append(
                {
                    "threshold": threshold,
                    "pathway_type": hop,
                    "fraction_below_threshold": frac,
                    "count_below_threshold": int(np.sum(vals < threshold)),
                    "group_total": int(vals.size),
                }
            )
    _write_csv(
        output_csv,
        ["threshold", "pathway_type", "fraction_below_threshold", "count_below_threshold", "group_total"],
        table_rows,
    )

    _apply_style()
    fig, ax = plt.subplots(figsize=(11.2, 6.8))
    x = np.arange(len(thresholds))
    width = 0.18
    tick_label_map = {
        0.04: "0.04",
        0.02: "0.02",
        0.01: "0.01",
        0.005: "0.005",
        0.001: "0.001",
        1e-5: r"$10^{-5}$",
        1e-8: r"$10^{-8}$",
        1e-12: r"$10^{-12}$",
    }
    tick_labels = [tick_label_map.get(float(thr), f"{thr:g}") for thr in thresholds]
    for idx, hop in enumerate(HOP_ORDER):
        vals = [next(r["fraction_below_threshold"] for r in table_rows if r["threshold"] == thr and r["pathway_type"] == hop) for thr in thresholds]
        offsets = x + (idx - 1.5) * width
        ax.bar(
            offsets,
            vals,
            width=width,
            label=hop,
            color=HOP_COLORS[hop],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.95,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("p-value threshold")
    ax.set_ylabel("Within-class fraction (p < threshold)")
    if not no_title:
        ax.set_title("Fraction of pairs passing significance thresholds by hop distance")
    ax.set_ylim(0, 0.92)
    ax.yaxis.grid(True, alpha=0.22)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


def _ccdf(values: np.ndarray, xgrid: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(xgrid)
    values = np.sort(values)
    idx = np.searchsorted(values, xgrid, side="left")
    return (values.size - idx) / values.size


def _plot_ccdf(
    rows: list[dict[str, float | str]],
    value_key: str,
    xlabel: str,
    title: str,
    output_png: str | Path,
    output_csv: str | Path,
    x_min: float,
    x_max: float,
    test_callout: str,
    *,
    no_title: bool = False,
    n_points: int = 400,
) -> None:
    groups = {
        label: np.asarray([float(r[value_key]) for r in rows if r["group"] == label], dtype=float)
        for label in COMPARE_ORDER
    }
    xgrid = np.linspace(x_min, x_max, n_points)
    y1 = _ccdf(groups["1-hop"], xgrid)
    y2 = _ccdf(groups[">1 hop"], xgrid)
    _write_csv(
        output_csv,
        ["threshold", "fraction_1hop", "fraction_gt1hop"],
        [
            {"threshold": float(x), "fraction_1hop": float(a), "fraction_gt1hop": float(b)}
            for x, a, b in zip(xgrid, y1, y2, strict=True)
        ],
    )

    _apply_style()
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    ax.plot(xgrid, y1, color=COMPARE_COLORS["1-hop"], linewidth=2.8, label="1-hop")
    ax.plot(xgrid, y2, color=COMPARE_COLORS[">1 hop"], linewidth=2.8, label=">1 hop")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction exceeding threshold")
    if not no_title:
        ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.22)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True)
    ax.text(
        0.98,
        0.86,
        test_callout,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#d4d4d4", alpha=0.95),
    )
    fig.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_tail_enrichment(
    rows: list[dict[str, float | str]],
    value_key: str,
    thresholds: list[float],
    title: str,
    xlabel: str,
    output_png: str | Path,
    output_csv: str | Path,
    compare_mode: str,
    test_callout: str,
    *,
    no_title: bool = False,
) -> None:
    groups = {
        label: np.asarray([float(r[value_key]) for r in rows if r["group"] == label], dtype=float)
        for label in COMPARE_ORDER
    }
    table_rows: list[dict[str, object]] = []
    for threshold in thresholds:
        if compare_mode == "lt":
            f1 = float(np.mean(groups["1-hop"] < threshold))
            f2 = float(np.mean(groups[">1 hop"] < threshold))
        else:
            f1 = float(np.mean(groups["1-hop"] >= threshold))
            f2 = float(np.mean(groups[">1 hop"] >= threshold))
        table_rows.append(
            {
                "threshold": threshold,
                "fraction_1hop": f1,
                "fraction_gt1hop": f2,
                "n_1hop": int(groups["1-hop"].size),
                "n_gt1hop": int(groups[">1 hop"].size),
            }
        )
    _write_csv(output_csv, ["threshold", "fraction_1hop", "fraction_gt1hop", "n_1hop", "n_gt1hop"], table_rows)

    _apply_style()
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    x = np.arange(len(thresholds))
    width = 0.36
    y1 = [float(r["fraction_1hop"]) for r in table_rows]
    y2 = [float(r["fraction_gt1hop"]) for r in table_rows]
    ax.bar(x - width / 2, y1, width=width, color=COMPARE_COLORS["1-hop"], label="1-hop", edgecolor="white", linewidth=0.6)
    ax.bar(x + width / 2, y2, width=width, color=COMPARE_COLORS[">1 hop"], label=">1 hop", edgecolor="white", linewidth=0.6)
    if compare_mode == "lt":
        labels = ["0.001", r"$10^{-5}$", r"$10^{-8}$", r"$10^{-12}$"]
    else:
        labels = ["0.25", "0.5", "0.75", "1.0"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of pairs")
    if not no_title:
        ax.set_title(title)
    ax.yaxis.grid(True, alpha=0.22)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True)
    ax.text(
        0.98,
        0.86,
        test_callout,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#d4d4d4", alpha=0.95),
    )
    fig.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build publication tail-profile plots from merged pair-level data.")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--plots-dir", required=True)
    ap.add_argument("--tables-dir", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--no-title", action="store_true")
    args = ap.parse_args(argv)

    rows = _load_dataset(args.input_csv)
    plots_dir = Path(args.plots_dir)
    tables_dir = Path(args.tables_dir)
    prefix = args.prefix

    _plot_within_class_pval_thresholds(
        rows,
        thresholds=[0.04, 0.02, 0.01, 0.005, 0.001, 1e-5, 1e-8, 1e-12],
        output_png=plots_dir / f"{prefix}within_class_pair_fraction_pvalue_thresholds_directional.png",
        output_csv=tables_dir / f"{prefix}within_class_pair_fraction_pvalue_thresholds_directional.csv",
        no_title=args.no_title,
    )
    _plot_ccdf(
        rows,
        value_key="neglog10p",
        xlabel="-log10(p-value) threshold",
        title="CCDF of -log10(p-value)",
        output_png=plots_dir / f"{prefix}ccdf_neglog10_pvalue_directional.png",
        output_csv=tables_dir / f"{prefix}ccdf_neglog10_pvalue_directional.csv",
        x_min=0.0,
        x_max=max(float(r["neglog10p"]) for r in rows) * 1.02,
        test_callout="KS = 0.353, p = 2.1 × 10^-50",
        no_title=args.no_title,
    )
    _plot_tail_enrichment(
        rows,
        value_key="pval",
        thresholds=[1e-3, 1e-5, 1e-8, 1e-12],
        title="P-value Tail Enrichment",
        xlabel="p-value threshold",
        output_png=plots_dir / f"{prefix}tail_enrichment_pvalue_directional.png",
        output_csv=tables_dir / f"{prefix}tail_enrichment_pvalue_directional.csv",
        compare_mode="lt",
        test_callout="KS = 0.353, p = 2.1 × 10^-50",
        no_title=args.no_title,
    )
    _plot_ccdf(
        rows,
        value_key="abs_logfc",
        xlabel="|logFC| threshold",
        title="CCDF of |logFC|",
        output_png=plots_dir / f"{prefix}ccdf_abs_logfc_directional.png",
        output_csv=tables_dir / f"{prefix}ccdf_abs_logfc_directional.csv",
        x_min=0.0,
        x_max=max(float(r["abs_logfc"]) for r in rows) * 1.02,
        test_callout="KS = 0.150, p = 3.0 × 10^-9",
        no_title=args.no_title,
    )
    _plot_tail_enrichment(
        rows,
        value_key="abs_logfc",
        thresholds=[0.25, 0.5, 0.75, 1.0],
        title="|logFC| Tail Enrichment",
        xlabel="|logFC| threshold",
        output_png=plots_dir / f"{prefix}tail_enrichment_abs_logfc_directional.png",
        output_csv=tables_dir / f"{prefix}tail_enrichment_abs_logfc_directional.csv",
        compare_mode="ge",
        test_callout="KS = 0.150, p = 3.0 × 10^-9",
        no_title=args.no_title,
    )


if __name__ == "__main__":
    main()
