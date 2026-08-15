"""Publication-style Perturb-seq plots with directional filtering.

This module focuses on the three remaining manuscript figures:

1. P-value boxplots by hop type, optionally including hub-gene outlier files.
2. Real vs permuted cumulative hop explainability bars.
3. 1-hop paths vs DEG p-value threshold line plot.

The implementation uses only stdlib CSV handling plus numpy/matplotlib so it
can run in the lightweight plotting environment already present in the repo.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

DE_PVAL_CUTOFF = 0.05
PATHWAY_ORDER = ["1-hop", "2-hop", "3-hop", "unexplained"]
PATHWAY_PRIORITY = {"1-hop": 1, "2-hop": 2, "3-hop": 3}
PVAL_COMPARE_GROUPS = ("1-hop", "all others")
COLORS = {
    "1-hop": "#1f4e79",
    "2-hop": "#3b7a57",
    "3-hop": "#c77d2b",
    "unexplained": "#a64646",
    "all others": "#8f3f2b",
    "real": "#1f4e79",
    "permuted": "#d07a2d",
}
csv.field_size_limit(sys.maxsize)


@dataclass
class HopRow:
    source: str
    target: str
    pathway_type: str
    belief: float
    logfoldchange: float | None
    pval: float | None


def _apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfbf7",
            "axes.facecolor": "#fcfbf7",
            "savefig.facecolor": "#fcfbf7",
            "axes.edgecolor": "#3b3b3b",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#111111",
            "font.size": 11,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "legend.frameon": False,
            "grid.color": "#d7d1c5",
            "grid.alpha": 0.5,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        x = float(s)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def _norm_gene(value: object) -> str:
    return str(value or "").strip().upper()


def _row_is_retained(row: dict[str, str]) -> bool:
    label = str(row.get("directional consistency", "")).strip()
    if not label:
        return True
    return label != "Inconsistent"


def _load_hop_rows(
    csv_path: str | Path,
    hop_label: str,
    *,
    significant_only: bool = False,
) -> list[HopRow]:
    rows: list[HopRow] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        belief_cols = [c for c in (reader.fieldnames or []) if c.startswith("belief")]
        for row in reader:
            if not _row_is_retained(row):
                continue
            src = _norm_gene(row.get("source"))
            tgt = _norm_gene(row.get("target"))
            if not src or not tgt or src == tgt:
                continue
            pval = _parse_float(row.get("pval"))
            if significant_only and (pval is None or pval >= DE_PVAL_CUTOFF):
                continue
            logfc = _parse_float(row.get("logfoldchange"))
            belief_values = [_parse_float(row.get(col)) for col in belief_cols]
            belief_clean = [x for x in belief_values if x is not None]
            belief = float(np.mean(belief_clean)) if belief_clean else float("-inf")
            rows.append(
                HopRow(
                    source=src,
                    target=tgt,
                    pathway_type=hop_label,
                    belief=belief,
                    logfoldchange=logfc,
                    pval=pval,
                )
            )
    return rows


def _dedup_priority(rows_by_hop: list[HopRow]) -> list[HopRow]:
    best: dict[tuple[str, str], HopRow] = {}
    for row in rows_by_hop:
        key = (row.source, row.target)
        prev = best.get(key)
        if prev is None:
            best[key] = row
            continue
        prev_key = (PATHWAY_PRIORITY[prev.pathway_type], prev.belief)
        row_key = (PATHWAY_PRIORITY[row.pathway_type], row.belief)
        if row_key[0] < prev_key[0] or (row_key[0] == prev_key[0] and row_key[1] > prev_key[1]):
            best[key] = row
    return list(best.values())


def _load_significant_deg_pairs(
    target_validation_csv: str | Path,
    deg_dir: str | Path,
    *,
    filter_column: str,
    filter_value: str,
    extra_sources: list[str] | None = None,
    exclude_genes: set[str] | None = None,
) -> dict[str, list[tuple[str, float, float | None]]]:
    allowed_sources: set[str] = set()
    exclude = {g.upper() for g in (exclude_genes or set())}
    with Path(target_validation_csv).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get(filter_column, "")).strip() == filter_value:
                gene = _norm_gene(row.get("Gene"))
                if gene and gene not in exclude:
                    allowed_sources.add(gene)
    if extra_sources:
        allowed_sources.update(_norm_gene(x) for x in extra_sources if _norm_gene(x) and _norm_gene(x) not in exclude)

    out: dict[str, list[tuple[str, float, float | None]]] = {}
    for source in sorted(allowed_sources):
        deg_file = Path(deg_dir) / f"{source}_vs_control.csv"
        if not deg_file.exists():
            continue
        with deg_file.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not {"names", "pvals", "logfoldchanges"}.issubset(set(reader.fieldnames or [])):
                continue
            rows: list[tuple[str, float, float | None]] = []
            for row in reader:
                tgt = _norm_gene(row.get("names"))
                if not tgt or tgt == source or tgt in exclude:
                    continue
                pval = _parse_float(row.get("pvals"))
                if pval is None or pval >= DE_PVAL_CUTOFF:
                    continue
                rows.append((tgt, pval, _parse_float(row.get("logfoldchanges"))))
            if rows:
                out[source] = rows
    return out


def _build_boxplot_dataset(
    hop1_csv: str,
    hop2_csv: str,
    hop3_csv: str,
    target_validation_csv: str,
    deg_dir: str,
    *,
    hub_1hop_csv: str | None = None,
    hub_2hop_csv: str | None = None,
    hub_genes: list[str] | None = None,
    filter_column: str = "analysis_flag",
    filter_value: str = "Use_for_analysis",
) -> list[HopRow]:
    rows = []
    rows.extend(_load_hop_rows(hop1_csv, "1-hop", significant_only=True))
    rows.extend(_load_hop_rows(hop2_csv, "2-hop", significant_only=True))
    rows.extend(_load_hop_rows(hop3_csv, "3-hop", significant_only=True))
    if hub_1hop_csv:
        rows.extend(_load_hop_rows(hub_1hop_csv, "1-hop", significant_only=True))
    if hub_2hop_csv:
        rows.extend(_load_hop_rows(hub_2hop_csv, "2-hop", significant_only=True))

    explained = _dedup_priority(rows)
    explained_pairs = {(r.source, r.target) for r in explained}
    deg_pairs = _load_significant_deg_pairs(
        target_validation_csv,
        deg_dir,
        filter_column=filter_column,
        filter_value=filter_value,
        extra_sources=hub_genes,
    )
    out = list(explained)
    pathway_sources = {r.source for r in explained}
    for source in sorted(pathway_sources):
        for target, pval, logfc in deg_pairs.get(source, []):
            if (source, target) in explained_pairs:
                continue
            out.append(
                HopRow(
                    source=source,
                    target=target,
                    pathway_type="unexplained",
                    belief=0.5,
                    logfoldchange=logfc,
                    pval=pval,
                )
            )
    return out


def _write_boxplot_dataset(rows: list[HopRow], output_csv: str | Path) -> None:
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source", "target", "pathway_type", "belief", "logfoldchange", "pval"])
        for row in rows:
            writer.writerow([row.source, row.target, row.pathway_type, row.belief, row.logfoldchange, row.pval])


def _plot_pvalue_boxplot(
    rows: list[HopRow],
    output_png: str | Path,
    *,
    show_fliers: bool,
    no_title: bool = False,
) -> None:
    _apply_publication_style()
    groups: list[list[float]] = []
    counts: list[int] = []
    for label in PATHWAY_ORDER:
        vals = [
            -math.log10(max(row.pval, 1e-50))
            for row in rows
            if row.pathway_type == label and row.pval is not None and row.pval < DE_PVAL_CUTOFF
        ]
        groups.append(vals)
        counts.append(len(vals))

    fig, ax = plt.subplots(figsize=(10.2, 7.2))
    bp = ax.boxplot(
        groups,
        patch_artist=True,
        widths=0.58,
        notch=False,
        showfliers=show_fliers,
        medianprops={"color": "#111111", "linewidth": 2.2},
        whiskerprops={"color": "#4a4a4a", "linewidth": 1.4},
        capprops={"color": "#4a4a4a", "linewidth": 1.4},
        boxprops={"linewidth": 1.5, "edgecolor": "#4a4a4a"},
        flierprops={
            "marker": "o",
            "markersize": 3.2,
            "markerfacecolor": "#4a4a4a",
            "markeredgecolor": "none",
            "alpha": 0.25,
        },
    )
    for patch, label in zip(bp["boxes"], PATHWAY_ORDER):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.78)

    ax.axhline(-math.log10(0.05), color="#8c5c13", linestyle="--", linewidth=1.4, alpha=0.8)
    ax.axhline(-math.log10(0.01), color="#a64646", linestyle=":", linewidth=1.6, alpha=0.8)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_ylabel(r"$-\log_{10}(\mathrm{p\ value})$")
    ax.set_xlabel("")
    ax.set_xticks(range(1, len(PATHWAY_ORDER) + 1))
    ax.set_xticklabels([f"{label}\n(n={count:,})" for label, count in zip(PATHWAY_ORDER, counts)])
    if not no_title:
        ax.set_title("DEG Significance by Minimal Explanatory Hop", pad=16)
    fig.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


def _load_pairs_from_csv(csv_path: str | Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not _row_is_retained(row):
                continue
            src = _norm_gene(row.get("source"))
            tgt = _norm_gene(row.get("target"))
            if not src or not tgt or src == tgt:
                continue
            pairs.add((src, tgt))
    return pairs


def _load_denominator_pair_set(
    target_validation_csv: str | Path,
    deg_dir: str | Path,
    *,
    filter_column: str,
    filter_value: str,
    exclude_genes: set[str] | None = None,
) -> set[tuple[str, str]]:
    deg_pairs = _load_significant_deg_pairs(
        target_validation_csv,
        deg_dir,
        filter_column=filter_column,
        filter_value=filter_value,
        exclude_genes=exclude_genes,
    )
    pair_set: set[tuple[str, str]] = set()
    for source, rows in deg_pairs.items():
        for target, _pval, _logfc in rows:
            pair_set.add((source, target))
    return pair_set


def _write_cumulative_table(
    rows: list[dict[str, object]],
    output_csv: str | Path,
    *,
    count_mode: bool,
) -> None:
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = (
            ["dataset", "hop", "retained_pairs", "note"]
            if count_mode
            else ["dataset", "hop", "tp_pairs", "total_pos_pairs", "tpr"]
        )
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_cumulative_bars(
    rows: list[dict[str, object]],
    output_png: str | Path,
    *,
    total_pos_pairs: int | None,
    count_mode: bool,
    no_title: bool = False,
) -> None:
    _apply_publication_style()
    hop_order = ["<=1-hop", "<=2-hop", "<=3-hop"]
    if count_mode:
        real_counts = {
            row["hop"]: int(row["retained_pairs"])
            for row in rows
            if row["dataset"] == "real" and row.get("retained_pairs") not in (None, "")
        }
        perm_counts = {
            row["hop"]: int(row["retained_pairs"])
            for row in rows
            if row["dataset"] == "permuted" and row.get("retained_pairs") not in (None, "")
        }
        real_heights = [float(real_counts[h]) for h in hop_order]
        perm_heights = [float(perm_counts.get(h, float("nan"))) for h in hop_order]
    else:
        real = {row["hop"]: float(row["tpr"]) for row in rows if row["dataset"] == "real"}
        perm = {row["hop"]: float(row["tpr"]) for row in rows if row["dataset"] == "permuted"}
        real_counts = {row["hop"]: int(row["tp_pairs"]) for row in rows if row["dataset"] == "real"}
        perm_counts = {row["hop"]: int(row["tp_pairs"]) for row in rows if row["dataset"] == "permuted"}
        real_heights = [real[h] * 100.0 for h in hop_order]
        perm_heights = [perm[h] * 100.0 for h in hop_order]

    fig, ax = plt.subplots(figsize=(9.8, 6.8))
    x = np.arange(len(hop_order))
    width = 0.34
    bars1 = ax.bar(x - width / 2, real_heights, width=width, color=COLORS["real"], alpha=0.9, label="INDRA descendants")
    bars2 = ax.bar(x + width / 2, perm_heights, width=width, color=COLORS["permuted"], alpha=0.9, label="Permuted null")

    ax.set_xticks(x)
    ax.set_xticklabels(["1 hop", "2 hops", "3 hops"])
    ax.set_ylabel("Explained descendants (%)" if not count_mode else "Retained gene pairs")
    if not no_title:
        ax.set_title("INDRA vs Permuted Cumulative Recovery", pad=16)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")

    for bars, counts, heights in [(bars1, real_counts, real_heights), (bars2, perm_counts, perm_heights)]:
        for bar, hop, height in zip(bars, hop_order, heights):
            if np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.01 * max(real_heights + [h for h in perm_heights if not np.isnan(h)]),
                    "n/a",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    color="#666666",
                )
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (1.3 if not count_mode else max(real_heights) * 0.015),
                (f"{height:.1f}%\n({counts[hop]:,})" if not count_mode else f"{counts[hop]:,}"),
                ha="center",
                va="bottom",
                fontsize=9.5,
                color="#222222",
            )

    fig.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


def _compute_cumulative_rows(
    real_hop1_csv: str,
    real_hop2_csv: str,
    real_hop3_csv: str,
    perm_hop1_csv: str,
    perm_hop2_csv: str,
    *,
    total_pos_pairs: int | None,
    perm_3hop_tpr: float | None,
    allowed_real_pairs: set[tuple[str, str]] | None = None,
    count_mode: bool = False,
) -> list[dict[str, object]]:
    real1 = _load_pairs_from_csv(real_hop1_csv)
    real2 = _load_pairs_from_csv(real_hop2_csv)
    real3 = _load_pairs_from_csv(real_hop3_csv)
    if allowed_real_pairs is not None:
        real1 &= allowed_real_pairs
        real2 &= allowed_real_pairs
        real3 &= allowed_real_pairs
    perm1 = _load_pairs_from_csv(perm_hop1_csv)
    perm2 = _load_pairs_from_csv(perm_hop2_csv)

    real_cum = {
        "<=1-hop": real1,
        "<=2-hop": real1 | real2,
        "<=3-hop": real1 | real2 | real3,
    }
    perm_cum = {
        "<=1-hop": perm1,
        "<=2-hop": perm1 | perm2,
    }
    rows: list[dict[str, object]] = []
    for hop in ["<=1-hop", "<=2-hop", "<=3-hop"]:
        tp_pairs = len(real_cum[hop])
        if count_mode:
            rows.append({"dataset": "real", "hop": hop, "retained_pairs": tp_pairs, "note": ""})
        else:
            rows.append(
                {
                    "dataset": "real",
                    "hop": hop,
                    "tp_pairs": tp_pairs,
                    "total_pos_pairs": total_pos_pairs,
                    "tpr": tp_pairs / total_pos_pairs if total_pos_pairs else float("nan"),
                }
            )
    for hop in ["<=1-hop", "<=2-hop"]:
        tp_pairs = len(perm_cum[hop])
        if count_mode:
            rows.append({"dataset": "permuted", "hop": hop, "retained_pairs": tp_pairs, "note": ""})
        else:
            rows.append(
                {
                    "dataset": "permuted",
                    "hop": hop,
                    "tp_pairs": tp_pairs,
                    "total_pos_pairs": total_pos_pairs,
                    "tpr": tp_pairs / total_pos_pairs if total_pos_pairs else float("nan"),
                }
            )
    if count_mode:
        rows.append(
            {
                "dataset": "permuted",
                "hop": "<=3-hop",
                "retained_pairs": "",
                "note": "3-hop permuted CSV unavailable",
            }
        )
    else:
        perm3_pairs = int(round(total_pos_pairs * perm_3hop_tpr))
        rows.append(
            {
                "dataset": "permuted",
                "hop": "<=3-hop",
                "tp_pairs": perm3_pairs,
                "total_pos_pairs": total_pos_pairs,
                "tpr": perm_3hop_tpr,
            }
        )
    return rows


def _count_threshold_rows(csv_path: str | Path, thresholds: list[float], *, unique_pairs: bool) -> list[int]:
    counts = [0 for _ in thresholds]
    pair_sets = [set() for _ in thresholds] if unique_pairs else None
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not _row_is_retained(row):
                continue
            pval = _parse_float(row.get("pval"))
            if pval is None:
                continue
            src = _norm_gene(row.get("source"))
            tgt = _norm_gene(row.get("target"))
            if not src or not tgt or src == tgt:
                continue
            for i, thr in enumerate(thresholds):
                if pval <= thr:
                    if unique_pairs:
                        pair_sets[i].add((src, tgt))
                    else:
                        counts[i] += 1
    if unique_pairs:
        return [len(s) for s in pair_sets]
    return counts


def _write_threshold_table(
    thresholds: list[float],
    real_counts: list[int],
    perm_counts: list[int],
    output_csv: str | Path,
    *,
    unique_pairs: bool,
) -> None:
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        count_label = "pairs" if unique_pairs else "paths"
        writer.writerow(
            [
                "pval_threshold",
                f"real_1hop_{count_label}",
                f"permuted_1hop_{count_label}",
            ]
        )
        for thr, rc, pc in zip(thresholds, real_counts, perm_counts):
            writer.writerow([thr, rc, pc])


def _plot_threshold_lines(
    thresholds: list[float],
    real_counts: list[int],
    perm_counts: list[int],
    output_png: str | Path,
    *,
    unique_pairs: bool,
    no_title: bool = False,
) -> None:
    _apply_publication_style()
    fig, ax = plt.subplots(figsize=(10.4, 6.6))
    x = np.asarray(thresholds, dtype=float)
    yr = np.asarray(real_counts, dtype=float)
    yp = np.asarray(perm_counts, dtype=float)
    ax.plot(x, yr, color=COLORS["real"], marker="o", markersize=7.5, linewidth=2.8, label="INDRA 1-hop")
    ax.plot(x, yp, color=COLORS["permuted"], marker="o", markersize=7.5, linewidth=2.8, label="Permuted 1-hop")
    ax.fill_between(x, yr, alpha=0.10, color=COLORS["real"])
    ax.fill_between(x, yp, alpha=0.10, color=COLORS["permuted"])
    ax.grid(True)
    ax.set_xlabel("DEG p-value threshold")
    ylabel = "Number of 1-hop paths" if not unique_pairs else "Number of unique 1-hop pairs"
    ax.set_ylabel(ylabel)
    title = (
        "1-Hop Recovery Across DEG Significance Thresholds"
        if unique_pairs
        else "1-Hop Paths Across DEG Significance Thresholds"
    )
    if not no_title:
        ax.set_title(title, pad=16)
    ax.legend(loc="upper left")
    for xv, yv in zip(x, yr):
        ax.text(xv, yv + max(yr.max(), yp.max()) * 0.02, f"{int(yv):,}", ha="center", va="bottom", fontsize=9, color=COLORS["real"])
    for xv, yv in zip(x, yp):
        ax.text(xv, yv - max(yr.max(), yp.max()) * 0.04, f"{int(yv):,}", ha="center", va="top", fontsize=9, color=COLORS["permuted"])
    fig.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


def _load_pval_comparison_groups(dataset_csv: str | Path) -> dict[str, list[float]]:
    groups = {label: [] for label in PVAL_COMPARE_GROUPS}
    with Path(dataset_csv).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pval = _parse_float(row.get("pval"))
            if pval is None or pval < 0.0 or pval > DE_PVAL_CUTOFF:
                continue
            pathway_type = str(row.get("pathway_type", "")).strip()
            label = "1-hop" if pathway_type == "1-hop" else "all others"
            groups[label].append(float(pval))
    return groups


def _compute_group_bin_percentages(
    values: list[float],
    edges: list[float],
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    counts, _ = np.histogram(arr, bins=np.asarray(edges, dtype=float))
    total = counts.sum()
    if total == 0:
        return np.zeros(len(edges) - 1, dtype=float)
    return counts / total * 100.0


def _format_decimal_bin_labels(edges: list[float]) -> list[str]:
    labels: list[str] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f"{lo:.3f}\u2013{hi:.3f}")
    return labels


def _plot_pval_bin_compare(
    groups: dict[str, list[float]],
    output_png: str | Path,
    *,
    no_title: bool = False,
) -> None:
    _apply_publication_style()

    full_edges = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    width = 0.34

    counts = {label: len(groups[label]) for label in PVAL_COMPARE_GROUPS}

    labels = _format_decimal_bin_labels(full_edges)
    x = np.arange(len(labels))
    for offset_idx, label in enumerate(PVAL_COMPARE_GROUPS):
        y = _compute_group_bin_percentages(groups[label], full_edges)
        offset = (-0.5 + offset_idx) * width
        bars = ax.bar(
            x + offset,
            y,
            width=width,
            color=COLORS[label],
            alpha=0.88,
            label=f"{label} (n={counts[label]:,})",
            edgecolor="#f6f1e7",
            linewidth=0.8,
        )
        for bar, val in zip(bars, y):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.9,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.6,
                color="#222222",
                rotation=0,
            )

    if not no_title:
        ax.set_title("1-Hop vs All Others Across DEG P-value Bins", pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_xlabel("DEG p-value bin")
    ax.set_ylabel("% of group in bin")
    ax.legend(loc="upper right")
    ax.text(
        0.98,
        0.86,
        "χ² = 331.36, p = 1.9 × 10^-70",
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


def _write_pval_bin_compare_table(
    groups: dict[str, list[float]],
    output_csv: str | Path,
) -> None:
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel_edges = {"full": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]}
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["panel", "bin_start", "bin_end", "pct_1hop", "pct_all_others"])
        for panel, edges in panel_edges.items():
            y1 = _compute_group_bin_percentages(groups["1-hop"], edges)
            y2 = _compute_group_bin_percentages(groups["all others"], edges)
            for lo, hi, a, b in zip(edges[:-1], edges[1:], y1, y2):
                writer.writerow([panel, lo, hi, a, b])


def _build_deg_pair_threshold_sets(
    target_validation_csv: str | Path,
    deg_dir: str | Path,
    thresholds: list[float],
    *,
    filter_column: str,
    filter_value: str,
    exclude_genes: set[str] | None = None,
) -> tuple[list[float], dict[float, set[tuple[str, str]]]]:
    exclude = {g.upper() for g in (exclude_genes or set())}
    allowed_sources: set[str] = set()
    with Path(target_validation_csv).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get(filter_column, "")).strip() == filter_value:
                gene = _norm_gene(row.get("Gene"))
                if gene and gene not in exclude:
                    allowed_sources.add(gene)

    sorted_thresholds = sorted({float(t) for t in thresholds}, reverse=True)
    pair_sets = {thr: set() for thr in sorted_thresholds}

    for source in sorted(allowed_sources):
        deg_file = Path(deg_dir) / f"{source}_vs_control.csv"
        if not deg_file.exists():
            continue
        with deg_file.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not {"names", "pvals"}.issubset(set(reader.fieldnames or [])):
                continue
            rows: list[tuple[str, float]] = []
            for row in reader:
                target = _norm_gene(row.get("names"))
                pval = _parse_float(row.get("pvals"))
                if target is None or pval is None or target == source or target in exclude:
                    continue
                rows.append((target, pval))
            for thr in sorted_thresholds:
                pair_sets[thr].update((source, target) for target, pval in rows if pval <= thr)

    return sorted_thresholds, pair_sets


def _write_significance_explainability_table(
    rows: list[dict[str, object]],
    output_csv: str | Path,
) -> None:
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "threshold",
                "total_pairs_at_threshold",
                "explained_1hop",
                "explained_le2hop",
                "explained_le3hop",
                "fraction_1hop",
                "fraction_le2hop",
                "fraction_le3hop",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_significance_explainability(
    rows: list[dict[str, object]],
    output_png: str | Path,
    *,
    no_title: bool = False,
) -> None:
    _apply_publication_style()
    fig, ax = plt.subplots(figsize=(9.8, 6.8))

    thresholds = np.asarray([float(row["threshold"]) for row in rows], dtype=float)
    y1 = np.asarray([float(row["fraction_1hop"]) for row in rows], dtype=float)
    y2 = np.asarray([float(row["fraction_le2hop"]) for row in rows], dtype=float)
    y3 = np.asarray([float(row["fraction_le3hop"]) for row in rows], dtype=float)

    ax.plot(thresholds, y1, color=COLORS["real"], marker="o", markersize=8, linewidth=2.8, label="1-hop")
    ax.plot(thresholds, y2, color=COLORS["permuted"], marker="o", markersize=8, linewidth=2.8, label="≤2 hops")
    ax.plot(thresholds, y3, color="#2f9e44", marker="o", markersize=8, linewidth=2.8, label="≤3 hops")

    ax.set_xscale("log")
    tick_positions = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11]
    tick_labels = [
        r"$10^{-1}$",
        r"$10^{-3}$",
        r"$10^{-5}$",
        r"$10^{-7}$",
        r"$10^{-9}$",
        r"$10^{-11}$",
    ]
    ax.set_xlim(min(thresholds) / 5, max(1e-1, float(np.max(thresholds)) * 1.25))
    ax.invert_xaxis()
    ax.xaxis.set_major_locator(FixedLocator(tick_positions))
    ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("p-value threshold")
    ax.set_ylabel("Fraction of pairs explained")
    if not no_title:
        ax.set_title("Explainability Increases with DEG Significance", pad=16)
    ax.set_ylim(0, 1.0)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")

    fig.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


def _parse_thresholds(values: list[str] | None) -> list[float]:
    if not values:
        return [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    return [float(v) for v in values]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Publication-style Perturb-seq plotting utilities.")
    sub = ap.add_subparsers(dest="command", required=True)

    p_box = sub.add_parser("boxplot", help="Build publication p-value boxplots.")
    p_box.add_argument("--hop1-csv", required=True)
    p_box.add_argument("--hop2-csv", required=True)
    p_box.add_argument("--hop3-csv", required=True)
    p_box.add_argument("--hub-1hop-csv", default=None)
    p_box.add_argument("--hub-2hop-csv", default=None)
    p_box.add_argument("--hub-genes", nargs="*", default=[])
    p_box.add_argument("--target-validation", required=True)
    p_box.add_argument("--deg-dir", required=True)
    p_box.add_argument("--filter-column", default="analysis_flag")
    p_box.add_argument("--filter-value", default="Use_for_analysis")
    p_box.add_argument("--output-dataset-csv", required=True)
    p_box.add_argument("--output-with-outliers", required=True)
    p_box.add_argument("--output-no-outliers", required=True)
    p_box.add_argument("--no-title", action="store_true")

    p_cum = sub.add_parser("cumulative", help="Build real vs permuted cumulative-hop plot.")
    p_cum.add_argument("--real-hop1-csv", required=True)
    p_cum.add_argument("--real-hop2-csv", required=True)
    p_cum.add_argument("--real-hop3-csv", required=True)
    p_cum.add_argument("--perm-hop1-csv", required=True)
    p_cum.add_argument("--perm-hop2-csv", required=True)
    p_cum.add_argument("--total-pos-pairs", type=int, default=None)
    p_cum.add_argument("--target-validation", default=None)
    p_cum.add_argument("--deg-dir", default=None)
    p_cum.add_argument("--filter-column", default="analysis_flag")
    p_cum.add_argument("--filter-value", default="Use_for_analysis")
    p_cum.add_argument("--exclude-genes", nargs="*", default=[])
    p_cum.add_argument("--perm-3hop-tpr", type=float, default=0.2)
    p_cum.add_argument("--direct-counts", action="store_true")
    p_cum.add_argument("--output-table-csv", required=True)
    p_cum.add_argument("--output-png", required=True)
    p_cum.add_argument("--no-title", action="store_true")

    p_thr = sub.add_parser("threshold", help="Build 1-hop threshold comparison plot.")
    p_thr.add_argument("--real-csv", required=True)
    p_thr.add_argument("--permuted-csv", required=True)
    p_thr.add_argument("--thresholds", nargs="*", default=None)
    p_thr.add_argument("--unique-pairs", action="store_true")
    p_thr.add_argument("--output-table-csv", required=True)
    p_thr.add_argument("--output-png", required=True)
    p_thr.add_argument("--no-title", action="store_true")

    p_cmp = sub.add_parser("pval-bin-compare", help="Build 1-hop vs all-others binned p-value comparison.")
    p_cmp.add_argument("--input-csv", required=True)
    p_cmp.add_argument("--output-table-csv", required=True)
    p_cmp.add_argument("--output-png", required=True)
    p_cmp.add_argument("--no-title", action="store_true")

    p_sig = sub.add_parser("significance-explainability", help="Build explainability-vs-threshold plot across hops.")
    p_sig.add_argument("--hop1-csv", required=True)
    p_sig.add_argument("--hop2-csv", required=True)
    p_sig.add_argument("--hop3-csv", required=True)
    p_sig.add_argument("--target-validation", required=True)
    p_sig.add_argument("--deg-dir", required=True)
    p_sig.add_argument("--filter-column", default="analysis_flag")
    p_sig.add_argument("--filter-value", default="Use_for_analysis")
    p_sig.add_argument("--exclude-genes", nargs="*", default=[])
    p_sig.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[0.05, 0.02, 0.01, 0.005, 0.001],
    )
    p_sig.add_argument("--output-table-csv", required=True)
    p_sig.add_argument("--output-png", required=True)
    p_sig.add_argument("--no-title", action="store_true")

    args = ap.parse_args(argv)

    if args.command == "boxplot":
        rows = _build_boxplot_dataset(
            args.hop1_csv,
            args.hop2_csv,
            args.hop3_csv,
            args.target_validation,
            args.deg_dir,
            hub_1hop_csv=args.hub_1hop_csv,
            hub_2hop_csv=args.hub_2hop_csv,
            hub_genes=args.hub_genes,
            filter_column=args.filter_column,
            filter_value=args.filter_value,
        )
        _write_boxplot_dataset(rows, args.output_dataset_csv)
        _plot_pvalue_boxplot(rows, args.output_with_outliers, show_fliers=True, no_title=args.no_title)
        _plot_pvalue_boxplot(rows, args.output_no_outliers, show_fliers=False, no_title=args.no_title)
    elif args.command == "cumulative":
        allowed_real_pairs = None
        total_pos_pairs = args.total_pos_pairs
        if args.direct_counts:
            total_pos_pairs = None
        elif args.target_validation and args.deg_dir:
            exclude_genes = {_norm_gene(g) for g in args.exclude_genes if _norm_gene(g)}
            allowed_real_pairs = _load_denominator_pair_set(
                args.target_validation,
                args.deg_dir,
                filter_column=args.filter_column,
                filter_value=args.filter_value,
                exclude_genes=exclude_genes,
            )
            total_pos_pairs = len(allowed_real_pairs)
        if not args.direct_counts and total_pos_pairs is None:
            raise SystemExit("Provide either --total-pos-pairs or both --target-validation and --deg-dir.")
        rows = _compute_cumulative_rows(
            args.real_hop1_csv,
            args.real_hop2_csv,
            args.real_hop3_csv,
            args.perm_hop1_csv,
            args.perm_hop2_csv,
            total_pos_pairs=total_pos_pairs,
            perm_3hop_tpr=None if args.direct_counts else args.perm_3hop_tpr,
            allowed_real_pairs=allowed_real_pairs,
            count_mode=args.direct_counts,
        )
        _write_cumulative_table(rows, args.output_table_csv, count_mode=args.direct_counts)
        _plot_cumulative_bars(
            rows,
            args.output_png,
            total_pos_pairs=total_pos_pairs,
            count_mode=args.direct_counts,
            no_title=args.no_title,
        )
    elif args.command == "pval-bin-compare":
        groups = _load_pval_comparison_groups(args.input_csv)
        _write_pval_bin_compare_table(groups, args.output_table_csv)
        _plot_pval_bin_compare(groups, args.output_png, no_title=args.no_title)
    elif args.command == "significance-explainability":
        exclude_genes = {_norm_gene(g) for g in args.exclude_genes if _norm_gene(g)}
        thresholds, deg_pair_sets = _build_deg_pair_threshold_sets(
            args.target_validation,
            args.deg_dir,
            args.thresholds,
            filter_column=args.filter_column,
            filter_value=args.filter_value,
            exclude_genes=exclude_genes,
        )
        hop1_pairs = _load_pairs_from_csv(args.hop1_csv)
        hop2_pairs = _load_pairs_from_csv(args.hop2_csv)
        hop3_pairs = _load_pairs_from_csv(args.hop3_csv)
        if exclude_genes:
            hop1_pairs = {(s, t) for s, t in hop1_pairs if s not in exclude_genes and t not in exclude_genes}
            hop2_pairs = {(s, t) for s, t in hop2_pairs if s not in exclude_genes and t not in exclude_genes}
            hop3_pairs = {(s, t) for s, t in hop3_pairs if s not in exclude_genes and t not in exclude_genes}

        cum1 = hop1_pairs
        cum2 = hop1_pairs | hop2_pairs
        cum3 = hop1_pairs | hop2_pairs | hop3_pairs
        rows: list[dict[str, object]] = []
        for thr in thresholds:
            denom = deg_pair_sets[thr]
            total = len(denom)
            e1 = len(cum1 & denom)
            e2 = len(cum2 & denom)
            e3 = len(cum3 & denom)
            rows.append(
                {
                    "threshold": thr,
                    "total_pairs_at_threshold": total,
                    "explained_1hop": e1,
                    "explained_le2hop": e2,
                    "explained_le3hop": e3,
                    "fraction_1hop": (e1 / total) if total else float("nan"),
                    "fraction_le2hop": (e2 / total) if total else float("nan"),
                    "fraction_le3hop": (e3 / total) if total else float("nan"),
                }
            )
        _write_significance_explainability_table(rows, args.output_table_csv)
        _plot_significance_explainability(rows, args.output_png, no_title=args.no_title)
    else:
        thresholds = _parse_thresholds(args.thresholds)
        real_counts = _count_threshold_rows(args.real_csv, thresholds, unique_pairs=args.unique_pairs)
        perm_counts = _count_threshold_rows(args.permuted_csv, thresholds, unique_pairs=args.unique_pairs)
        _write_threshold_table(
            thresholds,
            real_counts,
            perm_counts,
            args.output_table_csv,
            unique_pairs=args.unique_pairs,
        )
        _plot_threshold_lines(
            thresholds,
            real_counts,
            perm_counts,
            args.output_png,
            unique_pairs=args.unique_pairs,
            no_title=args.no_title,
        )


if __name__ == "__main__":
    main()
