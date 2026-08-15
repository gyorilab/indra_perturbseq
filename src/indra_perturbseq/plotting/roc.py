"""ROC curve computation and plotting from threshold-based TPR/FPR."""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from indra_perturbseq.deg import build_pvalue_map, deg_path_for_source
from indra_perturbseq.gene_lists import load_filtered_sources
from indra_perturbseq.hgnc import normalize_hgnc_symbol

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001]

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
    "font.size": 11,
    "savefig.dpi": 300,
})


def _load_explained_pairs(path: str) -> set[tuple[str, str]]:
    df = pd.read_csv(path, low_memory=False)
    if not {"source", "target"}.issubset(df.columns):
        raise ValueError(f"{path} missing source/target")
    df["source"] = df["source"].astype(str).str.strip().map(normalize_hgnc_symbol)
    df["target"] = df["target"].astype(str).str.strip().map(normalize_hgnc_symbol)
    df = df.dropna(subset=["source", "target"])
    df = df[df["source"] != df["target"]]
    return set(zip(df["source"], df["target"]))


def _auc_trapezoid(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan")
    order = np.argsort(x)
    return float(np.trapezoid(y[order], x[order]))


def plot_roc(fprs, tprs, title: str, out_path: str) -> float:
    """Plot an ROC curve from arrays of FPR / TPR values."""
    x, y = np.asarray(fprs, float), np.asarray(tprs, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    order = np.argsort(x)
    x, y = x[order], y[order]
    auc = _auc_trapezoid(x, y)

    dx, dy = x.ptp(), y.ptp()
    pad_x = max(0.005, 0.15 * dx if dx > 0 else 0.01)
    pad_y = max(0.01, 0.15 * dy if dy > 0 else 0.02)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, y, marker="o", markersize=6, label=f"AUC = {auc:.3f}")
    ax.plot([max(0, x.min() - pad_x), min(1, x.max() + pad_x)],
            [max(0, x.min() - pad_x), min(1, x.max() + pad_x)],
            "r--", linewidth=1.2, label="Chance")
    ax.set_xlim(max(0, x.min() - pad_x), min(1, x.max() + pad_x))
    ax.set_ylim(max(0, y.min() - pad_y), min(1, y.max() + pad_y))
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("ROC plot saved: %s (AUC=%.3f)", out_path, auc)
    return auc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Compute ROC curves from threshold-based TPR/FPR.",
    )
    ap.add_argument("--target-validation", required=True)
    ap.add_argument("--deg-dir", required=True)
    ap.add_argument("--pos-paths", nargs="+", required=True,
                    help="Positive path CSVs (e.g. 1hop_main, 2hop_main)")
    ap.add_argument("--neg-paths", nargs="+", required=True,
                    help="Negative path CSVs (e.g. fp_1hop, fp_2hop)")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-roc-overall", required=True)
    ap.add_argument("--output-roc-per-source", required=True)
    ap.add_argument("--source-column", default="Gene")
    ap.add_argument("--filter-column", default="analysis_flag")
    ap.add_argument("--filter-value", default="Use_for_analysis")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=DEFAULT_THRESHOLDS)
    args = ap.parse_args(argv)

    sources = load_filtered_sources(
        args.target_validation, args.source_column,
        args.filter_column, args.filter_value,
    )

    explained: set[tuple[str, str]] = set()
    for p in args.pos_paths + args.neg_paths:
        explained |= _load_explained_pairs(p)
    logger.info("Explained pairs: %d", len(explained))

    expl_by_src: dict[str, set[str]] = defaultdict(set)
    for s, t in explained:
        expl_by_src[s].add(t)

    cache: dict[str, tuple[np.ndarray, dict]] = {}
    for src in sources:
        f = deg_path_for_source(args.deg_dir, src)
        if not os.path.exists(f):
            continue
        pmap = build_pvalue_map(f, src)
        if pmap:
            cache[src] = (np.fromiter(pmap.values(), dtype=float), pmap)

    overall_tprs, overall_fprs = [], []
    persrc_tprs, persrc_fprs = [], []
    rows = []

    for thr in args.thresholds:
        tp_all = fp_all = pos_all = neg_all = 0
        src_tpr, src_fpr = [], []

        for src, (pvals, pmap) in cache.items():
            pos = int(np.sum(pvals <= thr))
            neg = int(np.sum(pvals > thr))
            tp = fp = 0
            for t in expl_by_src.get(src, set()):
                p = pmap.get(t)
                if p is None:
                    continue
                if p <= thr:
                    tp += 1
                else:
                    fp += 1
            tp_all += tp
            fp_all += fp
            pos_all += pos
            neg_all += neg
            src_tpr.append(tp / pos if pos else float("nan"))
            src_fpr.append(fp / neg if neg else float("nan"))

        tpr_o = tp_all / pos_all if pos_all else float("nan")
        fpr_o = fp_all / neg_all if neg_all else float("nan")
        overall_tprs.append(tpr_o)
        overall_fprs.append(fpr_o)
        persrc_tprs.append(float(np.nanmean(src_tpr)))
        persrc_fprs.append(float(np.nanmean(src_fpr)))

        rows.append({
            "threshold": float(thr),
            "TP_total": tp_all, "FP_total": fp_all,
            "positives_total": pos_all, "negatives_total": neg_all,
            "TPR_overall": tpr_o, "FPR_overall": fpr_o,
            "TPR_per_source_avg": persrc_tprs[-1],
            "FPR_per_source_avg": persrc_fprs[-1],
        })

    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    logger.info("Stats CSV: %s", args.output_csv)

    os.makedirs(os.path.dirname(args.output_roc_overall) or ".", exist_ok=True)
    plot_roc(overall_fprs, overall_tprs,
             "ROC curve (overall rate)", args.output_roc_overall)
    plot_roc(persrc_fprs, persrc_tprs,
             "ROC curve (per-source average)", args.output_roc_per_source)


if __name__ == "__main__":
    main()
