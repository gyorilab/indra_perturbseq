"""Compute TP/FP/TN/FN and TPR/FPR across p-value thresholds.

Works for any hop count. Requires:
- One or more path CSVs with source/target columns (union is taken)
- target_validation_expanded.csv (for filtered sources)
- DEG directory with <SOURCE>_vs_control.csv files
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from indra_perturbseq.deg import build_pvalue_map, deg_path_for_source
from indra_perturbseq.gene_lists import load_filtered_sources
from indra_perturbseq.hgnc import normalize_hgnc_symbol

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001]


def load_explained_pairs(
    path_csvs: list[str],
    source_col: str = "source",
    target_col: str = "target",
) -> set[tuple[str, str]]:
    """Load and union unique (source, target) pairs from path CSVs."""
    explained: set[tuple[str, str]] = set()
    for path_csv in path_csvs:
        df = pd.read_csv(path_csv, low_memory=False)
        if not {source_col, target_col}.issubset(df.columns):
            raise ValueError(
                f"{path_csv} missing {source_col}/{target_col}. "
                f"Has: {df.columns.tolist()}"
            )
        s = df[source_col].astype(str).str.strip().map(normalize_hgnc_symbol)
        t = df[target_col].astype(str).str.strip().map(normalize_hgnc_symbol)
        pairs = pd.DataFrame({"source": s, "target": t}).dropna()
        pairs = pairs[pairs["source"] != pairs["target"]]
        explained |= set(zip(pairs["source"], pairs["target"]))
    return explained


def compute_tpr_fpr(
    sources: list[str],
    explained_pairs: set[tuple[str, str]],
    deg_dir: str,
    thresholds: list[float],
) -> pd.DataFrame:
    """Compute per-threshold confusion-matrix statistics.

    Returns a DataFrame with one row per threshold, including both
    overall and per-source-averaged TPR/FPR.
    """
    expl_by_src: dict[str, set[str]] = defaultdict(set)
    for s, t in explained_pairs:
        expl_by_src[s].add(t)

    cache: dict[str, tuple[np.ndarray, dict]] = {}
    for src in sources:
        f = deg_path_for_source(deg_dir, src)
        if not os.path.exists(f):
            continue
        pmap = build_pvalue_map(f, src)
        if pmap:
            cache[src] = (np.fromiter(pmap.values(), dtype=float), pmap)
    logger.info("Cached %d sources, %d missing/unusable",
                len(cache), len(sources) - len(cache))

    rows = []
    for thr in thresholds:
        tp_all = fp_all = pos_all = neg_all = 0
        tpr_list: list[float] = []
        fpr_list: list[float] = []

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
            tpr_list.append(tp / pos if pos else np.nan)
            fpr_list.append(fp / neg if neg else np.nan)

        fn_all = pos_all - tp_all
        tn_all = neg_all - fp_all
        tpr = tp_all / pos_all if pos_all else float("nan")
        fpr = fp_all / neg_all if neg_all else float("nan")
        tpr_avg = float(np.nanmean(tpr_list)) if tpr_list else float("nan")
        fpr_avg = float(np.nanmean(fpr_list)) if fpr_list else float("nan")

        rows.append({
            "threshold": float(thr),
            "TP_total": tp_all, "FP_total": fp_all,
            "TN_total": tn_all, "FN_total": fn_all,
            "positives_total": pos_all, "negatives_total": neg_all,
            "TPR_overall": tpr, "FPR_overall": fpr,
            "TPR_per_source_avg": tpr_avg, "FPR_per_source_avg": fpr_avg,
        })
        logger.info(
            "thr %g | TP=%d FP=%d TN=%d FN=%d | TPR=%.4f FPR=%.4f | avg TPR=%.4f FPR=%.4f",
            thr, tp_all, fp_all, tn_all, fn_all, tpr, fpr, tpr_avg, fpr_avg,
        )

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Compute TP/FP/TN/FN + TPR/FPR for path CSVs.",
    )
    ap.add_argument("--paths", nargs="+", required=True,
                    help="One or more path CSVs to union")
    ap.add_argument("--target-validation", required=True,
                    help="target_validation_expanded.csv")
    ap.add_argument("--deg-dir", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--source-column", default="Gene")
    ap.add_argument("--filter-column", default="analysis_flag")
    ap.add_argument("--filter-value", default="Use_for_analysis")
    ap.add_argument("--path-source-col", default="source")
    ap.add_argument("--path-target-col", default="target")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=DEFAULT_THRESHOLDS)
    args = ap.parse_args(argv)

    sources = load_filtered_sources(
        args.target_validation,
        source_col=args.source_column,
        filter_column=args.filter_column,
        filter_value=args.filter_value,
    )
    logger.info("Filtered sources: %d", len(sources))

    pairs = load_explained_pairs(
        args.paths, args.path_source_col, args.path_target_col,
    )
    logger.info("Explained pairs: %d", len(pairs))

    result = compute_tpr_fpr(sources, pairs, args.deg_dir, args.thresholds)
    result.to_csv(args.output_csv, index=False)
    logger.info("Wrote %s", args.output_csv)
    logger.info("\n%s", result.to_string(index=False))


if __name__ == "__main__":
    main()
