"""Compute TP/FP/TN/FN and TPR/FPR across p-value thresholds.

Works for any hop count. Requires:
- A path CSV with source/target columns
- target_validation_expanded.csv (for Karen-filtered sources)
- DEG directory with <SOURCE>_vs_control.csv files
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from indra_perturbseq.deg import build_pvalue_map, deg_path_for_source, pick_sig_column
from indra_perturbseq.gene_lists import load_karen_sources
from indra_perturbseq.hgnc import normalize_hgnc_symbol

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001]


def load_explained_pairs(
    path_csv: str,
    source_col: str = "source",
    target_col: str = "target",
) -> set[tuple[str, str]]:
    """Load unique (source, target) pairs from a path CSV."""
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
    return set(zip(pairs["source"], pairs["target"]))


def compute_tpr_fpr(
    sources: list[str],
    explained_pairs: set[tuple[str, str]],
    de_dir: str,
    thresholds: list[float],
) -> pd.DataFrame:
    """Compute per-threshold confusion-matrix statistics.

    Returns a DataFrame with one row per threshold.
    """
    expl_by_src: dict[str, set[str]] = defaultdict(set)
    for s, t in explained_pairs:
        expl_by_src[s].add(t)

    cache: dict[str, tuple[np.ndarray, dict]] = {}
    for src in sources:
        f = deg_path_for_source(de_dir, src)
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

        fn_all = pos_all - tp_all
        tn_all = neg_all - fp_all
        tpr = tp_all / pos_all if pos_all else float("nan")
        fpr = fp_all / neg_all if neg_all else float("nan")

        rows.append({
            "threshold": float(thr),
            "TP_total": tp_all, "FP_total": fp_all,
            "TN_total": tn_all, "FN_total": fn_all,
            "positives_total": pos_all, "negatives_total": neg_all,
            "TPR_overall": tpr, "FPR_overall": fpr,
        })
        logger.info(
            "thr %g | TP=%d FP=%d TN=%d FN=%d | TPR=%.4f FPR=%.4f",
            thr, tp_all, fp_all, tn_all, fn_all, tpr, fpr,
        )

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Compute TP/FP/TN/FN + TPR/FPR for a path CSV.",
    )
    ap.add_argument("--paths-csv", required=True)
    ap.add_argument("--tv-path", required=True,
                    help="target_validation_expanded.csv")
    ap.add_argument("--de-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--tv-source-col", default="Gene")
    ap.add_argument("--karen-flag-col", default="Karen_Flag")
    ap.add_argument("--karen-flag-value", default="Use_for_analysis")
    ap.add_argument("--path-source-col", default="source")
    ap.add_argument("--path-target-col", default="target")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=DEFAULT_THRESHOLDS)
    args = ap.parse_args(argv)

    sources = load_karen_sources(
        args.tv_path,
        source_col=args.tv_source_col,
        flag_col=args.karen_flag_col,
        flag_value=args.karen_flag_value,
    )
    logger.info("Karen sources: %d", len(sources))

    pairs = load_explained_pairs(
        args.paths_csv, args.path_source_col, args.path_target_col,
    )
    logger.info("Explained pairs: %d", len(pairs))

    result = compute_tpr_fpr(sources, pairs, args.de_dir, args.thresholds)
    result.to_csv(args.out_csv, index=False)
    logger.info("Wrote %s", args.out_csv)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
