"""Compare original vs permuted INDRA path data.

Consolidated module that provides two analyses:

1. **Pair-overlap analysis** (from ``compare_original_vs_permuted.py``):
   Removes self-loops from real and permuted 1/2-hop path files, counts
   unique source-target pairs, and reports coverage rates.

2. **Source-shuffle comparison** (from ``comapare_with_permuted_data.py``):
   Permutes source labels in an original paths CSV, removes accidental
   true matches, computes sign-alignment and evidence statistics, and
   compares original vs permuted distributions.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Pair-overlap helpers
# ------------------------------------------------------------------

def remove_self_loops(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* without rows where source == target.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``source`` and ``target`` columns.

    Returns
    -------
    pd.DataFrame
    """
    before = len(df)
    out = df[df["source"] != df["target"]].copy()
    logger.info("Removed %d self-loops (%d -> %d rows)",
                before - len(out), before, len(out))
    return out


def unique_pairs(df: pd.DataFrame) -> set[tuple[str, str]]:
    """Return unique (source, target) pairs from *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``source`` and ``target`` columns.

    Returns
    -------
    set[tuple[str, str]]
    """
    return set(zip(df["source"], df["target"]))


def coverage_rate(
    explained: set[tuple[str, str]],
    total_pool: set[tuple[str, str]],
) -> float:
    """Fraction of *total_pool* pairs present in *explained*.

    Parameters
    ----------
    explained : set[tuple[str, str]]
    total_pool : set[tuple[str, str]]

    Returns
    -------
    float
    """
    if not total_pool:
        return float("nan")
    return len(explained & total_pool) / len(total_pool)


def pair_overlap_analysis(
    original_csv: str,
    permuted_csv: str,
) -> pd.DataFrame:
    """Compare unique pair counts and overlap between original and permuted.

    Parameters
    ----------
    original_csv : str
        CSV with ``source`` and ``target`` columns (original paths).
    permuted_csv : str
        CSV with ``source`` and ``target`` columns (permuted paths).

    Returns
    -------
    pd.DataFrame
        Summary table with pair counts, overlap, and Jaccard index.
    """
    orig = remove_self_loops(pd.read_csv(original_csv, low_memory=False))
    perm = remove_self_loops(pd.read_csv(permuted_csv, low_memory=False))

    orig_pairs = unique_pairs(orig)
    perm_pairs = unique_pairs(perm)
    overlap = orig_pairs & perm_pairs

    jaccard = (
        len(overlap) / len(orig_pairs | perm_pairs)
        if (orig_pairs | perm_pairs)
        else float("nan")
    )

    rows = [
        {
            "dataset": "original",
            "unique_pairs": len(orig_pairs),
            "overlap_with_other": len(overlap),
            "jaccard": jaccard,
        },
        {
            "dataset": "permuted",
            "unique_pairs": len(perm_pairs),
            "overlap_with_other": len(overlap),
            "jaccard": jaccard,
        },
    ]
    result = pd.DataFrame(rows)
    logger.info("Pair-overlap summary:\n%s", result.to_string(index=False))
    return result


# ------------------------------------------------------------------
# Source-shuffle comparison helpers
# ------------------------------------------------------------------

def _check_sign_alignment(row: pd.Series) -> str:
    """Determine whether the INDRA statement direction matches logFC.

    Parameters
    ----------
    row : pd.Series
        Row with ``logfoldchange`` (or ``logfoldchanges``), ``hop_number``,
        and per-hop ``stmt_type_*`` columns.

    Returns
    -------
    str
        ``"Yes"`` if aligned, ``"No"`` if misaligned, ``""`` if not
        determinable.
    """
    logfc = row.get("logfoldchange", row.get("logfoldchanges"))
    if logfc is None or pd.isna(logfc):
        return ""

    hop = row.get("hop_number", "")
    hop_map = {"1hop": "stmt_type_1", "2hop": "stmt_type_2",
               "3hop": "stmt_type_3", "4hop": "stmt_type_4"}
    stmt_col = hop_map.get(str(hop), "")
    stmt_type = row.get(stmt_col, "")
    if pd.isna(stmt_type) or not str(stmt_type).strip():
        return ""

    stmt_type = str(stmt_type).strip()
    if "DecreaseAmount" in stmt_type or "Inhibition" in stmt_type:
        return "Yes" if logfc > 0 else "No"
    if "IncreaseAmount" in stmt_type or "Activation" in stmt_type:
        return "Yes" if logfc < 0 else "No"
    return ""


def _compute_comparison_stats(df: pd.DataFrame, label: str) -> dict:
    """Compute summary statistics for a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Path-level DataFrame with evidence, belief, logfoldchange, pval
        columns.
    label : str
        Human-readable dataset label.

    Returns
    -------
    dict
    """
    stats: dict = {"dataset": label, "n_rows": len(df)}

    evidence_cols = [c for c in ("evidence_1", "evidence_2",
                                 "evidence_3", "evidence_4")
                     if c in df.columns]
    for c in evidence_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if evidence_cols:
        df["total_evidence"] = df[evidence_cols].sum(axis=1)
        stats["mean_total_evidence"] = df["total_evidence"].mean()
        stats["median_total_evidence"] = df["total_evidence"].median()

    belief_cols = [c for c in ("belief_1", "belief_2",
                               "belief_3", "belief_4")
                   if c in df.columns]
    for c in belief_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if belief_cols:
        vals = df[belief_cols].replace(0, np.nan).values.flatten()
        vals = vals[~np.isnan(vals)]
        stats["mean_belief"] = vals.mean() if len(vals) else 0.0
        stats["median_belief"] = float(np.median(vals)) if len(vals) else 0.0

    lfc_col = next(
        (c for c in ("logfoldchange", "logfoldchanges") if c in df.columns),
        None,
    )
    if lfc_col:
        stats["mean_abs_logfc"] = df[lfc_col].abs().mean()
    pval_col = next(
        (c for c in ("pval", "pvals") if c in df.columns), None,
    )
    if pval_col:
        stats["mean_pval"] = df[pval_col].mean()

    if "sign_aligned" in df.columns:
        checkable = df[df["sign_aligned"].isin(["Yes", "No"])]
        n_aligned = int((checkable["sign_aligned"] == "Yes").sum())
        n_total = len(checkable)
        stats["n_aligned"] = n_aligned
        stats["n_total_checkable"] = n_total
        stats["sign_alignment_rate"] = (
            n_aligned / n_total if n_total else 0.0
        )

    return stats


def source_shuffle_comparison(
    original_csv: str,
    output_csv: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Permute source labels and compare statistics with the original.

    Parameters
    ----------
    original_csv : str
        CSV with ``source``, ``target``, and evidence/belief columns.
    output_csv : str
        Path to write the comparison statistics CSV.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Two-row DataFrame (original, permuted) with summary statistics.
    """
    df_orig = pd.read_csv(original_csv, low_memory=False)
    true_pairs = set(zip(df_orig["source"], df_orig["target"]))
    logger.info("Loaded %d original rows (%d unique pairs)",
                len(df_orig), len(true_pairs))

    rng = np.random.default_rng(seed)
    df_perm = df_orig.copy()
    df_perm["source"] = rng.permutation(df_perm["source"].values)

    mask = df_perm.apply(
        lambda r: (r["source"], r["target"]) not in true_pairs, axis=1,
    )
    n_removed = int((~mask).sum())
    df_perm = df_perm[mask].reset_index(drop=True)
    logger.info("Removed %d accidental true matches after permutation",
                n_removed)

    df_orig["sign_aligned"] = df_orig.apply(_check_sign_alignment, axis=1)
    df_perm["sign_aligned"] = df_perm.apply(_check_sign_alignment, axis=1)

    orig_stats = _compute_comparison_stats(df_orig.copy(), "original")
    perm_stats = _compute_comparison_stats(df_perm.copy(), "permuted")

    comparison = pd.DataFrame([orig_stats, perm_stats])
    comparison.to_csv(output_csv, index=False)
    logger.info("Wrote comparison: %s", output_csv)
    logger.info("\n%s", comparison.to_string(index=False))
    return comparison


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Compare original vs permuted INDRA path data.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # pair-overlap subcommand
    p_overlap = sub.add_parser(
        "pair-overlap",
        help="Count unique pairs and overlap between original and permuted.",
    )
    p_overlap.add_argument("--original", required=True,
                           help="Original paths CSV")
    p_overlap.add_argument("--permuted", required=True,
                           help="Permuted paths CSV")
    p_overlap.add_argument("--output", required=True,
                           help="Output summary CSV")

    # source-shuffle subcommand
    p_shuffle = sub.add_parser(
        "source-shuffle",
        help="Shuffle source labels and compare evidence/belief statistics.",
    )
    p_shuffle.add_argument("--original", required=True,
                           help="Original paths CSV")
    p_shuffle.add_argument("--output", required=True,
                           help="Output comparison CSV")
    p_shuffle.add_argument("--seed", type=int, default=42)

    args = ap.parse_args(argv)

    if args.command == "pair-overlap":
        result = pair_overlap_analysis(args.original, args.permuted)
        result.to_csv(args.output, index=False)
        logger.info("Wrote %s", args.output)

    elif args.command == "source-shuffle":
        source_shuffle_comparison(args.original, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
