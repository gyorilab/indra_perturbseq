"""Compare real vs permuted 1-hop path counts across p-value thresholds.

Loads real and permuted 1-hop CSVs, counts paths (or unique source-target
pairs) at each threshold, and outputs a summary table.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]


def _find_pval_column(df: pd.DataFrame) -> str:
    """Detect the p-value column name in *df*.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    str
        Column name.

    Raises
    ------
    ValueError
        If no p-value column is found.
    """
    candidates = ("pval", "pvals", "PVAL", "PVALS", "pvals_adj")
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"No p-value column found. Columns: {df.columns.tolist()}"
    )


def count_paths(
    df: pd.DataFrame,
    pval_col: str,
    threshold: float,
    *,
    unique_pairs: bool = False,
) -> int:
    """Count paths (rows or unique pairs) at a given p-value threshold.

    Parameters
    ----------
    df : pd.DataFrame
    pval_col : str
        Name of the p-value column.
    threshold : float
        Keep rows with p-value <= threshold.
    unique_pairs : bool
        If True and ``source``/``target`` columns exist, count distinct
        (source, target) pairs instead of rows.

    Returns
    -------
    int
    """
    sub = df[df[pval_col] <= threshold]
    if unique_pairs and {"source", "target"}.issubset(sub.columns):
        return sub.drop_duplicates(subset=["source", "target"]).shape[0]
    return sub.shape[0]


def compare_at_thresholds(
    real_csv: str,
    permuted_csv: str,
    thresholds: list[float] | None = None,
    *,
    unique_pairs: bool = False,
) -> pd.DataFrame:
    """Count real vs permuted 1-hop paths at each p-value threshold.

    Parameters
    ----------
    real_csv : str
        Real 1-hop paths CSV.
    permuted_csv : str
        Permuted 1-hop paths CSV.
    thresholds : list[float] or None
        P-value thresholds to evaluate. Defaults to
        ``DEFAULT_THRESHOLDS``.
    unique_pairs : bool
        Count unique (source, target) pairs instead of rows.

    Returns
    -------
    pd.DataFrame
        One row per threshold with real and permuted counts.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    real = pd.read_csv(real_csv, low_memory=False)
    perm = pd.read_csv(permuted_csv, low_memory=False)

    real_pcol = _find_pval_column(real)
    perm_pcol = _find_pval_column(perm)
    real[real_pcol] = pd.to_numeric(real[real_pcol], errors="coerce")
    perm[perm_pcol] = pd.to_numeric(perm[perm_pcol], errors="coerce")

    rows: list[dict] = []
    for thr in thresholds:
        rc = count_paths(real, real_pcol, thr, unique_pairs=unique_pairs)
        pc = count_paths(perm, perm_pcol, thr, unique_pairs=unique_pairs)
        rows.append({
            "pval_threshold": thr,
            "real_1hop_paths": rc,
            "permuted_1hop_paths": pc,
        })
        logger.info("threshold=%.4f | real=%d  permuted=%d", thr, rc, pc)

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Compare real vs permuted 1-hop path counts across "
                    "p-value thresholds.",
    )
    ap.add_argument("--real-csv", required=True,
                    help="Real 1-hop paths CSV")
    ap.add_argument("--permuted-csv", required=True,
                    help="Permuted 1-hop paths CSV")
    ap.add_argument("--output-csv", required=True,
                    help="Output comparison CSV")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=DEFAULT_THRESHOLDS)
    ap.add_argument("--unique-pairs", action="store_true",
                    help="Count unique (source, target) pairs instead of rows")
    args = ap.parse_args(argv)

    result = compare_at_thresholds(
        args.real_csv, args.permuted_csv, args.thresholds,
        unique_pairs=args.unique_pairs,
    )
    result.to_csv(args.output_csv, index=False)
    logger.info("Wrote %s", args.output_csv)
    logger.info("\n%s", result.to_string(index=False))


if __name__ == "__main__":
    main()
