"""Bulk RNA-seq differential expression analysis using limma-trend (InMoose).
Each knockdown condition (2 replicates) is compared against negative-control.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NEG_CTRL_COLS: list[str] = [
    "TeloHAEC_B6_neg_ctrl_KD",
    "TeloHAEC_D2_neg_ctrl_KD",
    "TeloHAEC_D8_neg_ctrl_KD",
    "TeloHAEC_F4_neg_ctrl_KD",
]

GENE_TO_KD_COLS: dict[str, list[str]] = {
    "MAP2K5":   ["TeloHAEC_A11_MAP2K5_KD",  "TeloHAEC_B5_MAP2K5_KD"],
    "MAP3K3":   ["TeloHAEC_A4_MAP3K3_KD",   "TeloHAEC_C3_MAP3K3_KD"],
    "CCM2":     ["TeloHAEC_C2_CCM2_KD",     "TeloHAEC_F6_CCM2_KD"],
    "ITGB1BP1": ["TeloHAEC_C7_ITGB1BP1_KD", "TeloHAEC_G8_ITGB1BP1_KD"],
    "KLF2":     ["TeloHAEC_C9_KLF2_KD",     "TeloHAEC_G10_KLF2_KD"],
    "PDCD10":   ["TeloHAEC_D11_PDCD10_KD",  "TeloHAEC_G1_PDCD10_KD"],
}


def run_limma_trend(expr_df: pd.DataFrame, n_ctrl: int, n_kd: int) -> pd.DataFrame:
    """Run limma-trend via InMoose on log2-transformed expression.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Gene-by-sample expression matrix (log2(TPM+1)).
    n_ctrl : int
        Number of control samples (first columns).
    n_kd : int
        Number of knockdown samples (last columns).

    Returns
    -------
    pd.DataFrame
        DEG table with columns ``names``, ``logfoldchanges``, ``pvals``,
        ``pvals_adj``.
    """
    from inmoose.limma import eBayes, lmFit, topTable

    n_samples = n_ctrl + n_kd
    design = np.column_stack([
        np.ones(n_samples),
        np.array([0] * n_ctrl + [1] * n_kd, dtype=float),
    ])

    fit = lmFit(expr_df, design)
    fit = eBayes(fit, trend=True)

    coef_label = fit.coefficients.columns[1]
    logger.debug("Using coefficient: %r", coef_label)

    result = topTable(fit, coef=coef_label, number=np.inf, adjust_method="fdr_bh")
    return pd.DataFrame({
        "names": result.index,
        "logfoldchanges": result["log2FoldChange"].values,
        "pvals": result["pvalue"].values,
        "pvals_adj": result["adj_pvalue"].values,
    })


def run_ttest_fallback(expr_df: pd.DataFrame, n_ctrl: int) -> pd.DataFrame:
    """Welch t-test with BH correction as a fallback for limma-trend.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Gene-by-sample expression matrix (log2(TPM+1)).
    n_ctrl : int
        Number of control samples (first columns).

    Returns
    -------
    pd.DataFrame
        DEG table with columns ``names``, ``logfoldchanges``, ``pvals``,
        ``pvals_adj``.
    """
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests

    mat = expr_df.values
    ctrl = mat[:, :n_ctrl]
    kd = mat[:, n_ctrl:]

    lfc = kd.mean(axis=1) - ctrl.mean(axis=1)
    _, pvals = ttest_ind(kd, ctrl, axis=1, equal_var=False)
    pvals = np.where(np.isnan(pvals), 1.0, pvals)
    _, padj, _, _ = multipletests(pvals, method="fdr_bh")

    return pd.DataFrame({
        "names": expr_df.index,
        "logfoldchanges": lfc,
        "pvals": pvals,
        "pvals_adj": padj,
    })


def run_deg_analysis(
    tpm: pd.DataFrame,
    gene: str,
    kd_cols: list[str],
    *,
    method: str = "limma-trend",
) -> tuple[pd.DataFrame, str]:
    """Run DEG analysis for a single knockdown gene.

    Parameters
    ----------
    tpm : pd.DataFrame
        Full log2(TPM+1) expression matrix indexed by gene symbol.
    gene : str
        Knockdown gene name (used only for logging).
    kd_cols : list[str]
        Column names for the knockdown replicates.
    method : str
        ``"limma-trend"`` or ``"ttest"``.

    Returns
    -------
    tuple[pd.DataFrame, str]
        DEG results and the method name actually used.
    """
    all_cols = NEG_CTRL_COLS + kd_cols
    sub = tpm[all_cols].dropna()
    n_ctrl = len(NEG_CTRL_COLS)
    n_kd = len(kd_cols)
    logger.info(
        "%s: %d ctrl + %d KD samples, %d genes after dropping NaN",
        gene, n_ctrl, n_kd, len(sub),
    )

    if method == "limma-trend":
        try:
            return run_limma_trend(sub, n_ctrl, n_kd), "limma-trend"
        except ImportError:
            logger.warning("InMoose not found; falling back to Welch t-test + BH")
            return run_ttest_fallback(sub, n_ctrl), "ttest-fallback"
    return run_ttest_fallback(sub, n_ctrl), "ttest"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] or None
        Argument list; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Bulk RNA-seq DEG analysis (limma-trend or t-test).",
    )
    parser.add_argument(
        "--counts", required=True, type=Path,
        help="Tab-separated TPM matrix with gene symbols as row index.",
    )
    parser.add_argument("--metadata", type=Path, default=None, help="Metadata CSV (unused, reserved).")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for per-gene CSVs.")
    parser.add_argument(
        "--method", default="limma-trend", choices=["limma-trend", "ttest"],
        help="DEG method (default: limma-trend).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for bulk RNA-seq DEG CLI."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading TPM matrix from %s", args.counts)
    tpm = pd.read_csv(args.counts, sep="\t", index_col="Symbol")
    tpm = tpm.apply(pd.to_numeric, errors="coerce")
    logger.info("Loaded %d genes x %d samples", *tpm.shape)

    log_tpm = np.log2(tpm + 1)

    for gene, kd_cols in GENE_TO_KD_COLS.items():
        result, method_used = run_deg_analysis(log_tpm, gene, kd_cols, method=args.method)
        out_path = args.output_dir / f"{gene}_vs_control.csv"
        result.to_csv(out_path, index=False)

        n_sig = (result["pvals"] < 0.05).sum()
        n_fdr = (result["pvals_adj"] < 0.05).sum()
        logger.info(
            "%s [%s]: %d DEGs (p<0.05), %d (FDR<0.05) -> %s",
            gene, method_used, n_sig, n_fdr, out_path,
        )

    logger.info("All DEG results written to %s", args.output_dir)


if __name__ == "__main__":
    main()
