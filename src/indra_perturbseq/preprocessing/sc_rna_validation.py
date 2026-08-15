"""scRNA-seq DEG analysis for validating selected perturbations vs control.

Runs Scanpy Wilcoxon rank-sum tests per perturbation gene against pooled
control cells, optionally restricted to an endothelial gene universe.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

logger = logging.getLogger(__name__)


def load_endothelial_set(path: Path, column: str = "gene") -> set[str]:
    """Load endothelial gene universe from a CSV file.

    Parameters
    ----------
    path : Path
        CSV file containing gene names.
    column : str
        Column name to read gene symbols from.

    Returns
    -------
    set[str]
        Set of gene symbols.

    Raises
    ------
    ValueError
        If *column* is not present in the CSV.
    """
    df = pd.read_csv(path, low_memory=False)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {df.columns.tolist()}")
    genes = set(df[column].astype(str).str.strip().replace("", pd.NA).dropna().tolist())
    logger.info("Loaded %d endothelial genes from %s", len(genes), path)
    return genes


def compute_gene_variance(X: np.ndarray | sparse.spmatrix) -> np.ndarray:
    """Compute per-gene variance across cells.

    Parameters
    ----------
    X : array-like or sparse matrix
        Expression matrix (cells x genes).

    Returns
    -------
    np.ndarray
        Variance per gene.
    """
    if sparse.issparse(X):
        mean_sq = np.array(X.power(2).mean(axis=0)).ravel()
        mean = np.array(X.mean(axis=0)).ravel()
        return mean_sq - mean ** 2
    return np.array(X).var(axis=0)


def write_empty_deg(path: Path) -> None:
    """Write an empty DEG CSV with expected columns.

    Parameters
    ----------
    path : Path
        Output file path.
    """
    cols = ["names", "scores", "logfoldchanges", "pvals", "pvals_adj", "pct_nz_group"]
    pd.DataFrame({c: [] for c in cols}).to_csv(path, index=False)


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
        description="scRNA-seq DEG validation for selected perturbations vs control (Wilcoxon).",
    )
    parser.add_argument("--adata", required=True, type=Path, help="Path to .h5ad AnnData file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for per-gene CSVs.")
    parser.add_argument(
        "--genes", nargs="+", required=True,
        help="Perturbation genes to test.",
    )
    parser.add_argument("--perturb-col", default="Gene", help="adata.obs column with perturbation labels.")
    parser.add_argument(
        "--controls", nargs="+", default=["negative-control", "safe-targeting"],
        help="Control labels in the perturbation column.",
    )
    parser.add_argument("--strip-suffix", default="-TSS2", help="Suffix to strip from perturbation labels.")
    parser.add_argument("--min-cells", type=int, default=1, help="Minimum cells required per perturbation.")
    parser.add_argument("--endothelial-list", required=True, type=Path, help="CSV of endothelial gene universe.")
    parser.add_argument("--endothelial-column", default="gene", help="Column name in endothelial-list CSV.")
    parser.add_argument(
        "--top-variable-fraction", type=float, default=0.0,
        help="Keep top fraction of most variable genes within endothelial set (0 = keep all).",
    )
    parser.add_argument("--normalize-log1p", action="store_true", help="Run normalize_total + log1p before DE.")
    parser.add_argument("--target-sum", type=float, default=1e4, help="Target sum for normalize_total.")
    parser.add_argument(
        "--symbol-col", default=None,
        help="adata.var column with gene symbols if var_names are Ensembl IDs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for scRNA-seq validation DEG CLI."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    endo_set = load_endothelial_set(args.endothelial_list, args.endothelial_column)

    logger.info("Reading AnnData from %s", args.adata)
    adata = sc.read_h5ad(args.adata)

    if args.symbol_col is not None:
        if args.symbol_col not in adata.var.columns:
            raise ValueError(
                f"--symbol-col '{args.symbol_col}' not in adata.var. "
                f"Available: {adata.var.columns.tolist()}"
            )
        adata = adata.copy()
        adata.var_names = adata.var[args.symbol_col].astype(str).str.strip().values

    if args.perturb_col not in adata.obs.columns:
        raise ValueError(
            f"adata.obs missing '{args.perturb_col}'. "
            f"Available: {adata.obs.columns.tolist()}"
        )

    labels = adata.obs[args.perturb_col].astype(str).str.strip()
    if args.strip_suffix:
        labels = labels.str.replace(args.strip_suffix, "", regex=False)
    adata.obs["Gene_Clean"] = labels

    controls = {str(x).strip() for x in args.controls}
    requested = [str(x).strip() for x in args.genes if str(x).strip()]
    keep_groups = set(requested) | controls

    adata = adata[adata.obs["Gene_Clean"].isin(keep_groups)].copy()
    adata.obs["Gene_Group"] = adata.obs["Gene_Clean"].where(
        ~adata.obs["Gene_Clean"].isin(controls), "control",
    )

    counts = adata.obs["Gene_Clean"].value_counts().to_dict()
    runnable = [g for g in requested if counts.get(g, 0) >= args.min_cells]

    if not runnable:
        for g in requested:
            write_empty_deg(args.output_dir / f"{g}_vs_control.csv")
        logger.warning("No perturbations met --min-cells. Wrote empty DEG CSVs.")
        return

    if args.normalize_log1p:
        sc.pp.normalize_total(adata, target_sum=args.target_sum)
        sc.pp.log1p(adata)

    endo_mask = adata.var_names.astype(str).isin(endo_set)
    if int(endo_mask.sum()) == 0:
        raise RuntimeError(
            "No overlap between adata.var_names and endothelial gene list. "
            "If var_names are Ensembl IDs, use --symbol-col."
        )
    adata = adata[:, endo_mask].copy()

    if 0.0 < args.top_variable_fraction <= 1.0:
        var = compute_gene_variance(adata.X)
        cutoff = np.quantile(var, 1.0 - args.top_variable_fraction)
        adata = adata[:, var >= cutoff].copy()
    elif args.top_variable_fraction > 1.0:
        raise ValueError("--top-variable-fraction must be in (0, 1].")

    logger.info("Running Wilcoxon DE for %d perturbations vs control", len(runnable))
    sc.tl.rank_genes_groups(
        adata, groupby="Gene_Group", reference="control", method="wilcoxon", pts=True,
    )

    groups_done = set(adata.uns["rank_genes_groups"]["names"].dtype.names)
    for g in requested:
        out_path = args.output_dir / f"{g}_vs_control.csv"
        if g not in groups_done:
            write_empty_deg(out_path)
            continue
        df = sc.get.rank_genes_groups_df(adata, group=g)
        df.to_csv(out_path, index=False)
        logger.info("%s: %d DEGs written to %s", g, len(df), out_path)

    logger.info("All DEG results written to %s", args.output_dir)


if __name__ == "__main__":
    main()
