"""Run per-gene differential expression on Perturb-seq AnnData.

For each perturbation group, performs a Wilcoxon rank-sum test against the
control group (negative-control + safe-targeting cells) using Scanpy and
writes one ``<GENE>_vs_control.csv`` file per group.

Output columns match the format expected by the downstream hop-analysis
pipelines: ``names``, ``logfoldchanges``, ``pvals``, ``pvals_adj``.
"""

from __future__ import annotations

import argparse
import logging
import os

import scanpy as sc

logger = logging.getLogger(__name__)

CONTROL_LABELS = {"negative-control", "safe-targeting"}


def run_deg(
    adata_path: str,
    out_dir: str,
    *,
    gene_column: str = "Gene",
    tss_suffix: str = "-TSS2",
    method: str = "wilcoxon",
) -> int:
    """Run DE analysis and write per-gene CSVs.

    Returns the number of groups exported.
    """
    logger.info("Loading AnnData: %s", adata_path)
    adata = sc.read_h5ad(adata_path)

    clean = adata.obs[gene_column].str.replace(tss_suffix, "", regex=False)
    adata.obs["gene_group"] = clean.where(
        ~clean.isin(CONTROL_LABELS), "control",
    )

    logger.info("Running DE (%s vs control)...", method)
    sc.tl.rank_genes_groups(
        adata,
        groupby="gene_group",
        reference="control",
        method=method,
        pts=True,
    )

    os.makedirs(out_dir, exist_ok=True)
    groups = adata.uns["rank_genes_groups"]["names"].dtype.names

    for g in groups:
        df = sc.get.rank_genes_groups_df(adata, group=g)
        df.to_csv(os.path.join(out_dir, f"{g}_vs_control.csv"), index=False)

    logger.info("Wrote %d DEG CSVs to %s", len(groups), out_dir)
    return len(groups)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Per-gene Perturb-seq differential expression via Scanpy.",
    )
    ap.add_argument("--adata", required=True,
                    help="Path to .h5ad AnnData file")
    ap.add_argument("--output-dir", required=True,
                    help="Directory for <GENE>_vs_control.csv outputs")
    ap.add_argument("--gene-column", default="Gene",
                    help="obs column with perturbation labels")
    ap.add_argument("--tss-suffix", default="-TSS2",
                    help="Suffix stripped from gene labels before grouping")
    ap.add_argument("--method", default="wilcoxon",
                    choices=["wilcoxon", "t-test", "t-test_overestim_var",
                             "logreg"],
                    help="Scanpy DE method")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_deg(
        args.adata,
        args.output_dir,
        gene_column=args.gene_column,
        tss_suffix=args.tss_suffix,
        method=args.method,
    )


if __name__ == "__main__":
    main()
