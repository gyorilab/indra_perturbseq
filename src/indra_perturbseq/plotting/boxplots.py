"""Boxplot visualizations for pathway p-value and LFC distributions."""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from indra_perturbseq.deg import pick_sig_column
from indra_perturbseq.hgnc import normalize_hgnc_symbol

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def _load_hop_csv(path: str, hop_label: str, belief_cutoff: float,
                  exclude_sources: set[str]) -> pd.DataFrame:
    """Load a hop CSV and apply belief filtering."""
    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    belief_cols = [c for c in df.columns if c.startswith("belief")]
    if len(belief_cols) > 1:
        df["belief"] = df[belief_cols].mean(axis=1)
    elif "belief" not in df.columns and belief_cols:
        df["belief"] = df[belief_cols[0]]

    if "belief" in df.columns:
        df = df[df["belief"] >= belief_cutoff].copy()

    if "source" in df.columns and exclude_sources:
        src_norm = df["source"].astype(str).map(normalize_hgnc_symbol)
        df = df[~src_norm.isin(exclude_sources)].copy()

    df["pathway_type"] = hop_label
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Generate p-value and LFC boxplots.",
    )
    ap.add_argument("--hop1-csv", required=True)
    ap.add_argument("--hop2-csv", required=True)
    ap.add_argument("--hop3-csv", required=True)
    ap.add_argument("--tv-csv", required=True,
                    help="target_validation_expanded.csv")
    ap.add_argument("--de-dir", required=True)
    ap.add_argument("--belief-cutoff", type=float, default=0.7)
    ap.add_argument("--pval-cutoff", type=float, default=0.05)
    ap.add_argument("--exclude-sources", nargs="*", default=["TP53", "CDKN1A"])
    ap.add_argument("--out-pval-plot", default="pvalue_distributions_boxplot.png")
    ap.add_argument("--out-lfc-plot", default="logfoldchange_distributions_boxplot.png")
    ap.add_argument("--out-csv", default="final_dataset_for_boxplots.csv")
    args = ap.parse_args(argv)

    exclude = set(args.exclude_sources or [])
    df1 = _load_hop_csv(args.hop1_csv, "1-hop", args.belief_cutoff, exclude)
    df2 = _load_hop_csv(args.hop2_csv, "2-hop", args.belief_cutoff, exclude)
    df3 = _load_hop_csv(args.hop3_csv, "3-hop", args.belief_cutoff, exclude)

    logger.info("1-hop: %d, 2-hop: %d, 3-hop: %d", len(df1), len(df2), len(df3))

    combined = pd.concat([df1, df2, df3], ignore_index=True)
    if "pval" in combined.columns:
        combined["pval"] = pd.to_numeric(combined["pval"], errors="coerce")
    if "logfoldchange" in combined.columns:
        combined["logfoldchange"] = pd.to_numeric(combined["logfoldchange"], errors="coerce")

    combined.to_csv(args.out_csv, index=False)
    logger.info("Combined data: %d rows -> %s", len(combined), args.out_csv)

    fig, ax = plt.subplots(figsize=(10, 6))
    if "pval" in combined.columns:
        sns.boxplot(data=combined, x="pathway_type", y="pval",
                    order=["1-hop", "2-hop", "3-hop"], ax=ax,
                    showfliers=False)
        ax.set_ylabel("p-value")
        ax.set_title("P-value distributions by hop distance")
    fig.tight_layout()
    fig.savefig(args.out_pval_plot, dpi=300)
    plt.close(fig)
    logger.info("P-value boxplot -> %s", args.out_pval_plot)

    fig, ax = plt.subplots(figsize=(10, 6))
    if "logfoldchange" in combined.columns:
        sns.boxplot(data=combined, x="pathway_type", y="logfoldchange",
                    order=["1-hop", "2-hop", "3-hop"], ax=ax,
                    showfliers=False)
        ax.set_ylabel("Log fold-change")
        ax.set_title("Log fold-change distributions by hop distance")
    fig.tight_layout()
    fig.savefig(args.out_lfc_plot, dpi=300)
    plt.close(fig)
    logger.info("LFC boxplot -> %s", args.out_lfc_plot)


if __name__ == "__main__":
    main()
