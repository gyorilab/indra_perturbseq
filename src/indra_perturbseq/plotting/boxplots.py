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

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

PATHWAY_ORDER = ["1-hop", "2-hop", "3-hop", "unexplained"]
PALETTE = {
    "1-hop": "blue",
    "2-hop": "green",
    "3-hop": "orange",
    "unexplained": "red",
}
DE_PVAL_CUTOFF = 0.05


def _load_hop_csv(path: str, hop_label: str) -> pd.DataFrame:
    """Load a hop CSV, compute averaged belief for multi-edge paths."""
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path, low_memory=False)

    belief_cols = [c for c in df.columns if c.startswith("belief")]
    if len(belief_cols) > 1:
        df["belief"] = df[belief_cols].mean(axis=1)
    elif belief_cols and "belief" not in df.columns:
        df["belief"] = df[belief_cols[0]]

    df["pathway_type"] = hop_label
    return df


def _priority_dedup(df1: pd.DataFrame, df2: pd.DataFrame,
                    df3: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate source-target pairs: 1-hop > 2-hop > 3-hop, tie-break by belief."""
    priority_map = {"1-hop": 1, "2-hop": 2, "3-hop": 3}
    frames = []
    for df, label in [(df1, "1-hop"), (df2, "2-hop"), (df3, "3-hop")]:
        sub = df[["source", "target", "belief", "logfoldchange", "pval", "pathway_type"]].copy()
        sub["priority"] = priority_map[label]
        frames.append(sub)

    combined = pd.concat(frames, ignore_index=True)
    best = (
        combined
        .sort_values(["priority", "belief"], ascending=[True, False])
        .drop_duplicates(subset=["source", "target"], keep="first")
        .reset_index(drop=True)
    )

    logger.info("After priority dedup: %d unique source-target pairs", len(best))
    logger.info("\n%s", best["pathway_type"].value_counts().to_string())
    return best


def _add_unexplained_targets(
    combined_df: pd.DataFrame,
    target_validation: str,
    deg_dir: str,
    filter_column: str,
    filter_value: str,
) -> pd.DataFrame:
    """Append unexplained targets: significant DEGs not explained by any hop."""
    tv = pd.read_csv(target_validation)
    allowed_sources = set(
        tv.loc[tv[filter_column] == filter_value, "Gene"]
        .astype(str).str.strip().tolist()
    )

    explained_pairs = set(zip(combined_df["source"], combined_df["target"]))
    pathway_sources = combined_df["source"].dropna().unique()

    unexplained_rows: list[dict] = []
    for source in pathway_sources:
        if source not in allowed_sources:
            continue
        deg_file = Path(deg_dir) / f"{source}_vs_control.csv"
        if not deg_file.exists():
            continue
        try:
            deg_df = pd.read_csv(deg_file)
        except Exception:
            continue
        if not {"names", "pvals", "logfoldchanges"}.issubset(deg_df.columns):
            continue

        sig = deg_df[deg_df["pvals"] < DE_PVAL_CUTOFF]
        for _, row in sig.iterrows():
            target = row["names"]
            if target == source:
                continue
            if (source, target) in explained_pairs:
                continue
            unexplained_rows.append({
                "source": source, "target": target, "belief": 0.5,
                "logfoldchange": row["logfoldchanges"], "pval": row["pvals"],
                "pathway_type": "unexplained", "priority": 4, "explained": False,
            })

    explained_df = combined_df.copy()
    explained_df["explained"] = True

    if unexplained_rows:
        final_df = pd.concat([explained_df, pd.DataFrame(unexplained_rows)], ignore_index=True)
    else:
        final_df = explained_df

    logger.info("Final dataset: %d rows", len(final_df))
    logger.info("\n%s", final_df["pathway_type"].value_counts().to_string())
    return final_df


def _plot_pvalue(final_df: pd.DataFrame, out_path: str,
                 show_outliers: bool = False) -> None:
    """Boxplot of -log10(p-value) by pathway type."""
    plot_df = final_df.copy()
    plot_df["neg_log10_pval"] = -np.log10(plot_df["pval"].clip(lower=1e-50))

    box_df = (
        plot_df.loc[plot_df["pathway_type"].isin(PATHWAY_ORDER),
                    ["pathway_type", "neg_log10_pval"]]
        .dropna()
        .copy()
    )
    counts = box_df.groupby("pathway_type").size().reindex(PATHWAY_ORDER, fill_value=0)

    plt.figure(figsize=(10, 7))
    if box_df.empty:
        plt.text(0.5, 0.5, "No p-value data available",
                 ha="center", va="center", transform=plt.gca().transAxes)
    else:
        sns.boxplot(data=box_df, x="pathway_type", y="neg_log10_pval",
                    order=PATHWAY_ORDER, palette=PALETTE,
                    showfliers=show_outliers)

    plt.xticks(ticks=range(len(PATHWAY_ORDER)),
               labels=[f"{p}\n(n={counts[p]:,})" for p in PATHWAY_ORDER],
               fontsize=10)
    plt.axhline(y=-np.log10(0.05), color="orange", linestyle="--", alpha=0.7, label="p=0.05")
    plt.axhline(y=-np.log10(0.01), color="red", linestyle="--", alpha=0.7, label="p=0.01")
    plt.xlabel("Pathway Type")
    plt.ylabel("$-\\log_{10}$(p-value)")
    plt.title("P-value Distributions by Pathway Type")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("P-value boxplot -> %s", out_path)


def _plot_abs_lfc(final_df: pd.DataFrame, out_path: str,
                  show_outliers: bool = False) -> None:
    """Boxplot of |log fold-change| by pathway type."""
    plot_df = final_df.copy()
    plot_df["abs_logfoldchange"] = pd.to_numeric(
        plot_df["logfoldchange"], errors="coerce"
    ).abs()

    box_df = (
        plot_df.loc[plot_df["pathway_type"].isin(PATHWAY_ORDER),
                    ["pathway_type", "abs_logfoldchange"]]
        .dropna()
        .copy()
    )
    counts = box_df.groupby("pathway_type").size().reindex(PATHWAY_ORDER, fill_value=0)

    plt.figure(figsize=(10, 7))
    if box_df.empty:
        plt.text(0.5, 0.5, "No log fold-change data available",
                 ha="center", va="center", transform=plt.gca().transAxes)
    else:
        sns.boxplot(data=box_df, x="pathway_type", y="abs_logfoldchange",
                    order=PATHWAY_ORDER, palette=PALETTE,
                    showfliers=show_outliers)

    plt.xticks(ticks=range(len(PATHWAY_ORDER)),
               labels=[f"{p}\n(n={counts[p]:,})" for p in PATHWAY_ORDER],
               fontsize=10)
    plt.axhline(y=0.0, color="black", linestyle="--", alpha=0.7, label="|logFC|=0")
    plt.xlabel("Pathway Type")
    plt.ylabel("|Log Fold-Change|")
    plt.title("|Log Fold-Change| Distributions by Pathway Type")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("LFC boxplot -> %s", out_path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Generate -log10(p-value) and |LFC| boxplots by pathway type.",
    )
    ap.add_argument("--hop1-csv", required=True, help="1-hop INDRA path CSV.")
    ap.add_argument("--hop2-csv", required=True, help="2-hop INDRA path CSV/Excel.")
    ap.add_argument("--hop3-csv", required=True, help="3-hop INDRA path CSV.")
    ap.add_argument("--target-validation", required=True,
                    help="Target validation CSV (e.g. target_validation_expanded.csv).")
    ap.add_argument("--deg-dir", required=True,
                    help="Directory containing per-gene DEG CSVs (<GENE>_vs_control.csv).")
    ap.add_argument("--filter-column", default="analysis_flag",
                    help="Column in target-validation CSV used for filtering.")
    ap.add_argument("--filter-value", default="Use_for_analysis",
                    help="Required value in --filter-column.")
    ap.add_argument("--no-outliers", action="store_true",
                    help="Hide outlier points from boxplots.")
    ap.add_argument("--output-pval-plot", default="pvalue_distributions_boxplot.png")
    ap.add_argument("--output-lfc-plot", default="logfoldchange_distributions_boxplot.png")
    ap.add_argument("--output-csv", default="final_dataset_for_boxplots.csv")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df1 = _load_hop_csv(args.hop1_csv, "1-hop")
    df2 = _load_hop_csv(args.hop2_csv, "2-hop")
    df3 = _load_hop_csv(args.hop3_csv, "3-hop")
    logger.info("Loaded -- 1-hop: %d, 2-hop: %d, 3-hop: %d", len(df1), len(df2), len(df3))

    combined = _priority_dedup(df1, df2, df3)
    final_df = _add_unexplained_targets(
        combined, args.target_validation, args.deg_dir,
        args.filter_column, args.filter_value,
    )

    final_df.to_csv(args.output_csv, index=False)
    logger.info("Saved combined data -> %s", args.output_csv)

    show = not args.no_outliers
    _plot_pvalue(final_df, args.output_pval_plot, show_outliers=show)
    _plot_abs_lfc(final_df, args.output_lfc_plot, show_outliers=show)

    logger.info("Done. Generated: %s, %s", args.output_pval_plot, args.output_lfc_plot)


if __name__ == "__main__":
    main()
