"""Generate boxplots for hop-based pathway p-value and LFC distributions.
Includes deduplication and unexplained target category comparisons.

Hub genes (e.g. TP53, CDKN1A) can be included via --hub-1hop-csv /
--hub-2hop-csv flags. When supplied, hub paths are merged into the main hop
data before deduplication, and hub DEG files (looked up from --deg-dir) are
added to the unexplained pool. Both a with-outliers and a no-outliers version
of the p-value boxplot are always produced.
"""

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

    if "belief" not in df.columns:
        df["belief"] = float("nan")

    df["pathway_type"] = hop_label
    return df


def _merge_hub(main_df: pd.DataFrame, hub_path: str | None, hop_label: str) -> pd.DataFrame:
    """Concatenate an optional hub CSV into a main hop dataframe."""
    if not hub_path:
        return main_df
    hub = _load_hop_csv(hub_path, hop_label)
    logger.info("Hub %s rows loaded from %s: %d", hop_label, hub_path, len(hub))
    return pd.concat([main_df, hub], ignore_index=True)


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
    extra_sources: list[str] | None = None,
) -> pd.DataFrame:
    """Append unexplained targets: significant DEGs not explained by any hop.

    ``extra_sources`` (e.g. hub gene names) are added to the allowed set even
    if they are absent from the target-validation CSV.  Self-loop targets
    (source == target) are always excluded.
    """
    tv = pd.read_csv(target_validation)
    allowed_sources = set(
        tv.loc[tv[filter_column] == filter_value, "Gene"]
        .astype(str).str.strip().tolist()
    )
    if extra_sources:
        allowed_sources.update(s.strip() for s in extra_sources if s.strip())

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
                 show_outliers: bool = False, title_suffix: str = "") -> None:
    """Boxplot of -log10(p-value) by pathway type.

    Only pairs with pval < 0.05 are included so that all hop categories are
    comparable to the unexplained category (which is already filtered to
    pval < 0.05 during construction).
    """
    plot_df = final_df.copy()
    plot_df["pval"] = pd.to_numeric(plot_df["pval"], errors="coerce")
    plot_df = plot_df[plot_df["pval"] < DE_PVAL_CUTOFF]
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
                    order=PATHWAY_ORDER, hue="pathway_type", palette=PALETTE,
                    showfliers=show_outliers, legend=False)

    plt.xticks(ticks=range(len(PATHWAY_ORDER)),
               labels=[f"{p}\n(n={counts[p]:,})" for p in PATHWAY_ORDER],
               fontsize=10)
    plt.axhline(y=-np.log10(0.05), color="orange", linestyle="--", alpha=0.7, label="p=0.05")
    plt.axhline(y=-np.log10(0.01), color="red", linestyle="--", alpha=0.7, label="p=0.01")
    plt.xlabel("Pathway Type")
    plt.ylabel("$-\\log_{10}$(p-value)")
    outlier_tag = "with outliers" if show_outliers else "no outliers"
    full_title = f"P-value Distributions by Pathway Type ({outlier_tag})"
    if title_suffix:
        full_title += f" — {title_suffix}"
    plt.title(full_title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("P-value boxplot -> %s", out_path)


def _plot_abs_lfc(final_df: pd.DataFrame, out_path: str,
                  show_outliers: bool = False, title_suffix: str = "") -> None:
    """Boxplot of |log fold-change| by pathway type.

    Only pairs with pval < 0.05 are included, matching _plot_pvalue so all
    hop categories are comparable.
    """
    plot_df = final_df.copy()
    plot_df["pval"] = pd.to_numeric(plot_df["pval"], errors="coerce")
    plot_df = plot_df[plot_df["pval"] < DE_PVAL_CUTOFF]
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
                    order=PATHWAY_ORDER, hue="pathway_type", palette=PALETTE,
                    showfliers=show_outliers, legend=False)

    plt.xticks(ticks=range(len(PATHWAY_ORDER)),
               labels=[f"{p}\n(n={counts[p]:,})" for p in PATHWAY_ORDER],
               fontsize=10)
    plt.axhline(y=0.0, color="black", linestyle="--", alpha=0.7, label="|logFC|=0")
    plt.xlabel("Pathway Type")
    plt.ylabel("|Log Fold-Change|")
    title = "|Log Fold-Change| Distributions by Pathway Type"
    if title_suffix:
        title += f" — {title_suffix}"
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("LFC boxplot -> %s", out_path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate -log10(p-value) and |LFC| boxplots by pathway type. "
            "Optionally include hub gene paths via --hub-1hop-csv / --hub-2hop-csv."
        ),
    )
    ap.add_argument("--hop1-csv", required=True, help="1-hop INDRA path CSV.")
    ap.add_argument("--hop2-csv", required=True, help="2-hop INDRA path CSV/Excel.")
    ap.add_argument("--hop3-csv", required=True, help="3-hop INDRA path CSV.")
    ap.add_argument("--hub-1hop-csv", default=None,
                    help="Optional hub gene 1-hop CSV (e.g. outliers_1hop_endothelial_universe.csv).")
    ap.add_argument("--hub-2hop-csv", default=None,
                    help="Optional hub gene 2-hop CSV (e.g. outliers_2hop_endothelial_universe.csv).")
    ap.add_argument("--hub-genes", nargs="*", default=[],
                    help="Hub gene names to include in unexplained pool (e.g. TP53 CDKN1A).")
    ap.add_argument("--target-validation", required=True,
                    help="Target validation CSV (e.g. target_validation_expanded.csv).")
    ap.add_argument("--deg-dir", required=True,
                    help="Directory containing per-gene DEG CSVs (<GENE>_vs_control.csv).")
    ap.add_argument("--filter-column", default="analysis_flag",
                    help="Column in target-validation CSV used for filtering.")
    ap.add_argument("--filter-value", default="Use_for_analysis",
                    help="Required value in --filter-column.")
    ap.add_argument("--output-pval-with-outliers",
                    default="pvalue_distributions_boxplot_with_outliers.png")
    ap.add_argument("--output-pval-no-outliers",
                    default="pvalue_distributions_boxplot_no_outliers.png")
    ap.add_argument("--output-lfc-plot", default="logfoldchange_distributions_boxplot.png")
    ap.add_argument("--output-csv", default="final_dataset_for_boxplots.csv")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    with_hubs = bool(args.hub_1hop_csv or args.hub_2hop_csv or args.hub_genes)
    title_suffix = "with hubs" if with_hubs else ""
    output_lfc = (
        "logfoldchange_with_hubs.png"
        if with_hubs and args.output_lfc_plot == "logfoldchange_distributions_boxplot.png"
        else args.output_lfc_plot
    )

    df1 = _load_hop_csv(args.hop1_csv, "1-hop")
    df2 = _load_hop_csv(args.hop2_csv, "2-hop")
    df3 = _load_hop_csv(args.hop3_csv, "3-hop")
    logger.info("Loaded -- 1-hop: %d, 2-hop: %d, 3-hop: %d", len(df1), len(df2), len(df3))

    df1 = _merge_hub(df1, args.hub_1hop_csv, "1-hop")
    df2 = _merge_hub(df2, args.hub_2hop_csv, "2-hop")

    combined = _priority_dedup(df1, df2, df3)
    final_df = _add_unexplained_targets(
        combined, args.target_validation, args.deg_dir,
        args.filter_column, args.filter_value,
        extra_sources=args.hub_genes or [],
    )

    final_df.to_csv(args.output_csv, index=False)
    logger.info("Saved combined data -> %s", args.output_csv)

    _plot_pvalue(final_df, args.output_pval_with_outliers,
                 show_outliers=True, title_suffix=title_suffix)
    _plot_pvalue(final_df, args.output_pval_no_outliers,
                 show_outliers=False, title_suffix=title_suffix)
    _plot_abs_lfc(final_df, output_lfc, show_outliers=False, title_suffix=title_suffix)

    logger.info(
        "Done. Generated: %s, %s, %s",
        args.output_pval_with_outliers, args.output_pval_no_outliers, output_lfc,
    )


if __name__ == "__main__":
    main()
