"""Evaluate INDRA-predicted targets against bulk RNA DEG evidence.
Computes per-source confusion metrics and aggregate performance summaries."""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def _safe_rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator) if denominator else float("nan")


def load_gene_universe(path: str, gene_column: str = "gene") -> set[str]:
    """Load a set of gene symbols from a single-column CSV.

    Parameters
    ----------
    path : str
        Path to a CSV with at least one column of gene symbols.
    gene_column : str
        Name of the column containing gene names.

    Returns
    -------
    set[str]
        Unique, whitespace-stripped gene symbols.
    """
    df = pd.read_csv(path, low_memory=False)
    if gene_column not in df.columns:
        raise ValueError(
            f"Gene universe file must have '{gene_column}' column. "
            f"Got: {df.columns.tolist()}"
        )
    return set(df[gene_column].astype(str).str.strip().dropna().loc[lambda s: s != ""])


def load_predicted_targets(
    indra_csv: str,
    gene_universe: set[str],
    allowed_hops: set[int],
) -> tuple[set[str], pd.DataFrame]:
    """Load INDRA path targets, restricted to allowed hops and gene universe.

    Parameters
    ----------
    indra_csv : str
        CSV with ``target`` and ``hop`` columns.
    gene_universe : set[str]
        Restrict targets to this set.
    allowed_hops : set[int]
        Only keep rows whose hop value is in this set.

    Returns
    -------
    tuple[set[str], pd.DataFrame]
        Predicted target set and a DataFrame with the minimum hop per target.
    """
    df = pd.read_csv(indra_csv, low_memory=False)
    for col in ("target", "hop"):
        if col not in df.columns:
            raise ValueError(
                f"INDRA file must include '{col}'. Columns: {df.columns.tolist()}"
            )
    df["target"] = df["target"].astype(str).str.strip()
    df["hop"] = pd.to_numeric(df["hop"], errors="coerce")
    df = df[df["hop"].isin(allowed_hops) & df["target"].isin(gene_universe)]

    minhop = (
        df.dropna(subset=["hop"])
        .groupby("target", as_index=False)["hop"]
        .min()
        .rename(columns={"hop": "min_hop"})
    )
    return set(minhop["target"]), minhop


def load_deg(deg_csv: str) -> pd.DataFrame:
    """Load and validate a bulk RNA-seq DEG CSV.

    Parameters
    ----------
    deg_csv : str
        CSV with ``names``, ``logfoldchanges``, ``pvals``, ``pvals_adj``.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(deg_csv, low_memory=False)
    required = {"names", "logfoldchanges", "pvals", "pvals_adj"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DEG file missing columns {sorted(missing)}. "
            f"Columns: {df.columns.tolist()}"
        )
    df["names"] = df["names"].astype(str).str.strip()
    for c in ("logfoldchanges", "pvals", "pvals_adj"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def evaluate_gene(
    source: str,
    indra_csv: str,
    deg_csv: str,
    gene_universe: set[str],
    allowed_hops: set[int],
    fdr_threshold: float,
) -> tuple[dict, pd.DataFrame]:
    """Compute TP/FP/FN for a single source gene.

    Parameters
    ----------
    source : str
        Source gene symbol.
    indra_csv : str
        INDRA predictions CSV for this source.
    deg_csv : str
        Bulk RNA DEG CSV for this source.
    gene_universe : set[str]
        Endothelial or other gene universe.
    allowed_hops : set[int]
        Hop numbers to include.
    fdr_threshold : float
        FDR significance threshold.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        Summary metrics dict and per-target detail DataFrame.
    """
    predicted, minhop_df = load_predicted_targets(
        indra_csv, gene_universe, allowed_hops,
    )
    deg = load_deg(deg_csv)

    universe = set(deg["names"]) & gene_universe
    empirical_sig = set(
        deg.loc[deg["pvals_adj"] < fdr_threshold, "names"]
    ) & gene_universe

    tp = predicted & empirical_sig
    fp = predicted - empirical_sig
    fn = empirical_sig - predicted
    tn = universe - (predicted | empirical_sig)

    deg_join = deg.rename(columns={"names": "target"})
    detail = minhop_df.copy()
    detail["source"] = source
    detail = detail.merge(
        deg_join[["target", "logfoldchanges", "pvals", "pvals_adj"]],
        on="target",
        how="left",
    )
    detail["is_empirical_sig"] = detail["pvals_adj"] < fdr_threshold
    detail["classification"] = detail["is_empirical_sig"].map(
        {True: "TP", False: "FP"},
    )

    fn_df = deg_join.loc[
        (deg_join["pvals_adj"] < fdr_threshold)
        & (~deg_join["target"].isin(predicted))
        & (deg_join["target"].isin(gene_universe)),
        ["target", "logfoldchanges", "pvals", "pvals_adj"],
    ].copy()
    fn_df["source"] = source
    fn_df["min_hop"] = pd.NA
    fn_df["is_empirical_sig"] = True
    fn_df["classification"] = "FN"

    detail = pd.concat([detail, fn_df], ignore_index=True)
    col_order = [
        "source", "target", "min_hop",
        "logfoldchanges", "pvals", "pvals_adj",
        "is_empirical_sig", "classification",
    ]
    extra = [c for c in detail.columns if c not in col_order]
    detail = detail[col_order + extra]

    hops_label = "+".join(str(h) for h in sorted(allowed_hops))
    summary = {
        "gene": source,
        "hops_used": hops_label,
        "fdr_threshold": fdr_threshold,
        "universe_size": len(universe),
        "n_predicted_targets": len(predicted),
        "n_empirical_sig_fdr": len(empirical_sig),
        "TP": len(tp),
        "FP": len(fp),
        "FN": len(fn),
        "TN": len(tn),
        "TPR": _safe_rate(len(tp), len(tp) + len(fn)),
        "FNR": _safe_rate(len(fn), len(tp) + len(fn)),
        "FPR": _safe_rate(len(fp), len(fp) + len(tn)),
        "precision": _safe_rate(len(tp), len(tp) + len(fp)),
    }
    return summary, detail


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="TP/FP/FN evaluation of INDRA predictions against bulk RNA-seq.",
    )
    ap.add_argument("--indra-dir", required=True,
                    help="Directory with <GENE>_all_hops.csv files")
    ap.add_argument("--deg-dir", required=True,
                    help="Directory with <GENE>_vs_control.csv bulk DEG files")
    ap.add_argument("--gene-universe-csv", required=True,
                    help="CSV with gene-universe column (e.g. endothelial list)")
    ap.add_argument("--gene-universe-col", default="gene")
    ap.add_argument("--genes", nargs="+", required=True,
                    help="Source gene symbols to evaluate")
    ap.add_argument("--indra-suffix", default="_all_hops.csv")
    ap.add_argument("--deg-suffix", default="_vs_control.csv")
    ap.add_argument("--fdr-threshold", type=float, default=0.05)
    ap.add_argument("--allowed-hops", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    gene_universe = load_gene_universe(
        args.gene_universe_csv, gene_column=args.gene_universe_col,
    )
    allowed_hops = set(args.allowed_hops)
    summary_rows: list[dict] = []

    for gene in args.genes:
        indra_path = os.path.join(args.indra_dir, f"{gene}{args.indra_suffix}")
        deg_path = os.path.join(args.deg_dir, f"{gene}{args.deg_suffix}")

        if not os.path.exists(indra_path):
            raise FileNotFoundError(f"Missing INDRA file: {indra_path}")
        if not os.path.exists(deg_path):
            raise FileNotFoundError(f"Missing DEG file: {deg_path}")

        summary, detail = evaluate_gene(
            gene, indra_path, deg_path,
            gene_universe, allowed_hops, args.fdr_threshold,
        )
        hops_label = "+".join(str(h) for h in sorted(allowed_hops))
        out_path = os.path.join(
            args.output_dir,
            f"{gene}_bulk_eval_hops_{hops_label}.csv",
        )
        detail.to_csv(out_path, index=False)
        summary["per_gene_output_csv"] = out_path
        summary_rows.append(summary)
        logger.info("[%s] wrote %s", gene, out_path)

    summary_df = pd.DataFrame(summary_rows).sort_values("gene")
    summary_out = os.path.join(args.output_dir, "bulk_rna_eval_summary.csv")
    summary_df.to_csv(summary_out, index=False)
    logger.info("Wrote summary: %s", summary_out)
    logger.info("\n%s", summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
