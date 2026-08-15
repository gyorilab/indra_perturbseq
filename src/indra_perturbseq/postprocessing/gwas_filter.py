"""GWAS-aware filtering of 2-hop endothelial pathway data.

Pipeline:

1. Filter to endothelial intermediates (optionally excluding GWAS genes
   from the endothelial set).
2. Select top *N* rows by absolute log-fold-change with (source, target)
   uniqueness.
3. Collect all rows containing any GWAS gene, apply per-gene uniqueness,
   and annotate with directionality and GWAS gene membership.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _load_gene_set(path: str, column: str = "gene") -> set[str]:
    """Read a gene list CSV into a set."""
    df = pd.read_csv(path, low_memory=False)
    if column not in df.columns:
        raise ValueError(
            f"Expected column '{column}' in {path}. "
            f"Found: {df.columns.tolist()}"
        )
    return set(df[column].dropna().unique())


def filter_endothelial(
    df: pd.DataFrame,
    endothelial_genes: set[str],
    gwas_genes: set[str] | None = None,
    exclude_gwas: bool = False,
) -> pd.DataFrame:
    """Keep rows with an endothelial intermediate.

    Parameters
    ----------
    df : pd.DataFrame
        2-hop data.
    endothelial_genes : set[str]
        Endothelial gene symbols.
    gwas_genes : set[str] or None
        GWAS gene symbols to optionally remove from endothelial set.
    exclude_gwas : bool
        If ``True``, remove *gwas_genes* from the endothelial set before
        filtering.

    Returns
    -------
    pd.DataFrame
        Filtered copy.
    """
    allowed = set(endothelial_genes)
    if exclude_gwas and gwas_genes:
        removed = allowed & gwas_genes
        allowed -= gwas_genes
        logger.info("Removed %d GWAS genes from endothelial set", len(removed))

    mask = df["intermediate"].isin(allowed)
    before = len(df)
    out = df[mask].copy()
    logger.info("Endothelial filter: %d -> %d rows", before, len(out))
    return out


def top_by_logfc(
    df: pd.DataFrame,
    n: int = 100,
) -> pd.DataFrame:
    """Select top *n* rows by |logfoldchange| with (source, target) uniqueness.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered data.
    n : int
        Number of rows to return.

    Returns
    -------
    pd.DataFrame
        Top *n* rows.
    """
    df = df.copy()
    df["_abs_lfc"] = df["logfoldchange"].abs()
    unique = (
        df.sort_values("_abs_lfc", ascending=False)
        .drop_duplicates(subset=["source", "target"], keep="first")
    )
    top = unique.head(n).drop(columns=["_abs_lfc"])
    logger.info("Top %d by |logFC|: %d rows", n, len(top))
    return top


def annotate_directionality(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``directionality`` column.

    ``Yes`` when stmt_type_2 direction agrees with logfoldchange sign.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``stmt_type_2`` and ``logfoldchange``.

    Returns
    -------
    pd.DataFrame
        Copy with ``directionality`` column appended.
    """
    df = df.copy()
    df["directionality"] = "No"
    agree = (
        ((df["stmt_type_2"] == "IncreaseAmount") & (df["logfoldchange"] < 0))
        | ((df["stmt_type_2"] == "DecreaseAmount") & (df["logfoldchange"] > 0))
    )
    df.loc[agree, "directionality"] = "Yes"
    return df


def collect_gwas_rows(
    df: pd.DataFrame,
    gwas_genes: set[str],
) -> pd.DataFrame:
    """Extract rows containing any GWAS gene, with per-gene uniqueness.

    For each GWAS gene appearing as source or target the (source, target)
    pair is deduplicated by highest |logfoldchange|.  Rows where a GWAS
    gene appears only as intermediate are kept without deduplication.

    Parameters
    ----------
    df : pd.DataFrame
        Endothelial-filtered data with ``_abs_lfc`` or ``logfoldchange``.
    gwas_genes : set[str]
        GWAS gene symbols.

    Returns
    -------
    pd.DataFrame
        Annotated GWAS rows with ``GWAS_genes_in_path`` column.
    """
    df = df.copy()
    if "_abs_lfc" not in df.columns:
        df["_abs_lfc"] = df["logfoldchange"].abs()

    parts: list[pd.DataFrame] = []
    for gene in sorted(gwas_genes):
        gene_mask = (
            (df["source"] == gene)
            | (df["intermediate"] == gene)
            | (df["target"] == gene)
        )
        rows = df[gene_mask]
        if rows.empty:
            continue

        as_src = rows[rows["source"] == gene]
        as_tgt = rows[rows["target"] == gene]
        as_mid = rows[rows["intermediate"] == gene]

        deduped_src = (
            as_src.sort_values("_abs_lfc", ascending=False)
            .drop_duplicates(subset=["source", "target"], keep="first")
        ) if not as_src.empty else as_src

        deduped_tgt = (
            as_tgt.sort_values("_abs_lfc", ascending=False)
            .drop_duplicates(subset=["source", "target"], keep="first")
        ) if not as_tgt.empty else as_tgt

        combined = pd.concat(
            [deduped_src, as_mid, deduped_tgt],
        ).drop_duplicates()
        parts.append(combined)
        logger.info("GWAS gene %s: %d -> %d rows", gene,
                     len(rows), len(combined))

    if not parts:
        return pd.DataFrame(columns=df.columns)

    result = pd.concat(parts).drop_duplicates()
    result.drop(columns=["_abs_lfc"], inplace=True, errors="ignore")

    def _gwas_in_path(row: pd.Series) -> str:
        found = sorted({
            g for g in (row["source"], row["intermediate"], row["target"])
            if g in gwas_genes
        })
        return ", ".join(found)

    result["GWAS_genes_in_path"] = result.apply(_gwas_in_path, axis=1)
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Filter 2-hop data for endothelial intermediates and "
                    "GWAS gene annotation.",
    )
    ap.add_argument("--input", required=True, help="Input 2-hop CSV.")
    ap.add_argument("--gene-list", required=True,
                    help="CSV with endothelial genes ('gene' column).")
    ap.add_argument("--gwas-csv", default=None,
                    help="CSV with GWAS genes ('gene' column). When omitted, "
                         "GWAS filtering is skipped.")
    ap.add_argument("--output", required=True,
                    help="Output: endothelial-filtered CSV.")
    ap.add_argument("--output-top", default=None,
                    help="Optional output: top N rows by |logFC|.")
    ap.add_argument("--output-gwas", default=None,
                    help="Optional output: GWAS-annotated rows.")
    ap.add_argument("--top-n", type=int, default=100,
                    help="Number of top rows to select.")
    ap.add_argument("--exclude-gwas-from-endo", action="store_true",
                    help="Remove GWAS genes from the endothelial set.")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.input, low_memory=False)
    logger.info("Loaded %d rows from %s", len(df), args.input)

    endo_genes = _load_gene_set(args.gene_list)
    gwas_genes: set[str] = set()
    if args.gwas_csv:
        gwas_genes = _load_gene_set(args.gwas_csv)
        logger.info("GWAS genes: %d", len(gwas_genes))

    df_endo = filter_endothelial(
        df, endo_genes, gwas_genes,
        exclude_gwas=args.exclude_gwas_from_endo,
    )
    df_endo.to_csv(args.output, index=False)
    logger.info("Endothelial-filtered -> %s (%d rows)",
                args.output, len(df_endo))

    if args.output_top:
        top = top_by_logfc(df_endo, n=args.top_n)
        top = annotate_directionality(top)
        top.to_csv(args.output_top, index=False)
        logger.info("Top %d -> %s", args.top_n, args.output_top)

    if args.output_gwas and gwas_genes:
        gwas_df = collect_gwas_rows(df_endo, gwas_genes)
        gwas_df = annotate_directionality(gwas_df)
        gwas_df = gwas_df.sort_values(
            "logfoldchange", key=abs, ascending=False,
        )
        gwas_df.to_csv(args.output_gwas, index=False)
        logger.info("GWAS paths -> %s (%d rows)",
                     args.output_gwas, len(gwas_df))


if __name__ == "__main__":
    main()
