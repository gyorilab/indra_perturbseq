"""Extract top pathway results.
Two sub-commands:.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

STANDARD_COLUMNS = [
    "source", "target", "hop_number",
    "intermediate_1", "intermediate_2", "intermediate_3",
    "stmt_type_1", "edge1_evidence_text", "edge1_pmids",
    "stmt_type_2", "edge2_evidence_text", "edge2_pmids",
    "stmt_type_3", "edge3_evidence_text", "edge3_pmids",
    "stmt_type_4", "edge4_evidence_text", "edge4_pmids",
    "logfoldchange", "pval",
    "belief_1", "belief_2", "belief_3", "belief_4",
    "evidence_1", "evidence_2", "evidence_3", "evidence_4",
]

HOP_PRIORITY = {"1hop": 1, "2hop": 2, "3hop": 3, "4hop": 4}


# ------------------------------------------------------------------
# top-pairs
# ------------------------------------------------------------------

def extract_top_pairs(
    df: pd.DataFrame,
    n: int = 150,
    sort_col: str = "logfoldchange",
) -> pd.DataFrame:
    """Return the top *n* unique (source, target) pairs by *sort_col*.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with ``source``, ``target``, and *sort_col*.
    n : int
        Number of top pairs to return.
    sort_col : str
        Column used for ranking (highest first).

    Returns
    -------
    pd.DataFrame
        Top *n* rows after deduplication.
    """
    df = (
        df.sort_values(sort_col, ascending=False)
        .drop_duplicates(subset=["source", "target"], keep="first")
    )
    logger.info("Unique (source, target) pairs: %d", len(df))
    top = df.head(n)
    logger.info("Selected top %d pairs (range %.4f - %.4f)",
                len(top), top[sort_col].min(), top[sort_col].max())
    return top


# ------------------------------------------------------------------
# combine-hops
# ------------------------------------------------------------------

def _standardize_hop(df: pd.DataFrame, hop_label: str) -> pd.DataFrame:
    """Map a per-hop dataframe to the standard column layout."""
    for col in ("source", "target", "logfoldchange", "pval"):
        if col not in df.columns:
            raise ValueError(f"{hop_label} missing required column '{col}'")

    df = df.copy()
    df["hop_number"] = hop_label

    out = pd.DataFrame()
    for col in STANDARD_COLUMNS:
        out[col] = df[col] if col in df.columns else ""
    return out


def combine_hops(
    hop_files: dict[str, str | Path],
    top_n: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine per-hop CSVs into a single ranked output.

    Parameters
    ----------
    hop_files : dict[str, str | Path]
        Mapping of hop labels (e.g. ``"1hop"``) to file paths.
    top_n : int
        Number of top rows to keep per ranking criterion.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(combined, duplicates)`` where *duplicates* are the rows
        removed during deduplication (may be empty).
    """
    dfs: list[pd.DataFrame] = []
    for label, path in hop_files.items():
        p = Path(path)
        if not p.exists():
            logger.warning("Missing file for %s: %s", label, path)
            continue
        df = pd.read_csv(p, low_memory=False)
        dfs.append(_standardize_hop(df, label))

    if not dfs:
        raise FileNotFoundError("No valid hop files found")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d rows across %d hop files", len(combined), len(dfs))

    combined["logfoldchange"] = pd.to_numeric(
        combined["logfoldchange"], errors="coerce",
    )
    combined["pval"] = pd.to_numeric(combined["pval"], errors="coerce")
    combined["_abs_logfc"] = combined["logfoldchange"].abs()
    combined["_hop_rank"] = combined["hop_number"].map(HOP_PRIORITY)

    dupes = combined[
        combined.duplicated(subset=["source", "target"], keep=False)
    ].copy()

    top_fc = (
        combined
        .sort_values(["_abs_logfc", "_hop_rank"], ascending=[False, True])
        .drop_duplicates(subset=["source", "target"], keep="first")
        .head(top_n)
    )
    top_pval = (
        combined
        .sort_values(["pval", "_hop_rank"], ascending=[True, True])
        .drop_duplicates(subset=["source", "target"], keep="first")
        .head(top_n)
    )

    final = (
        pd.concat([top_fc, top_pval], ignore_index=True)
        .sort_values(["_hop_rank", "pval"], ascending=[True, True])
        .drop_duplicates(subset=["source", "target"], keep="first")
    )
    final = final[STANDARD_COLUMNS]
    logger.info("Combined output: %d rows", len(final))

    dupes_out = dupes[STANDARD_COLUMNS] if not dupes.empty else dupes
    return final, dupes_out


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _cli_top_pairs(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input, low_memory=False)
    logger.info("Loaded %d rows from %s", len(df), args.input)
    top = extract_top_pairs(df, n=args.top_n, sort_col=args.sort_col)
    top.to_csv(args.output, index=False)
    logger.info("Saved -> %s", args.output)


def _cli_combine_hops(args: argparse.Namespace) -> None:
    hop_files: dict[str, str] = {}
    for spec in args.hop:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"Invalid --hop format: {spec!r}  "
                             f"(expected LABEL=PATH, e.g. 1hop=/path.csv)")
        hop_files[label] = path

    final, dupes = combine_hops(hop_files, top_n=args.top_n)
    final.to_csv(args.output, index=False)
    logger.info("Saved -> %s", args.output)

    if args.duplicates_output and not dupes.empty:
        dupes.to_csv(args.duplicates_output, index=False)
        logger.info("Duplicates -> %s (%d rows)",
                     args.duplicates_output, len(dupes))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Extract top pathway results.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    tp = sub.add_parser("top-pairs",
                        help="Top N unique (source, target) by logfoldchange.")
    tp.add_argument("--input", required=True)
    tp.add_argument("--output", required=True)
    tp.add_argument("--top-n", type=int, default=150)
    tp.add_argument("--sort-col", default="logfoldchange")

    ch = sub.add_parser("combine-hops",
                        help="Combine per-hop CSVs into a standardized output.")
    ch.add_argument("--hop", action="append", required=True,
                    help="LABEL=PATH, e.g. 1hop=/data/1hop.csv  (repeat)")
    ch.add_argument("--output", required=True)
    ch.add_argument("--duplicates-output", default=None,
                    help="Optional CSV for removed duplicates.")
    ch.add_argument("--top-n", type=int, default=100)

    args = ap.parse_args(argv)
    if args.command == "top-pairs":
        _cli_top_pairs(args)
    else:
        _cli_combine_hops(args)


if __name__ == "__main__":
    main()
