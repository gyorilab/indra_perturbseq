"""Convert UniProt identifiers to HGNC symbols in multi-hop result CSVs.

Scans columns matching ``intermediate_*`` and replaces UniProt-prefixed
identifiers with their corresponding HGNC gene symbols using INDRA clients.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from indra.databases import hgnc_client, uniprot_client

logger = logging.getLogger(__name__)


def convert_to_hgnc_symbol(identifier: str) -> str:
    """Convert a UniProt (or UniProt.chain) ID to an HGNC symbol.

    Parameters
    ----------
    identifier : str
        Raw identifier string, e.g. ``"uniprot:P12345"``.

    Returns
    -------
    str
        HGNC symbol if resolution succeeds, otherwise the original string.
    """
    if not isinstance(identifier, str):
        return identifier
    normalized = (
        identifier.strip()
        .replace("uniprot.chain:", "uniprot:")
        .replace("hgnc:uniprot:", "uniprot:")
    )
    if not normalized.startswith("uniprot:"):
        return identifier

    uid = normalized.split("uniprot:")[-1]
    hgnc_id = uniprot_client.get_hgnc_id(uid)
    if hgnc_id:
        symbol = hgnc_client.get_hgnc_name(hgnc_id)
        if symbol:
            return symbol
    return identifier


def convert_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int, set[str]]:
    """Convert all UniProt IDs in ``intermediate_*`` columns of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with intermediate columns.

    Returns
    -------
    tuple[pd.DataFrame, int, int, set[str]]
        Converted DataFrame, count of UniProt IDs seen, count converted,
        and set of IDs that could not be resolved.
    """
    intermediate_cols = [c for c in df.columns if c.startswith("intermediate_")]
    logger.info("Converting columns: %s", intermediate_cols)

    total_seen = 0
    total_converted = 0
    unmapped: set[str] = set()

    for col in intermediate_cols:
        original = df[col].copy()
        df[col] = df[col].apply(convert_to_hgnc_symbol)
        is_uniprot = original.astype(str).str.contains("uniprot", na=False)
        was_changed = original != df[col]
        total_seen += int(is_uniprot.sum())
        total_converted += int((is_uniprot & was_changed).sum())
        unmapped.update(original[is_uniprot & ~was_changed].tolist())

    logger.info("UniProt IDs found: %d, converted: %d, unmapped: %d", total_seen, total_converted, len(unmapped))
    return df, total_seen, total_converted, unmapped


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
        description="Convert UniProt IDs to HGNC symbols in multi-hop result CSVs.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Input CSV with intermediate columns.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for UniProt-to-HGNC conversion CLI."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Loading %s", args.input)
    df = pd.read_csv(args.input)

    df, total_seen, total_converted, unmapped = convert_dataframe(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved converted file to %s", args.output)

    if unmapped:
        unmapped_path = args.output.with_name(args.output.stem + "_unmapped.txt")
        unmapped_path.write_text("\n".join(sorted(unmapped)) + "\n")
        logger.info("Unmapped UniProt IDs saved to %s", unmapped_path)


if __name__ == "__main__":
    main()
