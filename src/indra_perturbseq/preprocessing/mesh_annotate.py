"""Annotate pathway CSVs with MeSH terms and filter by reference list.
This module provides preprocessing utilities and command-line data preparation workflows.
"""
from __future__ import annotations

import argparse
import logging
import re

import pandas as pd
from indra_cogex.client import get_mesh_ids_for_pmids
from indra_cogex.client.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

_MESH_ID_RE = re.compile(r"D\d{5,10}")


def _parse_pmids_cell(value: object) -> list[str]:
    """Parse semicolon-separated PMIDs from a single cell value."""
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return []
    return [tok for tok in text.replace(" ", "").split(";") if tok.isdigit()]


def _batch_iter(lst: list[str], n: int):
    """Yield successive *n*-sized slices of *lst*."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _is_valid_mesh_id(mid: str) -> bool:
    return bool(re.fullmatch(r"[DC]\d{6,7}", mid))


def load_valid_mesh_ids(reference_csv: str) -> set[str]:
    """Load reference CSV and return valid descriptor IDs."""
    ref = pd.read_csv(
        reference_csv, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False,
    )
    if "mesh_id" not in ref.columns:
        raise ValueError(
            f"Reference CSV must contain 'mesh_id'. Columns={ref.columns.tolist()}"
        )
    cleaned = (
        ref["mesh_id"]
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.replace(r"[^A-Za-z0-9]", "", regex=True)
        .str.upper()
    )
    return set(cleaned[cleaned.str.startswith("D", na=False)].tolist())


def _collect_all_pmids(df: pd.DataFrame, columns: list[str]) -> list[str]:
    pmids: set[str] = set()
    for col in columns:
        for val in df[col].tolist():
            pmids.update(_parse_pmids_cell(val))
    return sorted(pmids, key=int)


def _build_pmid_to_mesh(
    pmids: list[str],
    client: Neo4jClient,
    batch_size: int,
) -> dict[str, list[str]]:
    pmid_to_mesh: dict[str, list[str]] = {}
    total = (len(pmids) + batch_size - 1) // batch_size
    for i, batch in enumerate(_batch_iter(pmids, batch_size), start=1):
        pmid_to_mesh.update(get_mesh_ids_for_pmids(batch, client=client))
        logger.info("Batch %d/%d processed (%d PMIDs)", i, total, len(batch))
    return pmid_to_mesh


def _build_mesh_id_to_name(
    client: Neo4jClient,
    mesh_ids: list[str],
) -> dict[str, str]:
    """Fetch human-readable names for a list of MeSH IDs."""
    if not mesh_ids:
        return {}
    curies = [f"mesh:{mid}" for mid in mesh_ids]
    query = (
        "UNWIND $ids AS mid "
        "MATCH (b:BioEntity {id: mid}) "
        "RETURN b.id AS mesh_id, b.name AS mesh_name"
    )
    rows = client.query_tx(query, ids=curies)
    mapping: dict[str, str] = {}
    for r in rows:
        if isinstance(r, dict):
            mid = r.get("mesh_id", "")
            name = r.get("mesh_name", "")
        else:
            mid = r[0] if len(r) > 0 else ""
            name = r[1] if len(r) > 1 else ""
        if mid and name:
            mapping[mid.replace("mesh:", "").upper()] = str(name)
    return mapping


def _annotate_column(
    df: pd.DataFrame,
    pmid_col: str,
    pmid_to_mesh: dict[str, list[str]],
    mesh_id_to_name: dict[str, str],
    valid_ids: set[str] | None,
) -> pd.Series:
    """Build an annotation series for a single PMID column."""
    def _fmt(cell: object) -> str:
        pmids = _parse_pmids_cell(cell)
        mesh_ids: set[str] = set()
        for p in pmids:
            mesh_ids.update(pmid_to_mesh.get(p, []) or [])
        kept: list[str] = []
        for mid in sorted(mesh_ids):
            mid_u = mid.upper().strip()
            if not _is_valid_mesh_id(mid_u):
                continue
            if valid_ids is not None and mid_u not in valid_ids:
                continue
            name = mesh_id_to_name.get(mid_u)
            if name:
                kept.append(f"{name} ({mid_u})")
        return ", ".join(kept)

    return df[pmid_col].apply(_fmt)


def _resolve_hop_spec(
    mode: str,
    df: pd.DataFrame,
) -> list[tuple[str, str]]:
    """Return ``[(pmid_col, annotation_col), ...]`` for the selected mode."""
    if mode == "1hop":
        return [("pmids", "Annotated MeSH terms")]
    if mode == "2hop":
        return [
            ("pmids_hop1", "Annotated MeSH terms hop1"),
            ("pmids_hop2", "Annotated MeSH terms hop2"),
        ]
    hop_cols = sorted(c for c in df.columns if c.startswith("pmids_hop"))
    if not hop_cols:
        raise ValueError("No pmids_hop* columns found for 3hop mode")
    return [
        (col, f"Annotated MeSH terms ({col.replace('pmids_', '')})")
        for col in hop_cols
    ]


def annotate(
    df: pd.DataFrame,
    mode: str,
    reference_csv: str | None = None,
    batch_size: int = 200,
    *,
    client: Neo4jClient | None = None,
) -> pd.DataFrame:
    """Annotate a pathway DataFrame with MeSH terms from PMIDs."""
    if client is None:
        client = Neo4jClient()

    valid_ids = load_valid_mesh_ids(reference_csv) if reference_csv else None
    if valid_ids is not None:
        logger.info("Loaded %d valid MeSH IDs from reference", len(valid_ids))

    hop_spec = _resolve_hop_spec(mode, df)
    pmid_cols = [col for col, _ in hop_spec]

    for col in pmid_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    all_pmids = _collect_all_pmids(df, pmid_cols)
    logger.info("Unique PMIDs found: %d", len(all_pmids))

    pmid_to_mesh = _build_pmid_to_mesh(all_pmids, client, batch_size)

    seen_mesh: set[str] = set()
    for mids in pmid_to_mesh.values():
        for mid in mids or []:
            mid_u = mid.upper().strip()
            if re.fullmatch(r"D\d{5,10}", mid_u):
                seen_mesh.add(mid_u)
    mesh_id_to_name = _build_mesh_id_to_name(client, sorted(seen_mesh))

    df = df.copy()
    for pmid_col, out_col in hop_spec:
        df[out_col] = _annotate_column(
            df, pmid_col, pmid_to_mesh, mesh_id_to_name, valid_ids,
        )
        logger.info("Wrote column: %s", out_col)

    return df


def filter_mesh_columns(
    df: pd.DataFrame,
    reference_csv: str,
    mesh_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Filter existing MeSH annotation columns against a reference list."""
    valid_ids = load_valid_mesh_ids(reference_csv)
    logger.info("Loaded %d valid MeSH IDs from reference", len(valid_ids))

    if mesh_columns is None:
        mesh_columns = [c for c in df.columns if "mesh" in c.lower() and "annotated" in c.lower()]

    df = df.copy()

    for col in mesh_columns:
        if col not in df.columns:
            logger.warning("Column '%s' not found, skipping", col)
            continue
        df[col] = df[col].apply(lambda txt: _filter_cell(txt, valid_ids))
        logger.info("Filtered column: %s", col)

    drop_cols = [
        c for c in df.columns
        if any(s in c for s in [".1", ".2", ".3", "(copy)", "n_kept", "n_total", "n_dropped"])
    ]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        logger.info("Dropped artifact columns: %s", drop_cols)

    return df


def _filter_cell(text: object, valid_ids: set[str]) -> str:
    if pd.isna(text):
        return ""
    pairs = re.findall(r"[^,]+?\(D\d{5,10}\)", str(text))
    kept: list[str] = []
    for pair in pairs:
        match = re.search(r"\((D\d{5,10})\)", pair)
        if match and match.group(1) in valid_ids:
            kept.append(pair.strip())
    return ", ".join(kept)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point with ``annotate`` and ``filter`` subcommands."""
    parser = argparse.ArgumentParser(
        description="Annotate or filter MeSH terms in pathway CSVs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- annotate --
    ann = sub.add_parser("annotate", help="Fetch MeSH for PMIDs and annotate.")
    ann.add_argument("--input", required=True, help="Input pathway CSV.")
    ann.add_argument("--output", required=True, help="Output annotated CSV.")
    ann.add_argument(
        "--mode",
        choices=["1hop", "2hop", "3hop"],
        required=True,
        help="Hop layout of the input CSV.",
    )
    ann.add_argument(
        "--reference",
        default=None,
        help="Optional MeSH reference CSV for filtering annotations.",
    )
    ann.add_argument("--batch-size", type=int, default=200)

    # -- filter --
    filt = sub.add_parser("filter", help="Filter annotation columns by reference list.")
    filt.add_argument("--input", required=True, help="Input CSV with annotation columns.")
    filt.add_argument("--output", required=True, help="Output filtered CSV.")
    filt.add_argument("--reference", required=True, help="MeSH reference CSV.")
    filt.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Annotation columns to filter (auto-detected if omitted).",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "annotate":
        df = pd.read_csv(args.input, low_memory=False)
        logger.info("Loaded %d rows from %s", len(df), args.input)
        df = annotate(
            df,
            mode=args.mode,
            reference_csv=args.reference,
            batch_size=args.batch_size,
        )
        df.to_csv(args.output, index=False)
        logger.info("Saved annotated CSV to %s", args.output)

    elif args.command == "filter":
        df = pd.read_csv(args.input, low_memory=False)
        logger.info("Loaded %d rows from %s", len(df), args.input)
        df = filter_mesh_columns(
            df,
            reference_csv=args.reference,
            mesh_columns=args.columns,
        )
        df.to_csv(args.output, index=False)
        logger.info("Saved filtered CSV to %s", args.output)


if __name__ == "__main__":
    main()
