"""Map C14 cardiovascular-tree names to MeSH descriptor IDs.
Parses a pasted MeSH C14 tree listing, maps the extracted names to MeSH.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
from indra_cogex.client.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

C14_LINE_RE = re.compile(
    r"^(?P<name>.+?)\s*\[(?P<tree>C14(?:\.\d+)*?)\]\s*$",
    re.IGNORECASE,
)

_NOISE_NAMES = frozenset({"details", "qualifiers", "mesh tree structures", "concepts"})


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_c14_names_from_text(text: str) -> list[str]:
    """Extract unique C14 category names from a MeSH tree listing.

    Parameters
    ----------
    text
        Raw text copied from the MeSH browser C14 tree view.

    Returns
    -------
    list[str]
        De-duplicated names in insertion order.
    """
    names: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = C14_LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name").strip()
        if name.lower() in _NOISE_NAMES:
            continue
        names.append(name)

    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


def normalize_mesh_id(mid: str) -> str:
    """Return the ID if it looks like a MeSH descriptor (``D\\d+``), else ``""``."""
    mid = str(mid or "").strip().upper()
    return mid if re.fullmatch(r"D\d{5,10}", mid) else ""


# ---------------------------------------------------------------------------
# Neo4j queries
# ---------------------------------------------------------------------------
def map_names_to_mesh_ids(
    client: Neo4jClient,
    names: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Map C14 names to MeSH descriptor IDs via case-insensitive match.

    Parameters
    ----------
    client
        Active Neo4j client.
    names
        List of C14 category names.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        ``(mapped_df, unmapped_names)``.  The DataFrame has columns
        ``mesh_id, mesh_name, origin, child_of``.
    """
    empty = pd.DataFrame(columns=["mesh_id", "mesh_name", "origin", "child_of"])
    if not names:
        return empty, []

    query = (
        "UNWIND $names AS n "
        "MATCH (b:BioEntity) "
        "WHERE b.id STARTS WITH 'mesh:' "
        "  AND b.name IS NOT NULL "
        "  AND toLower(b.name) = toLower(n) "
        "RETURN DISTINCT "
        "  n AS input_name, "
        "  replace(b.id, 'mesh:', '') AS mesh_id, "
        "  b.name AS mesh_name"
    )
    rows = client.query_tx(query, names=names)

    mapped: list[dict[str, str]] = []
    hit_lower: set[str] = set()
    for r in rows:
        if isinstance(r, dict):
            input_name = r.get("input_name", "")
            mesh_id = r.get("mesh_id", "")
            mesh_name = r.get("mesh_name", "")
        else:
            input_name = r[0] if len(r) > 0 else ""
            mesh_id = r[1] if len(r) > 1 else ""
            mesh_name = r[2] if len(r) > 2 else ""

        if input_name:
            hit_lower.add(str(input_name).lower())
        mapped.append({
            "mesh_id": mesh_id,
            "mesh_name": mesh_name,
            "origin": "C14_from_pasted_tree",
            "child_of": "",
        })

    unmapped = [n for n in names if n.lower() not in hit_lower]

    df = pd.DataFrame(mapped)
    if not df.empty:
        df["mesh_id"] = df["mesh_id"].apply(normalize_mesh_id)
        df = df[df["mesh_id"] != ""].copy()

    return df, unmapped


def fetch_endothelium_keyword_mesh(client: Neo4jClient) -> pd.DataFrame:
    """Fetch MeSH descriptors whose names contain *endothelial* or *endothelium*.

    Parameters
    ----------
    client
        Active Neo4j client.

    Returns
    -------
    pd.DataFrame
        Columns ``mesh_id, mesh_name, origin, child_of``.
    """
    query = (
        "MATCH (b:BioEntity) "
        "WHERE b.id STARTS WITH 'mesh:' "
        "  AND b.name IS NOT NULL "
        "  AND (toLower(b.name) CONTAINS 'endothelial' "
        "       OR toLower(b.name) CONTAINS 'endothelium') "
        "RETURN DISTINCT "
        "  replace(b.id, 'mesh:', '') AS mesh_id, "
        "  b.name AS mesh_name"
    )
    rows = client.query_tx(query)

    records: list[dict[str, str]] = []
    for r in rows:
        if isinstance(r, dict):
            mesh_id = r.get("mesh_id", "")
            mesh_name = r.get("mesh_name", "")
        else:
            mesh_id = r[0] if len(r) > 0 else ""
            mesh_name = r[1] if len(r) > 1 else ""
        records.append({
            "mesh_id": mesh_id,
            "mesh_name": mesh_name,
            "origin": "endothelium_keyword",
            "child_of": "",
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["mesh_id"] = df["mesh_id"].apply(normalize_mesh_id)
        df = df[df["mesh_id"] != ""].copy()
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    """CLI entry point for C14 MeSH mapping."""
    parser = argparse.ArgumentParser(
        description="Map C14 tree names to MeSH descriptor IDs.",
    )
    parser.add_argument(
        "--c14-text-file",
        required=True,
        help="Text file containing the pasted C14 tree listing.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output reference CSV (mesh_id, mesh_name, origin, child_of).",
    )
    parser.add_argument(
        "--unmapped-out",
        default="",
        help="Optional path to write unmapped C14 names.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    text = Path(args.c14_text_file).read_text(encoding="utf-8")
    c14_names = parse_c14_names_from_text(text)
    logger.info("Parsed %d unique C14 names from text", len(c14_names))

    client = Neo4jClient()

    df_c14, unmapped = map_names_to_mesh_ids(client, c14_names)
    logger.info("Mapped %d C14 names to MeSH descriptor IDs", len(df_c14))
    logger.info("Unmapped C14 names: %d", len(unmapped))

    df_endo = fetch_endothelium_keyword_mesh(client)
    logger.info("Keyword endothelium/endothelial matches: %d", len(df_endo))

    df = pd.concat([df_c14, df_endo], ignore_index=True)
    df["mesh_id"] = df["mesh_id"].apply(normalize_mesh_id)
    df = df[df["mesh_id"] != ""].copy()

    origin_rank = {"C14_from_pasted_tree": 0, "endothelium_keyword": 1}
    df["_rank"] = df["origin"].map(lambda x: origin_rank.get(x, 99))
    df = df.sort_values(["mesh_id", "_rank"]).drop_duplicates(
        subset=["mesh_id"], keep="first",
    )
    df = df.drop(columns=["_rank"]).sort_values("mesh_id").reset_index(drop=True)

    df.to_csv(args.output_csv, index=False)
    logger.info("Wrote %d MeSH descriptor IDs to %s", len(df), args.output_csv)

    if args.unmapped_out:
        Path(args.unmapped_out).write_text(
            "\n".join(unmapped) + "\n", encoding="utf-8",
        )
        logger.info("Wrote unmapped C14 names to %s", args.unmapped_out)


if __name__ == "__main__":
    main()
