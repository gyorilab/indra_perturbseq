"""MeSH term annotation via INDRA CoGEx."""

from __future__ import annotations

import logging
import re

import pandas as pd
from indra_perturbseq.services.neo4j import (
    get_neo4j_client,
    safe_get_mesh_ids_for_pmids,
    safe_query_tx,
)

logger = logging.getLogger(__name__)


def load_valid_mesh_ids(reference_csv: str) -> set[str]:
    """Load valid MeSH IDs from a reference CSV.

    Looks for a ``mesh_id`` column (case-insensitive), normalises to
    uppercase alphanumeric IDs starting with ``D``.
    """
    ref = pd.read_csv(
        reference_csv, encoding="utf-8-sig",
        on_bad_lines="skip", low_memory=False,
    )
    cols_lower = {c.lower(): c for c in ref.columns}
    mesh_id_col = cols_lower.get("mesh_id")
    if not mesh_id_col:
        for c in ref.columns:
            if "mesh" in c.lower() and "id" in c.lower():
                mesh_id_col = c
                break
    if not mesh_id_col:
        raise ValueError(
            f"No MeSH ID column in {reference_csv}. "
            f"Columns: {ref.columns.tolist()}"
        )
    s = ref[mesh_id_col].astype(str)
    s = (s.str.replace(r"\s+", "", regex=True)
          .str.replace(r"[^A-Za-z0-9]", "", regex=True)
          .str.upper())
    return set(s[s.str.startswith("D", na=False)])


def build_mesh_id_to_name_map(client) -> dict[str, str]:
    """Map MeSH IDs (e.g. ``D012345``) to names via Neo4j BioEntity nodes."""
    query = (
        "MATCH (b:BioEntity) WHERE b.id STARTS WITH 'mesh:' "
        "RETURN b.id AS mesh_id, b.name AS mesh_name"
    )
    mapping: dict[str, str] = {}
    for r in safe_query_tx(client, query):
        if isinstance(r, dict):
            mid, name = r.get("mesh_id"), r.get("mesh_name")
        else:
            mid, name = r[0], r[1]
        if mid and name:
            mapping[str(mid).replace("mesh:", "").upper()] = str(name)
    return mapping


def _collect_pmids_from_column(series: pd.Series) -> set[str]:
    """Extract all numeric PMIDs from a semicolon-separated column."""
    pmids: set[str] = set()
    for cell in series.fillna("").astype(str):
        for tok in cell.replace(" ", "").split(";"):
            if tok.isdigit():
                pmids.add(tok)
    return pmids


def annotate_mesh(
    df: pd.DataFrame,
    mesh_reference_csv: str,
    pmid_columns: list[str] | None = None,
    mesh_batch_size: int = 200,
) -> pd.DataFrame:
    """Add MeSH annotation columns for the configured PMID columns."""
    if pmid_columns is None:
        pmid_columns = [c for c in ("pmids_hop1", "pmids_hop2", "pmids_hop3")
                        if c in df.columns]

    df = df.copy()
    for pc in pmid_columns:
        hop = re.search(r"hop(\d+)", pc)
        if hop:
            out_col = f"Annotated MeSH terms hop{hop.group(1)}"
        else:
            out_col = f"Annotated MeSH terms {pc}"
        if out_col not in df.columns:
            df[out_col] = ""

    valid_ids = load_valid_mesh_ids(mesh_reference_csv)

    all_pmids: set[str] = set()
    for pc in pmid_columns:
        all_pmids |= _collect_pmids_from_column(df[pc])
    all_pmids_sorted = sorted(all_pmids, key=int)

    if not all_pmids_sorted:
        return df

    client = get_neo4j_client()
    id_to_name = build_mesh_id_to_name_map(client)

    pmid_to_mesh: dict[str, list] = {}
    for i in range(0, len(all_pmids_sorted), mesh_batch_size):
        batch = all_pmids_sorted[i:i + mesh_batch_size]
        pmid_to_mesh.update(safe_get_mesh_ids_for_pmids(batch, client=client))

    def _mesh_for_cell(cell: str) -> str:
        pmids = [p for p in str(cell).replace(" ", "").split(";") if p.isdigit()]
        mesh_ids: set[str] = set()
        for p in pmids:
            mesh_ids.update(pmid_to_mesh.get(p, []) or [])
        kept = []
        for mid in sorted(str(m).upper().strip() for m in mesh_ids):
            if mid in valid_ids:
                name = id_to_name.get(mid)
                kept.append(f"{name} ({mid})" if name else mid)
        return ", ".join(kept)

    for pc in pmid_columns:
        hop = re.search(r"hop(\d+)", pc)
        if hop:
            out_col = f"Annotated MeSH terms hop{hop.group(1)}"
        else:
            out_col = f"Annotated MeSH terms {pc}"
        df[out_col] = df[pc].apply(_mesh_for_cell)

    return df
