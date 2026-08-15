"""Build and expand a comprehensive MeSH reference list.
This module provides preprocessing utilities and command-line data preparation workflows.
"""
from __future__ import annotations

import argparse
import logging
import time

import pandas as pd
from indra_cogex.client import get_mesh_ids_for_pmids
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.representation import norm_id

logger = logging.getLogger(__name__)

BROAD_STOPLIST: set[str] = {
    "D006801",  # Humans
    "D000818",  # Animals
    "D002477",  # Cells
    "D005260",  # Enzymes
    "D010801",  # Proteins
    "D000740",  # Anatomy
    "D009944",  # Metabolism
    "D012679",  # Signal Transduction
    "D004958",  # Genes
    "D013964",  # Subcellular Fractions
}

CAD_SEEDS: list[tuple[str, str]] = [
    ("D003324", "Coronary Artery Disease"),
    ("D001161", "Atherosclerosis"),
    ("D009203", "Myocardial Ischemia"),
    ("D009204", "Myocardial Infarction"),
    ("D006331", "Ischemic Cardiomyopathy"),
    ("D006333", "Heart Failure, Ischemic"),
    ("D014659", "Coronary Vasospasm"),
    ("D000787", "Angina Pectoris"),
    ("D054058", "Coronary Stenosis"),
    ("D003326", "Coronary Thrombosis"),
    ("D016893", "Coronary Restenosis"),
]

ENDOTHELIAL_SEEDS: list[tuple[str, str]] = [
    ("D004715", "Endothelial Cells"),
    ("D000070678", "Vascular Endothelial Cells"),
    ("D000068820", "Lymphatic Endothelial Cells"),
    ("D000068821", "Microvascular Endothelial Cells"),
    ("D000071254", "Venous Endothelial Cells"),
    ("D000070679", "Arterial Endothelial Cells"),
    ("D000071261", "Coronary Endothelial Cells"),
    ("D004717", "Endothelium, Vascular"),
    ("D004718", "Endothelium, Lymphatic"),
    ("D000068879", "Endothelial Dysfunction"),
]

HEART_SEEDS: list[tuple[str, str]] = [
    ("D006321", "Heart"),
    ("D004698", "Endocardium"),
    ("D005315", "Fetal Heart"),
    ("D006329", "Heart Atria"),
    ("D006330", "Heart Conduction System"),
    ("D006331", "Heart Septum"),
    ("D006333", "Heart Valves"),
    ("D006334", "Heart Ventricles"),
    ("D009205", "Myocardium"),
    ("D010493", "Pericardium"),
]

HEART_DISEASE_SEEDS: list[tuple[str, str]] = [
    ("D006331", "Heart Diseases"),
    ("D001145", "Arrhythmias, Cardiac"),
    ("D006301", "Carcinoid Heart Disease"),
    ("D018377", "Cardiac Conduction System Disease"),
    ("D006329", "Cardiac Output, High"),
    ("D006330", "Cardiac Output, Low"),
    ("D002319", "Cardiac Tamponade"),
    ("D006333", "Cardiomegaly"),
    ("D009203", "Cardiomyopathies"),
    ("D000068877", "Cardiotoxicity"),
    ("D004696", "Endocarditis"),
    ("D006324", "Heart Aneurysm"),
    ("D006323", "Heart Arrest"),
    ("D006402", "Heart Defects, Congenital"),
    ("D006333", "Heart Failure"),
    ("D006330", "Heart Neoplasms"),
    ("D006329", "Heart Rupture"),
    ("D006484", "Heart Valve Diseases"),
    ("D009203", "Myocardial Ischemia"),
    ("D015217", "Myocardial Stunning"),
    ("D010494", "Pericardial Effusion"),
    ("D010496", "Pericarditis"),
    ("D011005", "Pneumopericardium"),
    ("D000068879", "Post-Cardiac Arrest Syndrome"),
    ("D011339", "Postpericardiotomy Syndrome"),
    ("D011662", "Pulmonary Heart Disease"),
    ("D012213", "Rheumatic Heart Disease"),
    ("D014658", "Ventricular Dysfunction"),
    ("D014662", "Ventricular Outflow Obstruction"),
]

MECHANISTIC_SEEDS: dict[str, str] = {
    "D058506": "Coronary Artery Disease",
    "D001161": "Atherosclerosis",
    "D058226": "Plaque, Atherosclerotic",
    "D004730": "Endothelial Cells",
    "D056669": "Endothelial Dysfunction",
    "D014652": "Vascular Permeability",
    "D006439": "Hemodynamics",
    "D012711": "Shear Stress",
    "D000293": "Angiogenesis",
    "D015815": "Cell Adhesion Molecules",
    "D017207": "Extracellular Matrix",
    "D001446": "Basement Membrane",
    "D022001": "Focal Adhesions",
    "D000199": "Actins",
    "D003562": "Cytoskeleton",
    "D015398": "Signal Transduction",
    "D000076325": "Cerebral Cavernous Malformations",
    "D051741": "Kruppel-Like Transcription Factors",
    "D020928": "Mitogen-Activated Protein Kinases",
    "D020741": "Rho GTP-Binding Proteins",
    "D040542": "Mechanotransduction, Cellular",
    "D007365": "Intercellular Junctions",
    "D002462": "Cell Migration",
    "D002452": "Cell Shape",
    "D042461": "Vascular Endothelial Growth Factor A",
}


def _safe_norm_mesh(db_id: str) -> str:
    """Normalize a bare MeSH descriptor ID to CURIE."""
    try:
        curie = norm_id("mesh", db_id)
        return curie or f"mesh:{db_id}"
    except Exception:
        return f"mesh:{db_id}"


def _get_mesh_children(
    mesh_id: str,
    parent_name: str,
    depth: int | None,
    *,
    client: Neo4jClient,
) -> set[tuple[str, str, str]]:
    """Return ``(child_id, child_name, parent_name)`` for descendants."""
    curie = _safe_norm_mesh(mesh_id)
    if not curie:
        return set()

    depth_clause = f"*1..{depth}" if depth and depth > 0 else "*1.."
    query = (
        f"MATCH (c:BioEntity)-[:isa|partof{depth_clause}]->(:BioEntity {{id: $mesh_id}}) "
        "WHERE c.id STARTS WITH 'mesh:' "
        "RETURN DISTINCT c.id AS id, c.name AS name"
    )

    try:
        rows = client.query_tx(query, mesh_id=curie)
    except Exception:
        logger.warning("Error expanding %s", mesh_id, exc_info=True)
        return set()

    results: set[tuple[str, str, str]] = set()
    for row in rows:
        cid = row.get("id") if isinstance(row, dict) else row[0]
        cname = row.get("name") if isinstance(row, dict) else (row[1] if len(row) > 1 else "")
        if not cid:
            continue
        bare = cid.replace("mesh:", "").upper()
        if bare in BROAD_STOPLIST:
            continue
        results.add((bare, cname or "", parent_name))
    return results


def _expand_family(
    seeds: list[tuple[str, str]],
    depth: int | None,
    origin: str,
    client: Neo4jClient,
) -> set[tuple[str, str, str, str]]:
    """Expand one seed family and tag each row with origin/parent."""
    results: set[tuple[str, str, str, str]] = set()
    for mid, name in seeds:
        results.add((mid, name, origin, ""))
        for cid, cname, parent in _get_mesh_children(mid, name, depth, client=client):
            results.add((cid, cname, origin, parent))
    return results


def _get_names_for_mesh_ids(
    mesh_ids: list[str],
    *,
    client: Neo4jClient,
) -> dict[str, str]:
    """Fetch ``{mesh_id: mesh_name}`` for bare MeSH IDs."""
    if not mesh_ids:
        return {}
    normalized = [_safe_norm_mesh(mid) for mid in mesh_ids]
    query = (
        "UNWIND $ids AS mid "
        "MATCH (m:BioEntity {id: mid}) "
        "RETURN m.id AS id, m.name AS name"
    )
    try:
        rows = client.query_tx(query, ids=normalized)
    except Exception:
        logger.warning("Error fetching MeSH names", exc_info=True)
        return {}

    name_map: dict[str, str] = {}
    for row in rows:
        if isinstance(row, dict):
            rid = row.get("id", "").replace("mesh:", "")
            rname = row.get("name", "")
        else:
            rid = row[0].replace("mesh:", "")
            rname = row[1] if len(row) > 1 else ""
        if rid:
            name_map[rid] = rname
    return name_map


def build_mesh_reference(
    paper_pmid: str | None,
    child_depth: int,
    existing_csv: str | None = None,
    *,
    client: Neo4jClient | None = None,
) -> pd.DataFrame:
    """Build or expand the reference table with columns mesh_id/name/origin."""
    t0 = time.time()
    if client is None:
        client = Neo4jClient()

    combined: set[tuple[str, str, str, str]] = set()

    if existing_csv:
        df_old = pd.read_csv(existing_csv, dtype=str).fillna("")
        for _, row in df_old.iterrows():
            combined.add((row["mesh_id"], row["mesh_name"], row["origin"], row["child_of"]))
        logger.info("Loaded %d existing rows from %s", len(df_old), existing_csv)

    if paper_pmid:
        logger.info("Extracting MeSH from PubMed %s", paper_pmid)
        paper_map = get_mesh_ids_for_pmids([paper_pmid], client=client)
        raw_terms = paper_map.get(paper_pmid, []) or []
        for mid in raw_terms:
            if mid not in BROAD_STOPLIST:
                combined.add((mid, "", "Paper", ""))

    logger.info("Expanding seed families (depth=%d)", child_depth)
    for seeds, origin in [
        (CAD_SEEDS, "CAD"),
        (ENDOTHELIAL_SEEDS, "Endothelial"),
        (HEART_SEEDS, "Heart"),
        (HEART_DISEASE_SEEDS, "Heart Disease"),
    ]:
        combined |= _expand_family(seeds, child_depth, origin, client)

    logger.info("Expanding %d mechanistic seeds (depth=%d)", len(MECHANISTIC_SEEDS), child_depth)
    for mid, name in MECHANISTIC_SEEDS.items():
        combined.add((mid, name, "CAD Mechanistic", ""))
        for cid, cname, parent in _get_mesh_children(mid, name, child_depth, client=client):
            combined.add((cid, cname, "CAD Mechanistic", parent))

    ids_missing_name = [r[0] for r in combined if not r[1]]
    if ids_missing_name:
        name_map = _get_names_for_mesh_ids(ids_missing_name, client=client)
        updated: set[tuple[str, str, str, str]] = set()
        for mid, name, origin, parent in combined:
            if not name:
                name = name_map.get(mid, "")
            updated.add((mid, name, origin, parent))
        combined = updated

    df = pd.DataFrame(
        sorted(combined),
        columns=["mesh_id", "mesh_name", "origin", "child_of"],
    )
    df.drop_duplicates(subset=["mesh_id", "origin"], inplace=True)
    logger.info(
        "Built reference with %d rows in %.1f min",
        len(df),
        (time.time() - t0) / 60,
    )
    return df


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build or expand a comprehensive MeSH reference list.",
    )
    parser.add_argument(
        "--paper-pmid",
        default=None,
        help="PMID to extract MeSH annotations from.",
    )
    parser.add_argument(
        "--child-depth",
        type=int,
        default=3,
        help="Neo4j child traversal depth (default: 3).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for output CSV.",
    )
    parser.add_argument(
        "--existing-csv",
        default=None,
        help="Existing reference CSV to expand (append mechanistic seeds).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = build_mesh_reference(
        paper_pmid=args.paper_pmid,
        child_depth=args.child_depth,
        existing_csv=args.existing_csv,
    )
    df.to_csv(args.output, index=False)
    logger.info("Saved %d rows to %s", len(df), args.output)


if __name__ == "__main__":
    main()
