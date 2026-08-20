"""Evidence and optional MeSH enrichment for pipeline result tables."""

from __future__ import annotations

import logging

import pandas as pd

from indra_perturbseq.indra_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


def _rename_existing(df: pd.DataFrame, renames: dict[str, str]) -> pd.DataFrame:
    present = {src: dst for src, dst in renames.items() if src in df.columns}
    return df.rename(columns=present)


def enrich_1hop(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Add evidence, PMID, and optional MeSH columns to 1-hop results."""
    if df.empty:
        return df
    out = df.copy()
    if cfg.evidence.enabled:
        from indra_perturbseq.evidence import enrich_evidence

        out = enrich_evidence(
            out,
            hop_hash_columns={1: "stmt_hash"},
            neo4j_batch_size=cfg.evidence.neo4j_batch_size,
            max_texts_per_statement=cfg.evidence.max_evidence_texts_per_statement,
        )
        out = _rename_existing(out, {
            "evidence_text_hop1": "evidence_text",
            "pmids_hop1": "pmids",
        })
    else:
        out["evidence_text"] = ""
        out["pmids"] = ""

    if cfg.mesh.enabled:
        if "pmids" not in out.columns:
            logger.warning("Skipping MeSH: 1-hop results have no PMID column.")
            return out
        from indra_perturbseq.mesh import annotate_mesh

        temp = out.rename(columns={"pmids": "pmids_hop1"})
        temp = annotate_mesh(
            temp,
            cfg.mesh.terms_path,
            pmid_columns=["pmids_hop1"],
            mesh_batch_size=cfg.mesh.batch_size,
        )
        temp = _rename_existing(temp, {
            "pmids_hop1": "pmids",
            "Annotated MeSH terms hop1": "mesh_terms",
        })
        out = temp
    return out


def enrich_2hop(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Add evidence, PMID, and optional MeSH columns to 2-hop results."""
    if df.empty:
        return df
    out = df.copy()
    if cfg.evidence.enabled:
        from indra_perturbseq.evidence import enrich_evidence

        out = enrich_evidence(
            out,
            hop_hash_columns={1: "stmt_hash_1", 2: "stmt_hash_2"},
            neo4j_batch_size=cfg.evidence.neo4j_batch_size,
            max_texts_per_statement=cfg.evidence.max_evidence_texts_per_statement,
        )
        out = _rename_existing(out, {
            "evidence_text_hop1": "evidence_text_1",
            "evidence_text_hop2": "evidence_text_2",
            "pmids_hop1": "pmids_1",
            "pmids_hop2": "pmids_2",
        })
    else:
        out["evidence_text_1"] = ""
        out["evidence_text_2"] = ""
        out["pmids_1"] = ""
        out["pmids_2"] = ""

    if cfg.mesh.enabled:
        if "pmids_1" not in out.columns or "pmids_2" not in out.columns:
            logger.warning("Skipping MeSH: 2-hop results have no PMID columns.")
            return out
        from indra_perturbseq.mesh import annotate_mesh

        temp = out.rename(columns={"pmids_1": "pmids_hop1", "pmids_2": "pmids_hop2"})
        temp = annotate_mesh(
            temp,
            cfg.mesh.terms_path,
            pmid_columns=["pmids_hop1", "pmids_hop2"],
            mesh_batch_size=cfg.mesh.batch_size,
        )
        temp = _rename_existing(temp, {
            "pmids_hop1": "pmids_1",
            "pmids_hop2": "pmids_2",
            "Annotated MeSH terms hop1": "mesh_terms_1",
            "Annotated MeSH terms hop2": "mesh_terms_2",
        })
        out = temp
    return out
