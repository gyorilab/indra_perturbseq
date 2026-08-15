"""Shared optional enrichment helpers for hop pipelines.
Provides CLI args and reusable evidence/MeSH post-processing stages."""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from indra_perturbseq.evidence import enrich_evidence as enrich_evidence_rows
from indra_perturbseq.mesh import annotate_mesh

logger = logging.getLogger(__name__)


def add_enrichment_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add optional evidence and MeSH enrichment arguments."""
    parser.add_argument(
        "--enable-evidence-enrichment",
        action="store_true",
        help="Populate evidence_text_hop* and pmids_hop* columns from Neo4j.",
    )
    parser.add_argument(
        "--enable-mesh-annotation",
        action="store_true",
        help="Annotate MeSH terms from pmids_hop* columns.",
    )
    parser.add_argument(
        "--mesh-reference",
        default=None,
        help="MeSH reference CSV used to filter valid MeSH IDs.",
    )
    parser.add_argument(
        "--neo4j-evidence-batch-size",
        type=int,
        default=2000,
        help="Batch size for Neo4j Evidence-node lookup.",
    )
    parser.add_argument(
        "--mesh-batch-size",
        type=int,
        default=200,
        help="Batch size for PMID -> MeSH lookup.",
    )


def validate_enrichment_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate enrichment-related arguments."""
    if args.enable_mesh_annotation and not args.mesh_reference:
        parser.error("--mesh-reference is required when --enable-mesh-annotation is set.")


def enrich_evidence(
    df: pd.DataFrame,
    args: argparse.Namespace,
    hop_hash_columns: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Conditionally enrich evidence text and PMIDs from Neo4j."""
    if not args.enable_evidence_enrichment:
        logger.info("Skipping evidence enrichment (flag not set).")
        return df
    logger.info("Enriching evidence + PMIDs from Neo4j...")
    return enrich_evidence_rows(
        df,
        hop_hash_columns=hop_hash_columns,
        neo4j_batch_size=args.neo4j_evidence_batch_size,
    )


def annotate_mesh_terms(
    df: pd.DataFrame,
    args: argparse.Namespace,
    pmid_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Conditionally annotate MeSH terms using available PMID columns."""
    if not args.enable_mesh_annotation:
        logger.info("Skipping MeSH annotation (flag not set).")
        return df

    candidate_cols = pmid_columns or [c for c in df.columns if c.startswith("pmids_hop")]
    candidate_cols = [c for c in candidate_cols if c in df.columns]
    if not candidate_cols:
        logger.warning("Skipping MeSH annotation: no pmids_hop* columns found.")
        return df

    logger.info("Annotating MeSH terms...")
    return annotate_mesh(
        df,
        args.mesh_reference,
        pmid_columns=candidate_cols,
        mesh_batch_size=args.mesh_batch_size,
    )
