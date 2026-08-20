"""Neo4j access through INDRA CoGEx."""

from __future__ import annotations

import logging

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def get_neo4j_client():
    """Create and return a Neo4j client."""
    from indra_cogex.client.neo4j_client import Neo4jClient

    return Neo4jClient()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def query_tx_with_retry(client, query: str, **kwargs):
    """Run a Neo4j query with retry."""
    return client.query_tx(query, **kwargs)


def safe_query_tx(client, query: str, **kwargs):
    """Run a Neo4j query, returning an empty list on failure."""
    try:
        rows = query_tx_with_retry(client, query, **kwargs)
        return rows or []
    except Exception as exc:
        logger.warning("Neo4j query failed: %s", exc)
        return []


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def get_mesh_ids_for_pmids_with_retry(pmids: list[str], client):
    """Fetch PMID -> MeSH mapping with retry."""
    from indra_cogex.client import get_mesh_ids_for_pmids

    return get_mesh_ids_for_pmids(pmids, client=client)


def safe_get_mesh_ids_for_pmids(pmids: list[str], client) -> dict[str, list[str]]:
    """Fetch PMID -> MeSH mapping, returning an empty map on failure."""
    try:
        result = get_mesh_ids_for_pmids_with_retry(pmids, client=client)
        return result or {}
    except Exception as exc:
        logger.warning("MeSH PMID lookup failed for batch of %d PMIDs: %s", len(pmids), exc)
        return {}
