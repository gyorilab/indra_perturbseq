"""Service adapters for external systems (INDRA REST / Neo4j)."""

from .indra_db import (
    fetch_statement_json,
    get_statements_with_retry,
    safe_fetch_statement_json,
    safe_get_statements,
)
from .neo4j import (
    get_neo4j_client,
    query_tx_with_retry,
    safe_get_mesh_ids_for_pmids,
    safe_query_tx,
)

__all__ = [
    "fetch_statement_json",
    "safe_fetch_statement_json",
    "get_statements_with_retry",
    "safe_get_statements",
    "get_neo4j_client",
    "query_tx_with_retry",
    "safe_query_tx",
    "safe_get_mesh_ids_for_pmids",
]
