"""Service adapter for db.indra.bio and INDRA REST calls.
This module provides service adapters for external APIs and data backends.
"""

from __future__ import annotations

import json
import logging
import urllib.request

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_USER_AGENT = "indra-perturbseq"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def fetch_statement_json(
    stmt_hash: int,
    timeout_s: int = 60,
    user_agent: str = _USER_AGENT,
) -> dict:
    """Fetch statement JSON payload from db.indra.bio by statement hash."""
    url = f"https://db.indra.bio/statements/from_hash/{stmt_hash}?format=json"
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def safe_fetch_statement_json(
    stmt_hash: int,
    timeout_s: int = 60,
    user_agent: str = _USER_AGENT,
) -> dict | None:
    """Fetch statement JSON, returning ``None`` on failure."""
    try:
        return fetch_statement_json(
            stmt_hash=stmt_hash,
            timeout_s=timeout_s,
            user_agent=user_agent,
        )
    except Exception as exc:
        logger.warning("db.indra.bio fetch failed for hash %s: %s", stmt_hash, exc)
        return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def get_statements_with_retry(**kwargs):
    """Query INDRA REST with retry."""
    from indra.sources import indra_db_rest

    return indra_db_rest.get_statements(**kwargs)


def safe_get_statements(**kwargs):
    """Query INDRA REST, returning ``None`` on failure."""
    try:
        return get_statements_with_retry(**kwargs)
    except Exception as exc:
        logger.warning("INDRA REST query failed (kwargs=%s): %s", kwargs, exc)
        return None
