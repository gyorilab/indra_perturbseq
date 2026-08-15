"""Evidence fetching from db.indra.bio and Neo4j.
This module provides shared utilities used across the INDRA Perturb-seq codebase.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from threading import Lock

import numpy as np
import pandas as pd
from indra_perturbseq.services.indra_db import fetch_statement_json
from indra_perturbseq.services.neo4j import get_neo4j_client, safe_query_tx

logger = logging.getLogger(__name__)

_hash_cache_lock = Lock()
_hash_cache: dict[int, dict] = {}


# db.indra.bio (REST)

def fetch_from_hash_json(stmt_hash: int) -> dict:
    """Fetch statement JSON from db.indra.bio by *stmt_hash*."""
    return fetch_statement_json(stmt_hash)


def rich_stmt_text_from_hash(stmt_hash: int, cache: dict) -> str:
    """Return human-readable statement text, caching results in *cache*."""
    if stmt_hash in cache:
        return cache[stmt_hash]
    txt = ""
    try:
        data = fetch_from_hash_json(int(stmt_hash))
        payload = (data.get("statements", {}) or {}).get(str(stmt_hash))
        if isinstance(payload, str):
            try:
                import json

                payload = json.loads(payload)
            except Exception:
                payload = None
        if isinstance(payload, dict):
            txt = payload.get("english", "") or ""
    except Exception:
        txt = ""
    cache[stmt_hash] = txt
    return txt


def evidence_from_hash(
    stmt_hash: int,
    max_texts: int = 20,
) -> tuple[str, list[str], OrderedDict]:
    """Fetch evidence text, PMIDs, and source APIs from db.indra.bio.

    Returns ``(evidence_text, pmids, sources_seen)``.
    """
    with _hash_cache_lock:
        if stmt_hash in _hash_cache:
            c = _hash_cache[stmt_hash]
            return c["evidence_text"], c["pmids"], c["sources_seen"]

    pmids: list[str] = []
    texts: list[str] = []
    sources_seen: OrderedDict[str, None] = OrderedDict()

    try:
        data = fetch_from_hash_json(stmt_hash)
        ev_list = (
            (data.get("results", {}) or {})
            .get(str(stmt_hash), {})
            .get("evidence", [])
        ) or []

        for ev in ev_list:
            if not isinstance(ev, dict):
                continue
            pmid = ev.get("pmid")
            if pmid:
                pmids.append(str(pmid))
            source_api = ev.get("source_api", "") or ""
            source_sub = (
                (ev.get("annotations", {}) or {}).get("source_sub_id", "") or ""
            )
            key = f"{source_api}:{source_sub}" if source_sub else source_api
            if key:
                sources_seen[key] = None
            txt = ev.get("text")
            if txt:
                texts.append(str(txt).strip())
    except Exception:
        pass

    pmids = list(dict.fromkeys(pmids))
    texts = list(dict.fromkeys(texts))[:max_texts]

    if texts:
        evidence_text = "\n".join(
            f"{i}. {t}" for i, t in enumerate(texts, start=1)
        )
    elif sources_seen:
        evidence_text = f"Evidence from: {', '.join(sources_seen.keys())}"
    else:
        evidence_text = "No evidence returned (db.indra.bio)"

    with _hash_cache_lock:
        _hash_cache[stmt_hash] = {
            "pmids": pmids,
            "evidence_text": evidence_text,
            "sources_seen": sources_seen,
        }
    return evidence_text, pmids, sources_seen


def format_evidence_text(text: str) -> str:
    """Re-format numbered evidence text for readability."""
    if not isinstance(text, str):
        return text
    if text.startswith(("Evidence from:", "No evidence")):
        return text
    pattern = r"(^|\n|; )(\d+\.\s)"
    parts: list[str] = []
    last_idx = 0
    for m in re.finditer(pattern, text):
        start = m.start(2)
        if start > last_idx:
            parts.append(text[last_idx:start].strip())
        last_idx = start
    parts.append(text[last_idx:].strip())
    return "\n\n".join(
        re.sub(r"^(\d+)\.\s", r"\1) ", p) for p in parts if p
    )


# ------------------------------------------------------------------
# Neo4j Evidence-node enrichment
# ------------------------------------------------------------------

def _parse_evidence_node(ev_obj) -> tuple[str | None, str | None, str | None]:
    """Extract (text, pmid, source_key) from a Neo4j evidence object."""
    if ev_obj is None:
        return None, None, None
    if isinstance(ev_obj, str):
        return ev_obj.strip(), None, None
    if not isinstance(ev_obj, dict):
        return None, None, None

    d = ev_obj.get("evidence", ev_obj) if isinstance(
        ev_obj.get("evidence"), dict
    ) else ev_obj
    txt = d.get("text") or ev_obj.get("text")
    pmid = d.get("pmid") or ev_obj.get("pmid")
    source_api = d.get("source_api") or ev_obj.get("source_api")
    annotations = d.get("annotations") if isinstance(d.get("annotations"), dict) else {}
    sub_id = annotations.get("source_sub_id") if isinstance(annotations, dict) else None
    source_key = f"{source_api}:{sub_id}" if source_api and sub_id else source_api

    return (
        str(txt).strip() if txt else None,
        str(pmid).strip() if pmid else None,
        str(source_key).strip() if source_key else None,
    )


def fetch_evidence_from_neo4j(
    hashes: list[int],
    batch_size: int = 2000,
) -> dict[int, dict]:
    """Batch-fetch evidence from Neo4j Evidence nodes by stmt_hash."""
    client = get_neo4j_client()
    query = """
    UNWIND $hashes AS h
    MATCH (e:Evidence {stmt_hash: h})
    RETURN h AS stmt_hash, collect(e) AS ev_nodes
    """
    out: dict[int, dict] = {}
    for i in range(0, len(hashes), batch_size):
        batch = hashes[i:i + batch_size]
        for h in batch:
            out[int(h)] = {
                "evidence_text": "No evidence found (Neo4j)",
                "pmids": [],
                "sources": [],
            }
        recs = safe_query_tx(
            client,
            query,
            parameters={"hashes": [int(x) for x in batch]},
        )
        for r in recs:
            if isinstance(r, dict):
                sh, ev_nodes = r.get("stmt_hash"), r.get("ev_nodes", [])
            else:
                sh, ev_nodes = r[0], r[1] if len(r) > 1 else []
            if sh is None:
                continue

            texts, pmids = [], []
            sources: OrderedDict[str, None] = OrderedDict()
            for ev in ev_nodes:
                txt, pmid, skey = _parse_evidence_node(ev)
                if txt:
                    texts.append(txt)
                if pmid and pmid.isdigit():
                    pmids.append(pmid)
                if skey:
                    sources[skey] = None

            texts = list(dict.fromkeys(texts))
            pmids = list(dict.fromkeys(pmids))

            if texts:
                ev_text = "\n".join(
                    f"{j}. {t}" for j, t in enumerate(texts, start=1)
                )
            elif sources:
                ev_text = f"Evidence from: {', '.join(sources.keys())}"
            else:
                ev_text = "No evidence found (Neo4j)"

            out[int(sh)] = {
                "evidence_text": ev_text,
                "pmids": pmids,
                "sources": list(sources.keys()),
            }
    return out


def enrich_evidence(
    df: pd.DataFrame,
    hop_hash_columns: dict[int, str] | None = None,
    neo4j_batch_size: int = 2000,
) -> pd.DataFrame:
    """Add evidence text and PMIDs columns from Neo4j Evidence nodes.

    Parameters
    ----------
    df :
        Input dataframe containing statement hash columns.
    hop_hash_columns :
        Optional mapping from hop number to hash column name.
        When omitted, uses ``hopN_hash`` columns discovered on *df*.
    neo4j_batch_size :
        Batch size for Neo4j statement-hash queries.
    """
    df = df.copy()
    if hop_hash_columns is None:
        hop_hash_columns = {}
        for c in df.columns:
            m = re.fullmatch(r"hop(\d+)_hash", str(c))
            if m:
                hop_hash_columns[int(m.group(1))] = str(c)
    if not hop_hash_columns:
        return df

    for hop in sorted(hop_hash_columns):
        ev_col = f"evidence_text_hop{hop}"
        pm_col = f"pmids_hop{hop}"
        if ev_col not in df.columns:
            df[ev_col] = ""
        if pm_col not in df.columns:
            df[pm_col] = ""

    hashes: set[int] = set()
    for hash_col in hop_hash_columns.values():
        if hash_col not in df.columns:
            continue
        vals = pd.to_numeric(df[hash_col], errors="coerce").dropna().astype(np.int64)
        hashes.update(int(v) for v in vals)

    logger.info("Evidence fetch (Neo4j): %d unique stmt_hashes", len(hashes))
    if not hashes:
        return df

    ev_map = fetch_evidence_from_neo4j(sorted(hashes), batch_size=neo4j_batch_size)

    for idx in df.index:
        for hop, hash_col in hop_hash_columns.items():
            h = df.at[idx, hash_col] if hash_col in df.columns else None
            ev_col = f"evidence_text_hop{hop}"
            pm_col = f"pmids_hop{hop}"
            if not isinstance(h, (int, np.integer)):
                try:
                    h = int(h)
                except Exception:
                    df.at[idx, ev_col] = "No evidence found (Neo4j)"
                    df.at[idx, pm_col] = ""
                    continue
            d = ev_map.get(int(h))
            if d:
                df.at[idx, ev_col] = format_evidence_text(d["evidence_text"])
                df.at[idx, pm_col] = "; ".join(d["pmids"])
            else:
                df.at[idx, ev_col] = "No evidence found (Neo4j)"
                df.at[idx, pm_col] = ""
    return df


def enrich_evidence_1hop(
    df: pd.DataFrame,
    neo4j_batch_size: int = 2000,
) -> pd.DataFrame:
    """Add evidence text and PMIDs columns for hop1 from Neo4j."""
    return enrich_evidence(
        df,
        hop_hash_columns={1: "stmt_hash"},
        neo4j_batch_size=neo4j_batch_size,
    )


def enrich_evidence_2hop(
    df: pd.DataFrame,
    neo4j_batch_size: int = 2000,
) -> pd.DataFrame:
    """Add evidence text and PMIDs columns for hop1/hop2 from Neo4j."""
    return enrich_evidence(
        df,
        hop_hash_columns={1: "hop1_hash", 2: "hop2_hash"},
        neo4j_batch_size=neo4j_batch_size,
    )


def enrich_evidence_3hop(
    df: pd.DataFrame,
    neo4j_batch_size: int = 2000,
) -> pd.DataFrame:
    """Add evidence text and PMIDs columns for hop1/hop2/hop3 from Neo4j."""
    return enrich_evidence(
        df,
        hop_hash_columns={1: "hop1_hash", 2: "hop2_hash", 3: "hop3_hash"},
        neo4j_batch_size=neo4j_batch_size,
    )


def enrich_evidence_4hop(
    df: pd.DataFrame,
    neo4j_batch_size: int = 2000,
) -> pd.DataFrame:
    """Add evidence text and PMIDs columns for hop1--hop4 from Neo4j."""
    return enrich_evidence(
        df,
        hop_hash_columns={1: "hop1_hash", 2: "hop2_hash", 3: "hop3_hash", 4: "hop4_hash"},
        neo4j_batch_size=neo4j_batch_size,
    )
