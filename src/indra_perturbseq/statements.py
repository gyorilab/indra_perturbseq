"""Select and iterate INDRA edge statements for path scoring.
Prioritizes high-confidence beliefs and optional increase/decrease filtering."""

from __future__ import annotations

import math
from typing import Iterator

INCDEC = frozenset({"IncreaseAmount", "DecreaseAmount"})


def best_statement(
    edge_data: dict | None,
    require_incdec: bool = False,
) -> dict | None:
    """Pick the highest-quality statement on an edge.

    Ranking: highest belief, tie-broken by highest evidence_count.

    Parameters
    ----------
    edge_data :
        The ``dict`` returned by ``G.get_edge_data(u, v)``.
    require_incdec :
        If ``True``, only consider IncreaseAmount / DecreaseAmount statements.
    """
    stmts = (edge_data or {}).get("statements", [])
    if not isinstance(stmts, list) or not stmts:
        return None

    best, best_key = None, None
    for s in stmts:
        if not isinstance(s, dict):
            continue
        if require_incdec and s.get("stmt_type") not in INCDEC:
            continue

        try:
            bf = float(s["belief"]) if s.get("belief") is not None else None
        except (TypeError, ValueError):
            bf = None
        try:
            evf = int(s["evidence_count"]) if s.get("evidence_count") is not None else 0
        except (TypeError, ValueError):
            evf = 0

        if bf is None or not (math.isfinite(bf) and 0.0 <= bf <= 1.0):
            continue

        key = (bf, evf)
        if best_key is None or key > best_key:
            best_key = key
            best = s

    return best


def iter_incdec_statements(
    graph,
    src: str,
    tgt: str,
) -> Iterator[dict]:
    """Yield IncreaseAmount / DecreaseAmount statement dicts on an edge."""
    stmts = (graph.get_edge_data(src, tgt) or {}).get("statements", [])
    if not isinstance(stmts, list):
        return
    for s in stmts:
        if isinstance(s, dict) and s.get("stmt_type") in INCDEC:
            yield s


def indra_html_url(stmt_hash: int | None) -> str:
    """Return the db.indra.bio HTML viewer URL for a statement hash."""
    if stmt_hash is None:
        return ""
    try:
        return f"https://db.indra.bio/statements/from_hash/{int(stmt_hash)}?format=html"
    except (TypeError, ValueError):
        return ""
