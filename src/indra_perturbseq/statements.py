"""Statement formatting."""

from __future__ import annotations


def indra_html_url(stmt_hash: int | None) -> str:
    """Return the db.indra.bio HTML viewer URL for a statement hash."""
    if stmt_hash is None:
        return ""
    try:
        return f"https://db.indra.bio/statements/from_hash/{int(stmt_hash)}?format=html"
    except (TypeError, ValueError):
        return ""
