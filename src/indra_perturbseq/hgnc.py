"""HGNC symbol normalization."""

from __future__ import annotations

import pandas as pd
from indra.databases import hgnc_client


def normalize_hgnc_symbol(symbol: str | float | None) -> str | None:
    """Return the current HGNC symbol for *symbol*, or ``None``.

    If HGNC returns multiple IDs the lexicographically first is chosen
    so that results are deterministic.

    Parameters
    ----------
    symbol :
        Raw gene symbol (or ``None`` / ``NaN``).

    Returns
    -------
    :
        Current HGNC symbol, uppercased original if lookup fails,
        or ``None`` for missing / empty input.
    """
    if symbol is None or (isinstance(symbol, float) and pd.isna(symbol)):
        return None
    s = str(symbol).strip()
    if not s:
        return None

    hid = hgnc_client.get_current_hgnc_id(s.upper())
    if not hid:
        return s.upper()

    if isinstance(hid, (list, tuple, set)):
        hid = sorted(hid)[0] if hid else None
        if not hid:
            return s.upper()

    name = hgnc_client.get_hgnc_name(hid) if hid else None
    return name or s.upper()
