"""Normalize gene symbols to current HGNC-approved names.
Handles missing values and deterministic resolution of ambiguous IDs."""

from __future__ import annotations

import pandas as pd
from indra.databases import hgnc_client


def _canonicalize_symbol_text(symbol: str) -> str:
    """Normalize common formatting artifacts before HGNC lookup."""
    s = symbol.replace("\u00a0", " ").strip().upper()
    s = " ".join(s.split())

    # Handle spreadsheet-like artifacts such as "DKK 1.00" -> "DKK1".
    parts = s.split(" ")
    if (
        len(parts) == 2
        and parts[0].isalpha()
        and parts[1].replace(".", "").isdigit()
    ):
        numeric = parts[1]
        if "." in numeric:
            head, tail = numeric.split(".", 1)
            if tail and set(tail) == {"0"}:
                numeric = head
        s = f"{parts[0]}{numeric}"
    return s


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

    s = _canonicalize_symbol_text(s)
    hid = hgnc_client.get_current_hgnc_id(s)
    if not hid:
        return s

    if isinstance(hid, (list, tuple, set)):
        hid = sorted(hid)[0] if hid else None
        if not hid:
            return s

    name = hgnc_client.get_hgnc_name(hid) if hid else None
    return name or s
