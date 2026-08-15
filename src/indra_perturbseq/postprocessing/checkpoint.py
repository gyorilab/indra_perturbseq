"""Extract and inspect a 3-hop analysis checkpoint pickle file.

Reads a checkpoint saved during long-running 3-hop analysis, logs
summary statistics, and optionally exports the intermediate results
to CSV.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_checkpoint(path: str | Path) -> dict:
    """Load a checkpoint pickle and return its contents.

    Parameters
    ----------
    path : str or Path
        Path to the ``.pkl`` checkpoint file.

    Returns
    -------
    dict
        Dictionary with keys ``results``, ``processed_genes``,
        ``timestamp``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    with p.open("rb") as fh:
        return pickle.load(fh)  # noqa: S301


def summarise_checkpoint(
    data: dict,
    total_genes: int = 358,
) -> pd.DataFrame | None:
    """Log summary statistics and return results as a dataframe.

    Parameters
    ----------
    data : dict
        Checkpoint dictionary (see :func:`load_checkpoint`).
    total_genes : int
        Total number of genes in the analysis (for progress reporting).

    Returns
    -------
    pd.DataFrame or None
        Results dataframe, or ``None`` when results are empty.
    """
    results = data["results"]
    processed = data["processed_genes"]
    ts = data["timestamp"]

    logger.info("Checkpoint saved: %s", datetime.fromtimestamp(ts))
    logger.info("Genes completed: %d / %d (%.1f%%)",
                len(processed), total_genes,
                len(processed) / total_genes * 100)
    logger.info("Total 3-hop pathways: %d", len(results))

    if not results:
        logger.info("No results in checkpoint")
        return None

    df = pd.DataFrame(results)
    logger.info("Unique sources: %d, targets: %d",
                df["source"].nunique(), df["target"].nunique())
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Inspect a 3-hop checkpoint pickle and optionally "
                    "export results to CSV.",
    )
    ap.add_argument("--pickle-file", required=True,
                    help="Path to the checkpoint .pkl file.")
    ap.add_argument("--output", default=None,
                    help="Optional CSV path for extracted results.")
    ap.add_argument("--total-genes", type=int, default=358,
                    help="Total gene count for progress calculation.")
    args = ap.parse_args(argv)

    data = load_checkpoint(args.pickle_file)
    df = summarise_checkpoint(data, total_genes=args.total_genes)

    if df is not None and args.output:
        df.to_csv(args.output, index=False)
        logger.info("Results exported -> %s (%d rows)", args.output, len(df))


if __name__ == "__main__":
    main()
