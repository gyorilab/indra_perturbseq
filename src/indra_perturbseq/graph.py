"""INDRA network-export graph loading and node helpers."""

import logging
import pickle
import time

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def _install_numpy_dtype_shims() -> None:
    """Register missing numpy dtype aliases needed by older pickled graphs."""
    if "f16" not in np.sctypeDict:
        replacement = (
            np.longdouble
            if np.dtype(np.longdouble).itemsize == 16
            else np.float64
        )
        np.sctypeDict["f16"] = replacement
        if hasattr(np, "typeDict"):
            np.typeDict["f16"] = replacement


def load_graph(pkl_path: str) -> tuple[nx.DiGraph, float]:
    """Load a pickled INDRA directed graph.

    Parameters
    ----------
    pkl_path :
        Path to the ``.pkl`` file produced by the INDRA network export.

    Returns
    -------
    :
        A ``(graph, elapsed_seconds)`` tuple.
    """
    _install_numpy_dtype_shims()
    t0 = time.time()
    with open(pkl_path, "rb") as fh:
        graph = pickle.load(fh)
    elapsed = time.time() - t0
    logger.info(
        "Loaded graph in %.1f min | nodes=%s edges=%s",
        elapsed / 60,
        f"{graph.number_of_nodes():,}",
        f"{graph.number_of_edges():,}",
    )
    return graph, elapsed


def is_hgnc_node(graph: nx.DiGraph, node: str) -> bool:
    """Return True if *node* has ``ns == 'HGNC'`` in the graph."""
    return (graph.nodes.get(node, {}) or {}).get("ns") == "HGNC"
