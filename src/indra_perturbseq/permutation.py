"""Label-permuted network view for null-model evaluation."""

from __future__ import annotations

import numpy as np


class PermutationView:
    """Bijective label shuffle on HGNC nodes (topology unchanged).

    Parameters
    ----------
    graph :
        The original INDRA NetworkX DiGraph.
    hgnc_nodes :
        List of HGNC node identifiers in the graph.
    seed :
        Random seed for reproducibility.
    """

    def __init__(self, graph, hgnc_nodes: list[str], seed: int):
        rng = np.random.default_rng(seed)
        perm = np.array(hgnc_nodes, dtype=object)
        rng.shuffle(perm)
        perm_list = perm.tolist()
        self.phi: dict[str, str] = dict(zip(hgnc_nodes, perm_list))
        self.phi_inv: dict[str, str] = dict(zip(perm_list, hgnc_nodes))
        self.graph = graph

    def orig_for_label(self, label: str) -> str | None:
        """Map a permuted label back to the original node."""
        return self.phi_inv.get(label)

    def label_for_orig(self, orig_node: str) -> str | None:
        """Map an original node to its permuted label."""
        return self.phi.get(orig_node)
