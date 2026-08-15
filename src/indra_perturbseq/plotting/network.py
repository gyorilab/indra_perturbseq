"""Network visualizations for multi-hop pathway analysis.
Provides two subcommands:.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
from indra_perturbseq.pipelines.common import warn_deprecated_flags
from indra_perturbseq.runtime import add_log_level_arg, configure_logging

if TYPE_CHECKING:
    import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def _warn_deprecated_flags(argv: list[str] | None) -> None:
    warn_deprecated_flags(
        argv,
        {
            "--input-csv": "--input",
            "--indra-csv": "--indra",
            "--omnipath-csv": "--omnipath",
        },
        logger,
    )


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _overlay_edge_type(sources: set[str]) -> str:
    if sources == {"indra", "omnipath"}:
        return "both"
    if "indra" in sources:
        return "indra"
    return "omnipath"


def _edge_source_counts(edge_sources: dict[tuple[str, str], set[str]]) -> tuple[int, int, int]:
    indra_only = sum(1 for s in edge_sources.values() if s == {"indra"})
    omni_only = sum(1 for s in edge_sources.values() if s == {"omnipath"})
    shared = sum(1 for s in edge_sources.values() if s == {"indra", "omnipath"})
    return indra_only, omni_only, shared


def _node_buckets(
    G: nx.DiGraph,
    pos: dict[str, tuple[float, float]],
    intermediates: set[str],
    hover_builder,
    size_builder,
    label_builder,
) -> dict[str, dict[str, list]]:
    buckets = {
        "intermediate": {"x": [], "y": [], "text": [], "size": [], "labels": []},
        "source_target": {"x": [], "y": [], "text": [], "size": [], "labels": []},
    }
    for node in G.nodes():
        x, y = pos[node]
        ntype = "intermediate" if node in intermediates else "source_target"
        buckets[ntype]["x"].append(x)
        buckets[ntype]["y"].append(y)
        buckets[ntype]["text"].append(hover_builder(node, ntype))
        buckets[ntype]["size"].append(size_builder(node))
        buckets[ntype]["labels"].append(label_builder(node))
    return buckets


# Shared helpers
def _is_gene_identifier(identifier: object) -> bool:
    """Return ``True`` if *identifier* looks like a gene symbol or FamPlex ID."""
    if pd.isna(identifier):
        return False
    s = str(identifier)
    if s.startswith("fplx:"):
        return True
    return not any(s.startswith(p) for p in ("mesh:", "go:", "chebi:", "uniprot:"))


def _build_edge_trace(
    x0: float, y0: float, x1: float, y1: float,
    hover: str, width: float, color: str,
) -> go.Scatter:
    """Create a single Plotly edge line trace."""
    import plotly.graph_objects as go

    return go.Scatter(
        x=[x0, x1, None], y=[y0, y1, None],
        mode="lines",
        line=dict(width=width, color=color),
        hoverinfo="text", text=hover,
        showlegend=False,
    )


def _node_scatter(
    xs: list[float], ys: list[float],
    labels: list[str], hovers: list[str], sizes: list[float],
    name: str, marker_color: str, edge_color: str,
    font_size: int = 9, opacity: float = 0.9,
    edge_width: float = 2.0,
) -> go.Scatter:
    """Create a Plotly scatter trace for a class of nodes."""
    import plotly.graph_objects as go

    return go.Scatter(
        x=xs, y=ys,
        mode="markers+text", name=name,
        text=labels, textposition="top center",
        textfont=dict(size=font_size, color=edge_color),
        hovertext=hovers, hoverinfo="text",
        marker=dict(
            size=sizes, color=marker_color,
            line=dict(width=edge_width, color=edge_color),
            opacity=opacity,
        ),
    )


# Pathway subcommand
def load_top_pathways(path: str, n: int = 100) -> pd.DataFrame:
    """Load the top *n* pathways by p-value from a hop CSV.

    Parameters
    ----------
    path :
        CSV with at least ``pval`` and ``logfoldchange`` columns.
    n :
        Number of top pathways to select.
    """
    df = pd.read_csv(path, low_memory=False)
    df = df[np.isfinite(df["pval"]) & np.isfinite(df["logfoldchange"])].copy()
    top = df.nsmallest(n, "pval")
    logger.info(
        "Top %d pathways: p-value range %.2e - %.2e",
        len(top), top["pval"].min(), top["pval"].max(),
    )
    return top


def build_pathway_network(
    pathways: pd.DataFrame,
) -> tuple[nx.DiGraph, set[str], set[str]]:
    """Build a directed graph from multi-hop pathway rows.

    Returns
    -------
    G :
        NetworkX DiGraph.
    source_targets :
        Source and target gene nodes.
    intermediates :
        Intermediate-only nodes.
    """
    G = nx.DiGraph()
    source_targets: set[str] = set()
    intermediates: set[str] = set()

    for _, row in pathways.iterrows():
        src, tgt = row["source"], row["target"]
        int1 = row.get("intermediate_1")
        int2 = row.get("intermediate_2")

        source_targets.update([src, tgt])

        if pd.notna(int1):
            G.add_edge(src, int1)
            intermediates.add(int1)
            if pd.notna(int2):
                G.add_edge(int1, int2)
                G.add_edge(int2, tgt)
                intermediates.add(int2)
            else:
                G.add_edge(int1, tgt)
        else:
            G.add_edge(src, tgt)

    intermediates -= source_targets
    logger.info("Network: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G, source_targets, intermediates


def plot_pathway_interactive(
    G: nx.DiGraph,
    source_targets: set[str],
    intermediates: set[str],
    out_html: str,
    title: str = "Top 3-hop Pathways",
    spring_k: float = 1.0,
) -> None:
    """Write an interactive Plotly network visualization to *out_html*.

    Parameters
    ----------
    G :
        NetworkX DiGraph.
    source_targets :
        Source / target gene nodes.
    intermediates :
        Intermediate-only nodes.
    out_html :
        Output HTML path.
    title :
        Plot title.
    spring_k :
        Spring layout ``k`` parameter.
    """
    import plotly.graph_objects as go

    pos = nx.spring_layout(G, k=spring_k, iterations=100, seed=42)
    centrality = nx.degree_centrality(G)
    degrees = dict(G.degree())
    degree_thr = np.percentile(list(degrees.values()), 80)

    edge_traces = [
        _build_edge_trace(
            *pos[e[0]], *pos[e[1]],
            hover=f"{e[0]} -> {e[1]}",
            width=0.5, color="rgba(150,150,150,0.3)",
        )
        for e in G.edges()
    ]

    buckets = _node_buckets(
        G,
        pos,
        intermediates,
        hover_builder=lambda node, ntype: (
            f"<b>{node}</b><br>Type: {'Intermediate' if ntype == 'intermediate' else 'Source/Target'}<br>"
            f"Connections: {degrees[node]}<br>Centrality: {centrality[node]:.3f}"
        ),
        size_builder=lambda node: 10 + degrees[node] * 2,
        label_builder=lambda node: node if degrees[node] >= degree_thr else "",
    )

    int_trace = _node_scatter(
        **buckets["intermediate"],
        name="Intermediate", marker_color="lightgray",
        edge_color="gray", font_size=8, edge_width=1.5, opacity=0.8,
    )
    st_trace = _node_scatter(
        **buckets["source_target"],
        name="Source/Target", marker_color="lightblue",
        edge_color="darkblue", font_size=9, edge_width=2.0, opacity=0.9,
    )

    top_hubs = sorted(degrees.items(), key=lambda kv: kv[1], reverse=True)[:5]
    hub_text = "Top Hubs:<br>" + "<br>".join(f"{n}: {d}" for n, d in top_hubs)

    fig = go.Figure(
        data=edge_traces + [int_trace, st_trace],
        layout=go.Layout(
            title=dict(
                text=f"<b>{title}</b><br><sub>Hover for details</sub>",
                font=dict(size=18), x=0.5,
            ),
            showlegend=True, hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=80),
            plot_bgcolor="white",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[dict(
                text=hub_text, xref="paper", yref="paper",
                x=0.98, y=0.02, xanchor="right", yanchor="bottom",
                showarrow=False, bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black", borderwidth=1, font=dict(size=10),
            )],
        ),
    )

    _ensure_parent_dir(out_html)
    fig.write_html(out_html)
    logger.info("Interactive HTML saved: %s", out_html)


def plot_pathway_static(
    G: nx.DiGraph,
    source_targets: set[str],
    intermediates: set[str],
    out_png: str,
    title: str = "Top 3-hop Pathways Network",
    min_label_degree: int = 5,
) -> None:
    """Write a static matplotlib network visualization to *out_png*.

    Parameters
    ----------
    G :
        NetworkX DiGraph.
    source_targets :
        Source / target gene nodes.
    intermediates :
        Intermediate-only nodes.
    out_png :
        Output PNG path.
    title :
        Figure title.
    min_label_degree :
        Only label nodes with degree >= this value.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 16))
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax, arrows=True)

    int_nodes = [n for n in G.nodes() if n in intermediates]
    st_nodes = [n for n in G.nodes() if n in source_targets]

    nx.draw_networkx_nodes(
        G, pos, nodelist=int_nodes,
        node_color="gray", node_size=100, alpha=0.6, ax=ax,
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=st_nodes,
        node_color="lightblue", node_size=200,
        edgecolors="darkblue", linewidths=2, ax=ax,
    )

    high_deg = {n: n for n in G.nodes() if G.degree(n) >= min_label_degree}
    nx.draw_networkx_labels(G, pos, labels=high_deg, font_size=8, ax=ax)

    ax.set_title(title, fontsize=14)
    ax.axis("off")
    fig.tight_layout()

    _ensure_parent_dir(out_png)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Static PNG saved: %s", out_png)


def _run_pathway(args: argparse.Namespace) -> None:
    """Execute the ``pathway`` subcommand."""
    top = load_top_pathways(args.input, n=args.top_n)
    G, st, ints = build_pathway_network(top)

    if args.html:
        plot_pathway_interactive(
            G, st, ints, args.html,
            title=args.title, spring_k=args.spring_k,
        )
    if args.png:
        plot_pathway_static(
            G, st, ints, args.png,
            title=args.title, min_label_degree=args.min_label_degree,
        )


# ---------------------------------------------------------------------------
# Overlay subcommand
# ---------------------------------------------------------------------------

def load_matched_pathways(
    indra_path: str,
    omnipath_path: str,
    n_pairs: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load top *n_pairs* from INDRA and find matching OmniPath pathways.

    Parameters
    ----------
    indra_path :
        INDRA 3-hop results CSV.
    omnipath_path :
        OmniPath 3-hop results CSV.
    n_pairs :
        Number of top INDRA gene pairs to select.

    Returns
    -------
    indra_top :
        Top INDRA rows with an added ``pair`` column.
    omnipath_matched :
        OmniPath rows matching the selected pairs.
    """
    indra = pd.read_csv(indra_path, low_memory=False)
    indra = indra[np.isfinite(indra["pval"]) & np.isfinite(indra["logfoldchange"])].copy()
    indra_top = indra.nsmallest(n_pairs, "pval").copy()
    indra_top["pair"] = indra_top["source"] + "->" + indra_top["target"]

    omni = pd.read_csv(omnipath_path, low_memory=False)
    omni["pair"] = omni["source"] + "->" + omni["target"]

    target_pairs = set(indra_top["pair"])
    omni_matched = omni[omni["pair"].isin(target_pairs)].copy()

    logger.info(
        "INDRA pairs: %d, OmniPath matches: %d, unmatched: %d",
        len(indra_top), len(omni_matched),
        len(target_pairs) - len(set(omni_matched["pair"])),
    )
    return indra_top, omni_matched


def _add_pathway_edges(
    G: nx.DiGraph,
    df: pd.DataFrame,
    db_label: str,
    source_targets: set[str],
    intermediates: set[str],
    edge_sources: dict[tuple[str, str], set[str]],
    edge_info: dict[tuple[str, str], dict],
) -> None:
    """Add edges from one database's DataFrame into the shared graph."""
    for _, row in df.iterrows():
        src, tgt = row["source"], row["target"]
        pair = row["pair"]
        int1 = row.get("intermediate_1")
        int2 = row.get("intermediate_2")

        if not _is_gene_identifier(int1):
            continue
        if pd.notna(int2) and not _is_gene_identifier(int2):
            continue

        source_targets.update([src, tgt])

        info = {"database": db_label, "pair": pair}
        if "pval" in row.index and pd.notna(row.get("pval")):
            info["pval"] = row["pval"]
        if "logfoldchange" in row.index and pd.notna(row.get("logfoldchange")):
            info["logfc"] = row["logfoldchange"]

        edges_to_add: list[tuple[str, str]] = []
        if pd.notna(int1):
            intermediates.add(str(int1))
            edges_to_add.append((src, str(int1)))
            if pd.notna(int2):
                intermediates.add(str(int2))
                edges_to_add.append((str(int1), str(int2)))
                edges_to_add.append((str(int2), tgt))
            else:
                edges_to_add.append((str(int1), tgt))

        for edge in edges_to_add:
            G.add_edge(*edge)
            edge_sources.setdefault(edge, set()).add(db_label)
            edge_info[edge] = info


def build_overlay_network(
    indra_df: pd.DataFrame,
    omni_df: pd.DataFrame,
) -> tuple[
    nx.DiGraph, set[str], set[str],
    dict[tuple[str, str], set[str]],
    dict[tuple[str, str], dict],
]:
    """Build a combined network from INDRA and OmniPath pathway data.

    Returns
    -------
    G :
        Combined DiGraph.
    source_targets :
        Source/target gene nodes.
    intermediates :
        Intermediate-only nodes.
    edge_sources :
        ``{(u, v): {"indra"} | {"omnipath"} | {"indra", "omnipath"}}``.
    edge_info :
        Metadata dict per edge.
    """
    G = nx.DiGraph()
    source_targets: set[str] = set()
    intermediates: set[str] = set()
    edge_sources: dict[tuple[str, str], set[str]] = {}
    edge_info: dict[tuple[str, str], dict] = {}

    _add_pathway_edges(G, indra_df, "indra", source_targets, intermediates, edge_sources, edge_info)
    _add_pathway_edges(G, omni_df, "omnipath", source_targets, intermediates, edge_sources, edge_info)

    intermediates -= source_targets

    indra_only = sum(1 for s in edge_sources.values() if s == {"indra"})
    omni_only = sum(1 for s in edge_sources.values() if s == {"omnipath"})
    shared = sum(1 for s in edge_sources.values() if s == {"indra", "omnipath"})
    logger.info(
        "Overlay network: %d nodes, %d edges (INDRA-only=%d, OmniPath-only=%d, shared=%d)",
        G.number_of_nodes(), G.number_of_edges(), indra_only, omni_only, shared,
    )
    return G, source_targets, intermediates, edge_sources, edge_info


_OVERLAY_EDGE_COLORS = {
    "indra": "rgba(65,105,225,0.6)",
    "omnipath": "rgba(255,140,0,0.6)",
    "both": "rgba(148,0,211,0.9)",
}
_OVERLAY_EDGE_WIDTHS = {"indra": 2, "omnipath": 2, "both": 3.5}


def plot_overlay_interactive(
    G: nx.DiGraph,
    source_targets: set[str],
    intermediates: set[str],
    edge_sources: dict[tuple[str, str], set[str]],
    edge_info: dict[tuple[str, str], dict],
    out_html: str,
    spring_k: float = 2.0,
) -> None:
    """Write an interactive overlay visualization to *out_html*.

    Parameters
    ----------
    G :
        Combined DiGraph.
    source_targets :
        Source/target nodes.
    intermediates :
        Intermediate-only nodes.
    edge_sources :
        Database origin set per edge.
    edge_info :
        Metadata dict per edge.
    out_html :
        Output HTML path.
    spring_k :
        Spring layout ``k`` parameter.
    """
    import plotly.graph_objects as go

    pos = nx.spring_layout(G, k=spring_k, iterations=100, seed=42)

    edge_traces: list[go.Scatter] = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        etype = _overlay_edge_type(edge_sources.get(edge, set()))

        hover = f"<b>{edge[0]} -> {edge[1]}</b><br>Database: {etype.upper()}"
        info = edge_info.get(edge, {})
        if "pair" in info:
            hover += f"<br>Gene pair: {info['pair']}"
        if "pval" in info:
            hover += f"<br>p-value: {info['pval']:.2e}"
        if "logfc" in info:
            hover += f"<br>log2FC: {info['logfc']:.2f}"

        edge_traces.append(_build_edge_trace(
            x0, y0, x1, y1, hover,
            width=_OVERLAY_EDGE_WIDTHS[etype],
            color=_OVERLAY_EDGE_COLORS[etype],
        ))

    legend_traces = [
        go.Scatter(
            x=[None], y=[None], mode="lines",
            name=f"INDRA only", line=dict(width=2, color="royalblue"),
        ),
        go.Scatter(
            x=[None], y=[None], mode="lines",
            name=f"OmniPath only", line=dict(width=2, color="darkorange"),
        ),
        go.Scatter(
            x=[None], y=[None], mode="lines",
            name="Agreed pathway", line=dict(width=3.5, color="darkviolet"),
        ),
    ]

    degrees = dict(G.degree())
    buckets = _node_buckets(
        G,
        pos,
        intermediates,
        hover_builder=lambda node, ntype: (
            f"<b>{node}</b><br>Type: {'Intermediate' if ntype == 'intermediate' else 'Source/Target'}"
            f"<br>Connections: {degrees[node]}"
        ),
        size_builder=lambda node: 15 + degrees[node] * 3,
        label_builder=lambda node: node,
    )

    int_trace = _node_scatter(
        **buckets["intermediate"],
        name="Intermediate Proteins", marker_color="lightgray",
        edge_color="gray", font_size=10, edge_width=2.0, opacity=0.9,
    )
    st_trace = _node_scatter(
        **buckets["source_target"],
        name="Source/Target Genes", marker_color="lightblue",
        edge_color="darkblue", font_size=11, edge_width=2.5, opacity=0.95,
    )

    indra_only, omni_only, shared = _edge_source_counts(edge_sources)
    total = indra_only + omni_only + shared
    overlap_pct = 100.0 * shared / total if total > 0 else 0.0

    stats_text = (
        f"<b>Same Gene Pairs Comparison</b><br>"
        f"INDRA only edges: {indra_only}<br>"
        f"OmniPath only edges: {omni_only}<br>"
        f"<b>Agreed edges: {shared}</b><br>"
        f"Edge agreement: {overlap_pct:.1f}%<br>"
        f"Total edges: {G.number_of_edges()}"
    )

    fig = go.Figure(
        data=legend_traces + edge_traces + [int_trace, st_trace],
        layout=go.Layout(
            title=dict(
                text=(
                    "<b>INDRA vs OmniPath - Same Gene Pairs Pathway Comparison</b>"
                    "<br><sub>Blue=INDRA only | Orange=OmniPath only | Purple=Both agree</sub>"
                ),
                font=dict(size=20), x=0.5, xanchor="center",
            ),
            showlegend=True, hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=100),
            plot_bgcolor="white",
            width=1400, height=1000,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.95)",
                bordercolor="black", borderwidth=2, font=dict(size=12),
            ),
            annotations=[dict(
                text=stats_text, xref="paper", yref="paper",
                x=0.98, y=0.02, xanchor="right", yanchor="bottom",
                showarrow=False, bgcolor="rgba(255,255,255,0.95)",
                bordercolor="black", borderwidth=2, font=dict(size=12),
            )],
        ),
    )

    _ensure_parent_dir(out_html)
    fig.write_html(out_html)
    logger.info("Overlay HTML saved: %s", out_html)


def plot_overlay_static(
    G: nx.DiGraph,
    source_targets: set[str],
    intermediates: set[str],
    edge_sources: dict[tuple[str, str], set[str]],
    out_png: str,
    spring_k: float = 2.0,
) -> None:
    """Write a static overlay PNG comparing INDRA and OmniPath edges.

    Parameters
    ----------
    G :
        Combined DiGraph.
    source_targets :
        Source/target gene nodes.
    intermediates :
        Intermediate-only nodes.
    edge_sources :
        Database origin set per edge.
    out_png :
        Output PNG path.
    spring_k :
        Spring layout ``k`` parameter.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(20, 16))
    pos = nx.spring_layout(G, k=spring_k, iterations=100, seed=42)

    edge_groups = {
        "indra": ([], "royalblue", 2, 0.5, 15),
        "omnipath": ([], "darkorange", 2, 0.5, 15),
        "both": ([], "darkviolet", 3.5, 0.8, 20),
    }
    for e in G.edges():
        edge_groups[_overlay_edge_type(edge_sources.get(e, set()))][0].append(e)

    for edges, color, width, alpha, arrow in edge_groups.values():
        if edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=edges, edge_color=color,
                alpha=alpha, width=width, ax=ax, arrows=True,
                arrowsize=arrow, arrowstyle="->",
                connectionstyle="arc3,rad=0.1",
            )

    int_nodes = [n for n in G.nodes() if n in intermediates]
    st_nodes = [n for n in G.nodes() if n in source_targets]

    nx.draw_networkx_nodes(
        G, pos, nodelist=int_nodes, node_color="lightgray",
        node_size=400, alpha=0.9, edgecolors="gray", linewidths=2, ax=ax,
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=st_nodes, node_color="lightblue",
        node_size=500, edgecolors="darkblue", linewidths=3, ax=ax,
    )
    nx.draw_networkx_labels(
        G, pos, font_size=11, font_weight="bold", font_family="sans-serif", ax=ax,
    )

    indra_only, omni_only, shared = _edge_source_counts(edge_sources)
    stats = f"INDRA only: {indra_only}\nOmniPath only: {omni_only}\nAgreed: {shared}"
    ax.text(
        0.98, 0.02, stats, transform=ax.transAxes, fontsize=12,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9,
                  edgecolor="black", linewidth=2),
    )

    legend_elems = [
        Line2D([0], [0], color="royalblue", lw=3, label="INDRA only"),
        Line2D([0], [0], color="darkorange", lw=3, label="OmniPath only"),
        Line2D([0], [0], color="darkviolet", lw=4, label="Both databases agree"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=13, frameon=True,
              fancybox=True, shadow=True)
    ax.set_title(
        "INDRA vs OmniPath Pathway Comparison", fontsize=18, fontweight="bold", pad=20,
    )
    ax.axis("off")
    fig.tight_layout()

    _ensure_parent_dir(out_png)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Overlay PNG saved: %s", out_png)


def _run_overlay(args: argparse.Namespace) -> None:
    """Execute the ``overlay`` subcommand."""
    indra_top, omni_matched = load_matched_pathways(
        args.indra, args.omnipath, n_pairs=args.n_pairs,
    )
    if omni_matched.empty:
        logger.warning("No matching pathways found in OmniPath; nothing to plot.")
        return

    G, st, ints, edge_sources, edge_info = build_overlay_network(indra_top, omni_matched)
    if G.number_of_nodes() == 0:
        logger.warning("No valid pathways after filtering; nothing to plot.")
        return

    if args.html:
        plot_overlay_interactive(
            G, st, ints, edge_sources, edge_info,
            args.html, spring_k=args.spring_k,
        )
    if args.png:
        plot_overlay_static(
            G, st, ints, edge_sources, args.png,
            spring_k=args.spring_k,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI entry point for network visualizations."""
    _warn_deprecated_flags(argv)
    ap = argparse.ArgumentParser(
        description="Network visualizations for multi-hop pathway analysis.",
    )
    add_log_level_arg(ap, default="INFO")
    sub = ap.add_subparsers(dest="command", required=True)

    # -- pathway -----------------------------------------------------------
    pw = sub.add_parser(
        "pathway", help="Visualize top-N multi-hop pathways as a network.",
    )
    pw.add_argument(
        "--input",
        "--input-csv",
        required=True,
        help="Hop CSV (e.g. 3-hop results)",
    )
    pw.add_argument("--top-n", type=int, default=100,
                    help="Number of top pathways to plot (default: 100)")
    pw.add_argument("--html", help="Output interactive HTML path")
    pw.add_argument("--png", help="Output static PNG path")
    pw.add_argument("--title", default="Top 3-hop Pathways")
    pw.add_argument("--spring-k", type=float, default=1.0,
                    help="Spring layout k parameter (default: 1.0)")
    pw.add_argument("--min-label-degree", type=int, default=5,
                    help="Min degree for node labels in static plot (default: 5)")
    pw.set_defaults(func=_run_pathway)

    # -- overlay -----------------------------------------------------------
    ov = sub.add_parser(
        "overlay", help="Compare INDRA vs OmniPath pathways for same gene pairs.",
    )
    ov.add_argument(
        "--indra",
        "--indra-csv",
        required=True,
        help="INDRA 3-hop results CSV",
    )
    ov.add_argument(
        "--omnipath",
        "--omnipath-csv",
        required=True,
        help="OmniPath 3-hop results CSV",
    )
    ov.add_argument("--n-pairs", type=int, default=20,
                    help="Number of top INDRA pairs to compare (default: 20)")
    ov.add_argument("--html", help="Output interactive HTML path")
    ov.add_argument("--png", help="Output static PNG path")
    ov.add_argument("--spring-k", type=float, default=2.0,
                    help="Spring layout k parameter (default: 2.0)")
    ov.set_defaults(func=_run_overlay)

    args = ap.parse_args(argv)
    configure_logging(args.log_level)
    args.func(args)


if __name__ == "__main__":
    main()
