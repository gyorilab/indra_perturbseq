"""1-hop pathway extraction from an INDRA network export graph.
This module provides pipeline execution and command-line workflow orchestration.
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from indra_perturbseq.deg import load_deg_targets
from indra_perturbseq.evidence import (
    rich_stmt_text_from_hash,
)
from indra_perturbseq.graph import is_hgnc_node, load_graph
from indra_perturbseq.hgnc import normalize_hgnc_symbol
from indra_perturbseq.pipelines.common import (
    load_sources_from_args,
    warn_deprecated_flags,
    write_split_outputs,
)
from indra_perturbseq.pipelines.enrichment import (
    add_enrichment_cli_args,
    annotate_mesh_terms,
    enrich_evidence,
    validate_enrichment_args,
)
from indra_perturbseq.statements import indra_html_url, iter_incdec_statements

logger = logging.getLogger(__name__)


def run_1hop(
    graph,
    genes: list[str],
    deg_dir: str,
    p_threshold: float,
    prefer_fdr: bool,
) -> pd.DataFrame:
    """Extract 1-hop paths for all *genes* on *graph*.

    For each source gene, finds direct edges to significant DEG targets
    that carry an IncreaseAmount or DecreaseAmount statement.
    """
    all_rows: list[dict] = []
    t0 = time.time()

    for i, raw_gene in enumerate(genes, start=1):
        gene = normalize_hgnc_symbol(raw_gene)
        if not gene or gene not in graph or not is_hgnc_node(graph, gene):
            logger.debug("SKIP %s: not in graph as HGNC", raw_gene)
            continue

        targets, deg_map, err = load_deg_targets(
            deg_dir, raw_gene, p_threshold, prefer_fdr,
        )
        if err:
            logger.debug("[%d/%d] %s: %s", i, len(genes), raw_gene, err)
            continue

        found = 0
        for tgt in targets:
            if tgt not in graph or not is_hgnc_node(graph, tgt):
                continue
            if not graph.has_edge(gene, tgt):
                continue
            for s in iter_incdec_statements(graph, gene, tgt):
                found += 1
                all_rows.append({
                    "source": gene,
                    "target": tgt,
                    "stmt_type": s.get("stmt_type"),
                    "english_stmt": "",
                    "belief": s.get("belief"),
                    "evidence_count": s.get("evidence_count"),
                    "logfoldchange": deg_map.get(tgt, {}).get("logfoldchange"),
                    "pval": deg_map.get(tgt, {}).get("pval"),
                    "stmt_hash": s.get("stmt_hash"),
                    "hop1_indra_url": indra_html_url(s.get("stmt_hash")),
                    "source_counts": s.get("source_counts"),
                })

        elapsed = time.time() - t0
        logger.info(
            "[%d/%d] %s -> %s: rows=%d  elapsed=%.1fm",
            i, len(genes), raw_gene, gene, found, elapsed / 60,
        )

    df = pd.DataFrame(all_rows)
    return df.reindex(columns=[
        "source", "target", "stmt_type", "english_stmt",
        "belief", "evidence_count", "logfoldchange", "pval",
        "stmt_hash", "hop1_indra_url", "source_counts",
    ])


def _fill_english_statements(df: pd.DataFrame) -> pd.DataFrame:
    """Populate ``english_stmt`` by fetching from db.indra.bio."""
    cache: dict = {}

    def _fill(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        s = str(x).strip()
        if not s:
            return ""
        try:
            return rich_stmt_text_from_hash(int(s), cache)
        except Exception:
            return ""

    df = df.copy()
    df["english_stmt"] = df["stmt_hash"].apply(_fill)
    return df


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "source", "target", "stmt_type", "english_stmt",
        "belief", "evidence_count", "logfoldchange", "pval",
        "evidence_text_hop1", "pmids_hop1", "Annotated MeSH terms hop1",
        "stmt_hash", "hop1_indra_url", "source_counts",
    ]
    keep = [c for c in cols if c in df.columns]
    extra = [c for c in df.columns if c not in keep]
    return df[keep + extra]


def main(argv: list[str] | None = None) -> None:
    warn_deprecated_flags(
        argv,
        {
            "--perturbations-csv": "--source-genes-csv",
            "--out-csv-main": "--output-main",
            "--out-csv-self": "--output-self-targets",
            "--out-csv-self-targets": "--output-self-targets",
        },
        logger,
    )
    ap = argparse.ArgumentParser(
        description="1-hop pathway extraction from INDRA network export.",
    )
    ap.add_argument("--graph-pkl", required=True,
                    help="Path to INDRA network export .pkl")
    ap.add_argument("--source-genes-csv", "--perturbations-csv", required=True,
                    help="target_validation_expanded.csv")
    ap.add_argument("--deg-dir", required=True,
                    help="Folder with <GENE>_vs_control.csv files")
    ap.add_argument("--output-main", "--out-csv-main", required=True,
                    help="Output CSV for non-self paths")
    ap.add_argument(
        "--output-self-targets",
        "--out-csv-self",
        "--out-csv-self-targets",
        required=True,
        help="Output CSV for self paths",
    )
    ap.add_argument("--filter-column", default="analysis_flag")
    ap.add_argument("--filter-value", default="Use_for_analysis")
    ap.add_argument("--gene-column", default="Gene")
    ap.add_argument("--p-threshold", type=float, default=0.05)
    ap.add_argument("--prefer-fdr", action="store_true")
    add_enrichment_cli_args(ap)
    args = ap.parse_args(argv)
    validate_enrichment_args(args, ap)

    graph, _ = load_graph(args.graph_pkl)

    genes = load_sources_from_args(args)

    df = run_1hop(graph, genes, args.deg_dir, args.p_threshold, args.prefer_fdr)
    df = _fill_english_statements(df)
    df = enrich_evidence(df, args, hop_hash_columns={1: "stmt_hash"})
    df = annotate_mesh_terms(df, args, pmid_columns=["pmids_hop1"])
    df = _reorder_columns(df)
    write_split_outputs(df, args.output_main, args.output_self_targets, logger)


if __name__ == "__main__":
    main()
