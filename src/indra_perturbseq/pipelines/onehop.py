"""1-hop pathway extraction from an INDRA network export graph."""

from __future__ import annotations

import argparse
import logging
import os
import time

import pandas as pd

from indra_perturbseq.deg import load_deg_targets, pick_sig_column
from indra_perturbseq.evidence import rich_stmt_text_from_hash
from indra_perturbseq.gene_lists import load_source_genes
from indra_perturbseq.graph import is_hgnc_node, load_graph
from indra_perturbseq.hgnc import normalize_hgnc_symbol
from indra_perturbseq.statements import iter_incdec_statements

logger = logging.getLogger(__name__)


def run_1hop(
    graph,
    genes: list[str],
    de_dir: str,
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
            de_dir, raw_gene, p_threshold, prefer_fdr,
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
        "stmt_hash", "source_counts",
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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="1-hop pathway extraction from INDRA network export.",
    )
    ap.add_argument("--graph-pkl", required=True,
                    help="Path to INDRA network export .pkl")
    ap.add_argument("--perturbations-csv", required=True,
                    help="target_validation_expanded.csv")
    ap.add_argument("--de-dir", required=True,
                    help="Folder with <GENE>_vs_control.csv files")
    ap.add_argument("--out-csv-main", required=True,
                    help="Output CSV for non-self paths")
    ap.add_argument("--out-csv-self", required=True,
                    help="Output CSV for self paths")
    ap.add_argument("--karen-flag-col", default="Karen_Flag")
    ap.add_argument("--karen-flag-value", default="Use_for_analysis")
    ap.add_argument("--gene-col", default="Gene")
    ap.add_argument("--p-threshold", type=float, default=0.05)
    ap.add_argument("--prefer-fdr", action="store_true")
    args = ap.parse_args(argv)

    graph, _ = load_graph(args.graph_pkl)

    genes = load_source_genes(
        args.perturbations_csv,
        gene_col=args.gene_col,
        flag_col=args.karen_flag_col,
        flag_value=args.karen_flag_value,
    )

    df = run_1hop(graph, genes, args.de_dir, args.p_threshold, args.prefer_fdr)
    df = _fill_english_statements(df)

    main_df = df[df["source"] != df["target"]].copy()
    self_df = df[df["source"] == df["target"]].copy()

    os.makedirs(os.path.dirname(args.out_csv_main) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv_self) or ".", exist_ok=True)
    main_df.to_csv(args.out_csv_main, index=False)
    self_df.to_csv(args.out_csv_self, index=False)

    logger.info("Non-self rows: %d -> %s", len(main_df), args.out_csv_main)
    logger.info("Self rows:     %d -> %s", len(self_df), args.out_csv_self)


if __name__ == "__main__":
    main()
