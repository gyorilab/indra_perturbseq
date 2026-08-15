"""4-hop pathway extraction via INDRA Neo4j (HGNC-only intermediates).
Queries Neo4j for 4-hop paths: source -> mid1 -> mid2 -> mid3 -> target,.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from indra_perturbseq.deg import load_deg_targets
from indra_perturbseq.evidence import enrich_evidence_4hop
from indra_perturbseq.pipelines.common import (
    load_sources_from_args,
    warn_deprecated_flags,
    write_split_outputs,
)
from indra_perturbseq.services.neo4j import get_neo4j_client, safe_query_tx
from indra_perturbseq.hgnc import normalize_hgnc_symbol
from indra_perturbseq.statements import indra_html_url

logger = logging.getLogger(__name__)

_4HOP_QUERY = """
MATCH (a:BioEntity {id: $source})-[r1:indra_rel]->(m1:BioEntity)
      -[r2:indra_rel]->(m2:BioEntity)
      -[r3:indra_rel]->(m3:BioEntity)
      -[r4:indra_rel]->(b:BioEntity {id: $target})
WHERE r4.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
  AND m1.id STARTS WITH 'hgnc:'
  AND m2.id STARTS WITH 'hgnc:'
  AND m3.id STARTS WITH 'hgnc:'
RETURN a.id, m1.id, m2.id, m3.id, b.id,
       r1.stmt_type, r2.stmt_type, r3.stmt_type, r4.stmt_type,
       r1.belief, r2.belief, r3.belief, r4.belief,
       r1.evidence_count, r2.evidence_count, r3.evidence_count, r4.evidence_count,
       r1.stmt_hash, r2.stmt_hash, r3.stmt_hash, r4.stmt_hash
LIMIT 1
"""


def _hgnc_id_for_symbol(symbol: str) -> str | None:
    """Return ``hgnc:<id>`` for *symbol*, or ``None``."""
    from indra.databases.hgnc_client import get_current_hgnc_id

    hid = get_current_hgnc_id(symbol.upper())
    if isinstance(hid, (list, tuple)):
        hid = sorted(hid)[0] if hid else None
    return f"hgnc:{hid}" if hid else None


def _id_to_symbol(node_id: str) -> str:
    """Convert an ``hgnc:<id>`` Neo4j identifier to an HGNC symbol."""
    if not isinstance(node_id, str):
        return str(node_id)
    if node_id.startswith("hgnc:"):
        from indra.databases.hgnc_client import get_hgnc_name

        name = get_hgnc_name(node_id.removeprefix("hgnc:"))
        return name if name else node_id
    return node_id


def run_4hop_for_gene(
    gene: str,
    targets: list[str],
    deg_map: dict[str, dict],
    neo4j_timeout: int = 900,
) -> tuple[list[dict], str]:
    """Query Neo4j for 4-hop paths from *gene* to each target.

    Parameters
    ----------
    gene :
        Source gene symbol.
    targets :
        Target gene symbols to query against.
    deg_map :
        ``{target: {"logfoldchange": ..., "pval": ...}}``.
    neo4j_timeout :
        Per-query timeout in seconds.

    Returns
    -------
    :
        ``(rows, message)`` where *rows* is a list of result dicts.
    """
    source_id = _hgnc_id_for_symbol(gene)
    if not source_id:
        return [], f"SKIP {gene}: no HGNC ID"

    target_ids: dict[str, str] = {}
    for t in targets:
        tid = _hgnc_id_for_symbol(t)
        if tid:
            target_ids[t] = tid

    if not target_ids:
        return [], f"SKIP {gene}: no valid target HGNC IDs"

    client = get_neo4j_client()
    rows: list[dict] = []

    for tgt_symbol, tgt_id in target_ids.items():
        results = safe_query_tx(
            client,
            _4HOP_QUERY,
            source=source_id,
            target=tgt_id,
            timeout=neo4j_timeout,
        )

        for rec in results:
            if len(rec) < 21:
                continue
            (_, m1, m2, m3, _,
             s1, s2, s3, s4,
             b1, b2, b3, b4,
             e1, e2, e3, e4,
             h1, h2, h3, h4) = rec[:21]

            stats = deg_map.get(tgt_symbol, {})
            rows.append({
                "source": gene,
                "intermediate_1": _id_to_symbol(m1),
                "intermediate_2": _id_to_symbol(m2),
                "intermediate_3": _id_to_symbol(m3),
                "target": tgt_symbol,
                "stmt_type_1": s1, "stmt_type_2": s2,
                "stmt_type_3": s3, "stmt_type_4": s4,
                "belief_1": b1, "belief_2": b2,
                "belief_3": b3, "belief_4": b4,
                "evidence_1": e1, "evidence_2": e2,
                "evidence_3": e3, "evidence_4": e4,
                "hop1_hash": h1, "hop2_hash": h2,
                "hop3_hash": h3, "hop4_hash": h4,
                "hop1_indra_url": indra_html_url(h1),
                "hop2_indra_url": indra_html_url(h2),
                "hop3_indra_url": indra_html_url(h3),
                "hop4_indra_url": indra_html_url(h4),
                "logfoldchange": stats.get("logfoldchange"),
                "pval": stats.get("pval"),
            })
            break  # LIMIT 1 per query, but break after first record

    return rows, f"{gene}: produced {len(rows)} rows"


_COLUMN_ORDER = [
    "source", "intermediate_1", "intermediate_2", "intermediate_3", "target",
    "stmt_type_1", "evidence_text_hop1", "pmids_hop1",
    "stmt_type_2", "evidence_text_hop2", "pmids_hop2",
    "stmt_type_3", "evidence_text_hop3", "pmids_hop3",
    "stmt_type_4", "evidence_text_hop4", "pmids_hop4",
    "logfoldchange", "pval",
    "belief_1", "belief_2", "belief_3", "belief_4",
    "evidence_1", "evidence_2", "evidence_3", "evidence_4",
    "hop1_hash", "hop2_hash", "hop3_hash", "hop4_hash",
    "hop1_indra_url", "hop2_indra_url", "hop3_indra_url", "hop4_indra_url",
]


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in _COLUMN_ORDER if c in df.columns]
    extra = [c for c in df.columns if c not in keep]
    return df[keep + extra]


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for 4-hop pipeline."""
    warn_deprecated_flags(
        argv,
        {
            "--genes-csv": "--source-genes-csv",
            "--out-csv-main": "--output-main",
            "--out-csv-self": "--output-self-targets",
            "--out-csv-self-targets": "--output-self-targets",
        },
        logger,
    )
    ap = argparse.ArgumentParser(
        description="4-hop pipeline via INDRA Neo4j (HGNC-only intermediates).",
    )
    ap.add_argument("--source-genes-csv", "--genes-csv", required=True,
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
    ap.add_argument("--genes", nargs="+",
                    help="Explicit source genes (overrides CSV)")
    ap.add_argument("--limit-genes", type=int, default=0)
    ap.add_argument("--limit-targets", type=int, default=0)
    ap.add_argument("--path-workers", type=int, default=4)
    ap.add_argument("--neo4j-timeout", type=int, default=900)
    ap.add_argument("--neo4j-evidence-batch-size", type=int, default=2000)
    args = ap.parse_args(argv)

    genes = load_sources_from_args(args)

    all_rows: list[dict] = []
    logger.info("Running 4-hop extraction for %d genes...", len(genes))

    def _job(gene: str) -> tuple[list[dict], str]:
        norm = normalize_hgnc_symbol(gene)
        if not norm:
            return [], f"SKIP {gene}: could not normalize"
        targets, deg_map, err = load_deg_targets(
            args.deg_dir, gene, args.p_threshold, args.prefer_fdr,
        )
        if err:
            return [], f"SKIP {gene}: {err}"
        if args.limit_targets > 0:
            targets = targets[:args.limit_targets]
        return run_4hop_for_gene(
            norm, targets, deg_map, neo4j_timeout=args.neo4j_timeout,
        )

    with ThreadPoolExecutor(max_workers=args.path_workers) as ex:
        futs = {ex.submit(_job, g): g for g in genes}
        for fut in as_completed(futs):
            rows, msg = fut.result()
            logger.info(msg)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        logger.warning("No rows produced. Exiting.")
        return

    logger.info("Extraction complete: %d rows", len(df))
    logger.info("Enriching evidence + PMIDs (Neo4j)...")
    df = enrich_evidence_4hop(df, neo4j_batch_size=args.neo4j_evidence_batch_size)

    df = _reorder_columns(df)
    write_split_outputs(df, args.output_main, args.output_self_targets, logger)


if __name__ == "__main__":
    main()
