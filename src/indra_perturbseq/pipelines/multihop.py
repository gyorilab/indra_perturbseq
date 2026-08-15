"""CLI entrypoint for configurable multi-hop path discovery.
This module owns command-line concerns only: argument parsing (including.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from indra_perturbseq.graph import is_hgnc_node, load_graph
from indra_perturbseq.hgnc import normalize_hgnc_symbol
from indra_perturbseq.pipelines.common import warn_deprecated_flags
from indra_perturbseq.pipelines.enrichment import (
    add_enrichment_cli_args,
    annotate_mesh_terms,
    enrich_evidence,
    validate_enrichment_args,
)
from indra_perturbseq.pipelines.multihop_core import (
    load_intermediates_from_args,
    load_sources_from_args,
    load_targets_from_args,
    process_gene,
)
from indra_perturbseq.runtime import add_log_level_arg, configure_logging
from indra_perturbseq.utils.selected_statement_cache import SelectedStatementCache

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    warn_deprecated_flags(
        argv,
        {
            "--endothelial-list": "--intermediate-genes-csv",
            "--endo-list": "--intermediate-genes-csv",
            "--out-dir": "--output-dir",
            "--verbose": "--log-level DEBUG",
        },
        logger,
    )
    ap = argparse.ArgumentParser(
        description="INDRA multi-hop path discovery (supports hop > 3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--graph-pkl", required=True)
    ap.add_argument(
        "--intermediate-genes-csv",
        "--endothelial-list",
        "--endo-list",
        dest="intermediate_genes_csv",
        default=None,
        help="CSV containing allowed intermediate genes (legacy: endothelial list)",
    )
    ap.add_argument(
        "--intermediate-genes",
        nargs="+",
        default=None,
        help="Allowed intermediate genes as CLI list",
    )
    ap.add_argument(
        "--intermediate-genes-file",
        default=None,
        help="Plain text intermediate gene list (newline/comma/space separated)",
    )
    ap.add_argument("--deg-dir", required=True, help="Folder with <GENE>_vs_control.csv")
    ap.add_argument("--output-dir", "--out-dir", required=True)

    ap.add_argument("--genes", nargs="+")
    ap.add_argument("--source-genes-file", default=None,
                    help="Plain text source gene list (newline/comma/space separated)")
    ap.add_argument("--source-genes-csv", default=None)
    ap.add_argument("--gene-column", default="Gene")
    ap.add_argument("--filter-column", default=None)
    ap.add_argument("--filter-value", default=None)
    ap.add_argument(
        "--endothelial-column",
        "--intermediate-column",
        dest="endothelial_column",
        default="gene",
    )

    ap.add_argument("--target-genes", nargs="+", default=None)
    ap.add_argument("--target-genes-file", default=None,
                    help="Plain text target gene list (newline/comma/space separated)")
    ap.add_argument("--target-genes-csv", default=None)
    ap.add_argument("--target-column", default="gene")
    ap.add_argument("--target-filter-column", default=None)
    ap.add_argument("--target-filter-value", default=None)

    ap.add_argument("--hops", nargs="+", type=int, default=[1, 2, 3],
                    help="Hop lengths to run, e.g. --hops 1 2 3 4 5")
    ap.add_argument("--no-waterfall", action="store_true")
    ap.add_argument("--max-paths-3hop", type=int, default=1,
                    help="Per source-target cap for 3-hop mode")
    ap.add_argument("--max-paths-per-pair", type=int, default=1,
                    help="Per source-target cap for hops >= 4")
    ap.add_argument("--prefer-fdr", action="store_true")
    ap.add_argument("--limit-genes", type=int, default=0)
    ap.add_argument("--combine-output", action="store_true")
    ap.add_argument("--output-filename", default="all_genes_all_hops.csv")
    ap.add_argument(
        "--selected-edge-cache-out",
        default=None,
        help="Optional CSV path to persist selected edge statements during extraction.",
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    add_enrichment_cli_args(ap)
    add_log_level_arg(ap, default="INFO")
    args = ap.parse_args(argv)
    validate_enrichment_args(args, ap)
    return args


def _validate_paths(args: argparse.Namespace) -> None:
    required_paths = [
        ("--graph-pkl", args.graph_pkl),
        ("--deg-dir", args.deg_dir),
    ]
    for label, path in required_paths:
        if not os.path.exists(path):
            logger.error("%s not found: %s", label, path)
            sys.exit(1)
    for label, path in [
        ("--source-genes-file", args.source_genes_file),
        ("--source-genes-csv", args.source_genes_csv),
        ("--intermediate-genes-file", args.intermediate_genes_file),
        ("--intermediate-genes-csv", args.intermediate_genes_csv),
        ("--target-genes-file", args.target_genes_file),
        ("--target-genes-csv", args.target_genes_csv),
    ]:
        if path and not os.path.exists(path):
            logger.error("%s not found: %s", label, path)
            sys.exit(1)


def _resolve_gene_sets(args: argparse.Namespace, graph) -> tuple[list[tuple[str, str | None]], set[str], set[str]]:
    try:
        intermediate_set = load_intermediates_from_args(args)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)
    intermediate_set = {g for g in intermediate_set if g in graph and is_hgnc_node(graph, g)}
    if not intermediate_set:
        logger.error("No valid in-graph HGNC intermediates after filtering")
        sys.exit(1)
    logger.info("Intermediate whitelist in-graph: %d", len(intermediate_set))

    try:
        target_set = load_targets_from_args(args)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)
    if target_set is None:
        target_set = set(intermediate_set)
        logger.info("Target set not provided; using intermediate whitelist as target set.")
    target_set = {g for g in target_set if g in graph and is_hgnc_node(graph, g)}
    if not target_set:
        logger.error("No valid in-graph HGNC targets after filtering")
        sys.exit(1)
    logger.info("Target set in-graph: %d", len(target_set))

    try:
        raw_genes = load_sources_from_args(args)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)
    if not raw_genes:
        logger.error("No source genes resolved from input")
        sys.exit(1)

    gene_pairs: list[tuple[str, str | None]] = [(r, normalize_hgnc_symbol(r)) for r in raw_genes]
    logger.info("Source genes to process: %d", len(gene_pairs))
    return gene_pairs, intermediate_set, target_set


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.verbose:
        args.log_level = "DEBUG"
    configure_logging(args.log_level)

    args.hops = sorted(set(args.hops))
    if not args.hops or args.hops[0] < 1:
        logger.error("All values in --hops must be >= 1")
        sys.exit(1)
    max_hop = max(args.hops)

    _validate_paths(args)
    os.makedirs(args.output_dir, exist_ok=True)
    graph, _ = load_graph(args.graph_pkl)
    gene_pairs, intermediate_set, target_set = _resolve_gene_sets(args, graph)
    selection_cache = SelectedStatementCache() if args.selected_edge_cache_out else None

    results: dict[str, pd.DataFrame] = {}

    def _job(raw, norm):
        return raw, process_gene(
            raw,
            norm,
            graph,
            intermediate_set,
            target_set,
            args,
            max_hop,
            logger,
            selection_cache=selection_cache,
        )

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_job, r, n): r for r, n in gene_pairs}
            for fut in as_completed(futs):
                raw, df = fut.result()
                if df is not None:
                    results[raw] = df
    else:
        for raw, norm in gene_pairs:
            _, df = _job(raw, norm)
            if df is not None:
                results[raw] = df

    if not results:
        logger.error("No results produced. Exiting.")
        sys.exit(1)

    if args.combine_output:
        combined = pd.concat(list(results.values()), ignore_index=True)
        combined = enrich_evidence(combined, args)
        combined = annotate_mesh_terms(combined, args)
        out = os.path.join(args.output_dir, args.output_filename)
        combined.to_csv(out, index=False)
        logger.info("Combined: %d rows -> %s", len(combined), out)
    else:
        for raw, df in results.items():
            df = enrich_evidence(df, args)
            df = annotate_mesh_terms(df, args)
            out = os.path.join(args.output_dir, f"{raw}_all_hops.csv")
            df.to_csv(out, index=False)
            logger.info("%s: %d rows -> %s", raw, len(df), out)

    if selection_cache is not None:
        selection_cache.write_csv(args.selected_edge_cache_out)
        logger.info(
            "Selected edge cache: %d records -> %s",
            len(selection_cache),
            args.selected_edge_cache_out,
        )


if __name__ == "__main__":
    main()
