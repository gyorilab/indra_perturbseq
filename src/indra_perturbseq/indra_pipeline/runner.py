"""End-to-end orchestration for the flexible INDRA pipeline."""

from __future__ import annotations

import logging

import pandas as pd

from indra_perturbseq.graph import is_hgnc_node, load_graph
from indra_perturbseq.indra_pipeline.config import PipelineConfig
from indra_perturbseq.indra_pipeline.enrichment import enrich_1hop, enrich_2hop
from indra_perturbseq.indra_pipeline.inputs import (
    SourceRecord,
    TargetData,
    load_target_data_for_source,
    resolve_intermediate_universe,
    resolve_sources,
)
from indra_perturbseq.indra_pipeline.outputs import (
    build_summary,
    write_outputs,
    write_plots,
)
from indra_perturbseq.indra_pipeline.path_search import (
    run_1hop,
    run_1hop_source_target_table,
    run_2hop,
    run_2hop_source_target_table,
)
from indra_perturbseq.indra_pipeline.source_targets import (
    build_source_target_table,
    full_hgnc_nodes,
    resolve_source_target_table,
)

logger = logging.getLogger(__name__)


def _empty_target_data() -> TargetData:
    return TargetData(
        targets=pd.DataFrame(columns=["target", "logfoldchange", "pval", "DEGs-group"]),
    )


def _load_targets(
    cfg: PipelineConfig,
    sources: list[SourceRecord],
) -> dict[str, TargetData]:
    out: dict[str, TargetData] = {}
    shared_table: TargetData | None = None
    for source in sources:
        if not source.found:
            out[source.raw] = _empty_target_data()
            continue
        try:
            if cfg.targets.mode == "table":
                if shared_table is None:
                    shared_table = load_target_data_for_source(cfg, source)
                out[source.raw] = shared_table
            else:
                out[source.raw] = load_target_data_for_source(cfg, source)
        except Exception as exc:
            logger.warning("Skipping targets for %s: %s", source.raw, exc)
            out[source.raw] = _empty_target_data()
    return out


def _summary_stats(
    cfg: PipelineConfig,
    graph,
    sources: list[SourceRecord],
    targets_by_source: dict[str, TargetData],
    intermediates: set[str],
) -> dict[str, object]:
    present: set[str] = set()
    significant: set[str] = set()
    input_rows = 0
    significant_rows = 0
    target_data_values = list(targets_by_source.values())
    for td in target_data_values:
        present |= td.present_genes
        if not td.targets.empty:
            significant |= set(td.targets["target"].dropna().astype(str))

    if cfg.targets.mode == "deg_dir":
        input_rows = sum(td.input_rows for td in target_data_values)
        significant_rows = sum(td.significant_rows for td in target_data_values)
    elif target_data_values:
        input_rows = max(td.input_rows for td in target_data_values)
        significant_rows = max(td.significant_rows for td in target_data_values)

    graph_matched_targets = {
        t for t in significant
        if t in graph and is_hgnc_node(graph, t)
    }


def _summary_stats_source_target(
    cfg: PipelineConfig,
    table,
    intermediates: set[str],
) -> dict[str, object]:
    rows = table.rows
    return {
        "input_kind": cfg.input.kind,
        "source_target_input_rows": table.input_rows,
        "source_target_significant_rows": table.significant_rows,
        "source_target_resolved_rows": len(rows),
        "sources_input": rows["source_raw"].nunique() if "source_raw" in rows else 0,
        "sources_found": rows["source_node"].nunique() if "source_node" in rows else 0,
        "sources_skipped": max(
            0,
            (rows["source_raw"].nunique() if "source_raw" in rows else 0)
            - (rows["source_node"].nunique() if "source_node" in rows else 0),
        ),
        "target_unique_significant": rows["target_raw"].nunique() if "target_raw" in rows else 0,
        "target_graph_matched": rows["target_node"].nunique() if "target_node" in rows else 0,
        "present_gene_universe_size": len(table.present_hgnc_genes),
        "intermediate_universe_size": len(intermediates),
    }


def _intermediates_for_source_target_input(graph, cfg: PipelineConfig, table) -> set[str]:
    if cfg.intermediates.mode == "full_hgnc":
        return full_hgnc_nodes(graph)
    if cfg.intermediates.file:
        from indra_perturbseq.indra_pipeline.inputs import load_gene_list

        return load_gene_list(cfg.intermediates.file, cfg.intermediates.column)
    return {g for g in table.present_hgnc_genes if g in graph and is_hgnc_node(graph, g)}
    return {
        "sources_input": len(sources),
        "sources_found": sum(1 for s in sources if s.found),
        "sources_skipped": sum(1 for s in sources if not s.found),
        "target_input_rows": input_rows,
        "target_significant_rows": significant_rows,
        "target_unique_significant": len(significant),
        "target_graph_matched": len(graph_matched_targets),
        "present_gene_universe_size": len(present),
        "intermediate_universe_size": len(intermediates),
    }


def run_pipeline(cfg: PipelineConfig) -> dict[str, object]:
    """Run the configured pipeline and return paths plus dataframes."""
    graph, load_seconds = load_graph(cfg.graph.pkl_path)

    if cfg.input.kind:
        source_target_table = resolve_source_target_table(
            graph,
            build_source_target_table(cfg),
        )
        intermediates = _intermediates_for_source_target_input(
            graph,
            cfg,
            source_target_table,
        )
        if 1 in cfg.hops.include:
            onehop = run_1hop_source_target_table(
                graph,
                source_target_table.rows,
                cfg.hops.stmt_types,
                cfg.hops.representative_statement_only,
            )
            onehop = enrich_1hop(onehop, cfg)
        else:
            onehop = pd.DataFrame()

        if 2 in cfg.hops.include:
            twohop = run_2hop_source_target_table(
                graph,
                source_target_table.rows,
                intermediates,
                cfg.hops.stmt_types,
                cfg.hops.representative_statement_only,
            )
            twohop = enrich_2hop(twohop, cfg)
        else:
            twohop = pd.DataFrame()

        stats = _summary_stats_source_target(cfg, source_target_table, intermediates)
        stats["graph_load_seconds"] = round(load_seconds, 3)
        summary = build_summary(cfg, stats, onehop, twohop)
        paths = write_outputs(cfg, onehop, twohop, summary)
        paths["plots"] = write_plots(cfg, onehop, twohop)
        return {
            "paths": paths,
            "onehop": onehop,
            "twohop": twohop,
            "summary": summary,
        }

    sources = resolve_sources(graph, cfg)
    found_sources = [s for s in sources if s.found]
    for source in sources:
        if not source.found:
            logger.warning("Source not found in graph: %s", source.raw)

    targets_by_source = _load_targets(cfg, sources)
    present_genes: set[str] = set()
    for target_data in targets_by_source.values():
        present_genes |= target_data.present_genes
    intermediates = resolve_intermediate_universe(graph, cfg, present_genes)

    if 1 in cfg.hops.include:
        onehop = run_1hop(
            graph,
            found_sources,
            targets_by_source,
            cfg.hops.stmt_types,
            cfg.hops.representative_statement_only,
        )
        onehop = enrich_1hop(onehop, cfg)
    else:
        onehop = pd.DataFrame()

    if 2 in cfg.hops.include:
        twohop = run_2hop(
            graph,
            found_sources,
            targets_by_source,
            intermediates,
            cfg.hops.stmt_types,
            cfg.hops.representative_statement_only,
        )
        twohop = enrich_2hop(twohop, cfg)
    else:
        twohop = pd.DataFrame()

    stats = _summary_stats(cfg, graph, sources, targets_by_source, intermediates)
    stats["graph_load_seconds"] = round(load_seconds, 3)
    summary = build_summary(cfg, stats, onehop, twohop)
    paths = write_outputs(cfg, onehop, twohop, summary)
    plot_paths = write_plots(cfg, onehop, twohop)
    paths["plots"] = plot_paths

    return {
        "paths": paths,
        "onehop": onehop,
        "twohop": twohop,
        "summary": summary,
    }
