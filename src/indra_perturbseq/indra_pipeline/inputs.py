"""Input loading and under-the-hood normalization for the INDRA pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from indra_perturbseq.deg import deg_path_for_source, pick_sig_column
from indra_perturbseq.graph import is_hgnc_node
from indra_perturbseq.hgnc import normalize_hgnc_symbol
from indra_perturbseq.indra_pipeline.config import PipelineConfig, TargetsConfig


@dataclass
class SourceRecord:
    raw: str
    node: str | None
    namespace: str | None = None
    identifier: str | None = None
    found: bool = False


@dataclass
class TargetData:
    targets: pd.DataFrame
    present_genes: set[str] = field(default_factory=set)
    input_rows: int = 0
    significant_rows: int = 0
    unique_significant_targets: int = 0


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV/TSV/XLSX input tables."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(p, sep="\t", low_memory=False)
    return pd.read_csv(p, low_memory=False)


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        s = str(value).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def load_raw_sources(cfg: PipelineConfig) -> list[str]:
    """Load source strings from config values and/or a source file."""
    raw: list[str] = []
    raw.extend(cfg.sources.values or [])
    if cfg.sources.file:
        df = read_table(cfg.sources.file)
        if cfg.sources.column not in df.columns:
            raise ValueError(
                f"Source file missing column '{cfg.sources.column}'. "
                f"Columns: {df.columns.tolist()}"
            )
        raw.extend(df[cfg.sources.column].dropna().astype(str).tolist())
    return _dedupe_preserve_order(raw)


def resolve_source(graph, raw_source: str) -> SourceRecord:
    """Resolve a source to an INDRA graph node, preferring HGNC genes."""
    raw = str(raw_source).strip()
    normalized = normalize_hgnc_symbol(raw)
    candidates = [normalized, raw, raw.upper()]

    for candidate in [c for c in candidates if c]:
        if candidate in graph and is_hgnc_node(graph, candidate):
            data = graph.nodes.get(candidate, {}) or {}
            return SourceRecord(
                raw=raw,
                node=candidate,
                namespace=data.get("ns"),
                identifier=data.get("id"),
                found=True,
            )

    for candidate in [c for c in candidates if c]:
        if candidate in graph:
            data = graph.nodes.get(candidate, {}) or {}
            return SourceRecord(
                raw=raw,
                node=candidate,
                namespace=data.get("ns"),
                identifier=data.get("id"),
                found=True,
            )

    raw_upper = raw.upper()
    for node, data in graph.nodes(data=True):
        attrs = data or {}
        attr_values = {
            str(attrs.get("name", "")).upper(),
            str(attrs.get("id", "")).upper(),
            str(node).upper(),
        }
        if raw_upper in attr_values:
            return SourceRecord(
                raw=raw,
                node=node,
                namespace=attrs.get("ns"),
                identifier=attrs.get("id"),
                found=True,
            )

    return SourceRecord(raw=raw, node=None, found=False)


def resolve_sources(graph, cfg: PipelineConfig) -> list[SourceRecord]:
    """Resolve all configured sources against the graph."""
    return [resolve_source(graph, raw) for raw in load_raw_sources(cfg)]


def _configured_pval_column(df: pd.DataFrame, target_cfg: TargetsConfig) -> str:
    if target_cfg.pval_column and target_cfg.pval_column in df.columns:
        return target_cfg.pval_column
    return pick_sig_column(df, prefer_fdr=target_cfg.prefer_fdr)


def _first_existing(columns: Iterable[str | None], df: pd.DataFrame) -> str | None:
    for col in columns:
        if col and col in df.columns:
            return col
    return None


def _targets_from_frame(df: pd.DataFrame, target_cfg: TargetsConfig) -> TargetData:
    if target_cfg.gene_column not in df.columns:
        raise ValueError(
            f"Target table missing gene column '{target_cfg.gene_column}'. "
            f"Columns: {df.columns.tolist()}"
        )

    p_col = _configured_pval_column(df, target_cfg)
    lfc_col = _first_existing(
        [target_cfg.logfc_column, "logfoldchange", "logfoldchanges", "logFC"],
        df,
    )
    deg_group_col = target_cfg.deg_group_column if (
        target_cfg.deg_group_column in df.columns if target_cfg.deg_group_column else False
    ) else None

    work = df.copy()
    input_rows = len(work)
    work[p_col] = pd.to_numeric(work[p_col], errors="coerce")
    if lfc_col:
        work[lfc_col] = pd.to_numeric(work[lfc_col], errors="coerce")

    work["target"] = work[target_cfg.gene_column].map(normalize_hgnc_symbol)
    present_genes = set(work["target"].dropna().astype(str))

    sig = work[work[p_col] < target_cfg.p_threshold].copy()
    significant_rows = len(sig)
    if sig.empty:
        return TargetData(
            targets=pd.DataFrame(columns=["target", "logfoldchange", "pval", "DEGs-group"]),
            present_genes=present_genes,
            input_rows=input_rows,
            significant_rows=significant_rows,
            unique_significant_targets=0,
        )

    rows: list[dict] = []
    for _, row in sig.iterrows():
        target = row.get("target")
        if not target:
            continue
        rows.append({
            "target": target,
            "logfoldchange": row.get(lfc_col) if lfc_col else pd.NA,
            "pval": row.get(p_col),
            "DEGs-group": row.get(deg_group_col) if deg_group_col else pd.NA,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        unique_count = 0
    else:
        out = out.sort_values("pval", na_position="last")
        out = out.drop_duplicates(subset=["target"], keep="first")
        unique_count = out["target"].nunique()

    return TargetData(
        targets=out.reset_index(drop=True),
        present_genes=present_genes,
        input_rows=input_rows,
        significant_rows=significant_rows,
        unique_significant_targets=unique_count,
    )


def load_target_data_for_source(cfg: PipelineConfig, source: SourceRecord) -> TargetData:
    """Load normalized target data for one resolved source."""
    if cfg.targets.mode == "table":
        return _targets_from_frame(read_table(cfg.targets.path), cfg.targets)

    deg_dir = cfg.targets.deg_dir or cfg.targets.path
    candidate_paths = [
        deg_path_for_source(deg_dir, source.raw),
        deg_path_for_source(deg_dir, source.node or source.raw),
    ]
    path = next((p for p in candidate_paths if Path(p).exists()), candidate_paths[0])
    return _targets_from_frame(read_table(path), cfg.targets)


def load_gene_list(path: str, column: str) -> set[str]:
    """Load and HGNC-normalize a gene list from CSV/TSV/XLSX."""
    df = read_table(path)
    if column not in df.columns:
        raise ValueError(f"Gene list missing column '{column}'. Columns: {df.columns.tolist()}")
    genes = {normalize_hgnc_symbol(x) for x in df[column].dropna().astype(str)}
    genes.discard(None)
    genes.discard("")
    return set(genes)


def resolve_intermediate_universe(
    graph,
    cfg: PipelineConfig,
    present_genes: set[str],
) -> set[str]:
    """Resolve the allowed 2-hop intermediate gene universe."""
    if cfg.intermediates.mode == "full_hgnc":
        return {node for node in graph.nodes if is_hgnc_node(graph, node)}

    if cfg.intermediates.file:
        return load_gene_list(cfg.intermediates.file, cfg.intermediates.column)

    if cfg.intermediates.mode == "user_list":
        raise ValueError("intermediates.file is required for mode='user_list'.")

    return {g for g in present_genes if g in graph and is_hgnc_node(graph, g)}
