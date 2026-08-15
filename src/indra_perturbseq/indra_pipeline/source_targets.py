"""Canonical source-target table construction and graph resolution."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from indra_perturbseq.graph import is_hgnc_node
from indra_perturbseq.indra_pipeline.config import PipelineConfig
from indra_perturbseq.indra_pipeline.inputs import read_table, resolve_source


@dataclass
class SourceTargetTable:
    rows: pd.DataFrame
    input_rows: int
    significant_rows: int

    @property
    def present_hgnc_genes(self) -> set[str]:
        genes: set[str] = set()
        for col in ("source_node", "target_node"):
            if col not in self.rows.columns:
                continue
            ns_col = col.replace("_node", "_ns")
            if ns_col not in self.rows.columns:
                continue
            genes.update(
                self.rows.loc[self.rows[ns_col] == "HGNC", col]
                .dropna()
                .astype(str)
            )
        return genes


def build_source_target_table(cfg: PipelineConfig) -> SourceTargetTable:
    """Build the canonical source-target table from configured input."""
    if cfg.input.kind != "paired_table":
        raise ValueError("Canonical input currently supports input.kind='paired_table'.")

    cols = cfg.input.columns
    df = read_table(cfg.input.path)
    required = [cols.source, cols.target]
    if cols.pvalue:
        required.append(cols.pvalue)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input table missing required columns: {missing}")

    out = pd.DataFrame({
        "source_raw": df[cols.source].astype(str).str.strip(),
        "target_raw": df[cols.target].astype(str).str.strip(),
    })
    if cols.pvalue:
        out["pvalue"] = pd.to_numeric(df[cols.pvalue], errors="coerce")
        out["pvalue_column"] = cols.pvalue
    else:
        out["pvalue"] = pd.NA
        out["pvalue_column"] = ""
    if cols.logfc and cols.logfc in df.columns:
        out["logfoldchange"] = pd.to_numeric(df[cols.logfc], errors="coerce")
    else:
        out["logfoldchange"] = pd.NA

    for col in cols.metadata:
        if col in df.columns:
            out_col = f"raw_{col}" if col in out.columns or col == "pval" else col
            out[out_col] = df[col]

    input_rows = len(out)
    out = out[(out["source_raw"] != "") & (out["target_raw"] != "")].copy()
    if cols.pvalue and not cfg.input.significance.already_significant:
        out = out[out["pvalue"] < cfg.input.significance.threshold].copy()
    significant_rows = len(out)
    return SourceTargetTable(out.reset_index(drop=True), input_rows, significant_rows)


def resolve_source_target_table(graph, table: SourceTargetTable) -> SourceTargetTable:
    """Resolve canonical raw source/target values to INDRA graph nodes."""
    cache: dict[str, object] = {}

    def _resolve(raw: object):
        key = str(raw).strip()
        if key not in cache:
            cache[key] = resolve_source(graph, key)
        return cache[key]

    rows: list[dict] = []
    for row in table.rows.to_dict("records"):
        src = _resolve(row["source_raw"])
        tgt = _resolve(row["target_raw"])
        row.update({
            "source_node": src.node,
            "source_ns": src.namespace,
            "source_id": src.identifier,
            "source_found": src.found,
            "target_node": tgt.node,
            "target_ns": tgt.namespace,
            "target_id": tgt.identifier,
            "target_found": tgt.found,
        })
        rows.append(row)

    resolved = pd.DataFrame(rows)
    usable = resolved[
        resolved["source_found"]
        & resolved["target_found"]
        & resolved["source_node"].notna()
        & resolved["target_node"].notna()
    ].copy()
    return SourceTargetTable(usable.reset_index(drop=True), table.input_rows, table.significant_rows)


def full_hgnc_nodes(graph) -> set[str]:
    """Return all HGNC nodes from the graph."""
    return {node for node in graph.nodes if is_hgnc_node(graph, node)}
