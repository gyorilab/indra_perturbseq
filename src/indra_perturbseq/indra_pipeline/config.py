"""Configuration models and loading for the flexible INDRA pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_STMT_TYPES = (
    "IncreaseAmount",
    "DecreaseAmount",
    "Activation",
    "Inhibition",
)


@dataclass
class RunConfig:
    name: str = "indra_pipeline_run"
    output_dir: str = "outputs/indra_pipeline/run"


@dataclass
class GraphConfig:
    pkl_path: str = ""


@dataclass
class SourcesConfig:
    values: list[str] = field(default_factory=list)
    file: str | None = None
    column: str = "gene"


@dataclass
class InputColumnsConfig:
    source: str | None = None
    target: str | None = None
    pvalue: str | None = None
    logfc: str | None = None
    metadata: list[str] = field(default_factory=list)


@dataclass
class InputSignificanceConfig:
    threshold: float = 0.05
    already_significant: bool = False


@dataclass
class InputConfig:
    kind: str | None = None
    path: str | None = None
    columns: InputColumnsConfig = field(default_factory=InputColumnsConfig)
    significance: InputSignificanceConfig = field(default_factory=InputSignificanceConfig)


@dataclass
class TargetsConfig:
    mode: str = "table"
    path: str | None = None
    deg_dir: str | None = None
    gene_column: str = "names"
    pval_column: str | None = None
    logfc_column: str | None = None
    deg_group_column: str | None = None
    p_threshold: float = 0.05
    prefer_fdr: bool = False


@dataclass
class HopsConfig:
    include: list[int] = field(default_factory=lambda: [1, 2])
    stmt_types: list[str] = field(default_factory=lambda: list(DEFAULT_STMT_TYPES))
    representative_statement_only: bool = False


@dataclass
class IntermediatesConfig:
    mode: str = "present_genes"
    file: str | None = None
    column: str = "gene"


@dataclass
class EvidenceConfig:
    enabled: bool = True
    neo4j_batch_size: int = 2000
    max_evidence_texts_per_statement: int = 20


@dataclass
class MeshConfig:
    terms_path: str | None = None
    batch_size: int = 200

    @property
    def enabled(self) -> bool:
        return bool(self.terms_path)


@dataclass
class PlotsConfig:
    enabled: bool = True
    include: list[str] = field(default_factory=lambda: [
        "pval_histogram",
        "logfc_histogram",
        "logfc_vs_pval_scatter",
    ])


@dataclass
class PipelineConfig:
    run: RunConfig = field(default_factory=RunConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    input: InputConfig = field(default_factory=InputConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    targets: TargetsConfig = field(default_factory=TargetsConfig)
    hops: HopsConfig = field(default_factory=HopsConfig)
    intermediates: IntermediatesConfig = field(default_factory=IntermediatesConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    plots: PlotsConfig = field(default_factory=PlotsConfig)


def _read_config_file(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML config files require PyYAML. Use a .json config or install PyYAML."
        ) from exc
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {cfg_path}")
    return data


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    raw = data.get(key, {}) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return raw


def _config_from_dict(data: dict[str, Any]) -> PipelineConfig:
    input_raw = _section(data, "input")
    input_columns = InputColumnsConfig(**(input_raw.get("columns") or {}))
    input_significance = InputSignificanceConfig(**(input_raw.get("significance") or {}))
    input_kwargs = {
        k: v for k, v in input_raw.items()
        if k not in {"columns", "significance"}
    }
    return PipelineConfig(
        run=RunConfig(**_section(data, "run")),
        graph=GraphConfig(**_section(data, "graph")),
        input=InputConfig(
            **input_kwargs,
            columns=input_columns,
            significance=input_significance,
        ),
        sources=SourcesConfig(**_section(data, "sources")),
        targets=TargetsConfig(**_section(data, "targets")),
        hops=HopsConfig(**_section(data, "hops")),
        intermediates=IntermediatesConfig(**_section(data, "intermediates")),
        evidence=EvidenceConfig(**_section(data, "evidence")),
        mesh=MeshConfig(**_section(data, "mesh")),
        plots=PlotsConfig(**_section(data, "plots")),
    )


def load_config(path: str | Path, output_dir: str | None = None) -> PipelineConfig:
    """Load and validate a pipeline config from YAML or JSON."""
    cfg = _config_from_dict(_read_config_file(path))
    if output_dir:
        cfg.run.output_dir = output_dir
    validate_config(cfg)
    return cfg


def validate_config(cfg: PipelineConfig) -> None:
    """Validate required fields and supported V1 options."""
    if not cfg.graph.pkl_path:
        raise ValueError("graph.pkl_path is required.")

    if cfg.input.kind:
        if cfg.input.kind != "paired_table":
            raise ValueError("input.kind currently supports 'paired_table'.")
        if not cfg.input.path:
            raise ValueError("input.path is required for input.kind='paired_table'.")
        if not cfg.input.columns.source or not cfg.input.columns.target:
            raise ValueError(
                "input.columns.source and input.columns.target are required "
                "for input.kind='paired_table'."
            )
        if cfg.input.significance.threshold < 0:
            raise ValueError("input.significance.threshold must be non-negative.")
    elif not cfg.sources.values and not cfg.sources.file:
        raise ValueError("Provide sources.values or sources.file.")

    if not cfg.input.kind:
        if cfg.targets.mode not in {"table", "deg_dir"}:
            raise ValueError("targets.mode must be 'table' or 'deg_dir'.")
        if cfg.targets.mode == "table" and not cfg.targets.path:
            raise ValueError("targets.path is required when targets.mode='table'.")
        if cfg.targets.mode == "deg_dir" and not (cfg.targets.deg_dir or cfg.targets.path):
            raise ValueError("targets.deg_dir is required when targets.mode='deg_dir'.")
        if cfg.targets.p_threshold < 0:
            raise ValueError("targets.p_threshold must be non-negative.")

    unsupported_hops = sorted(set(cfg.hops.include) - {1, 2})
    if unsupported_hops:
        raise ValueError(f"V1 supports only hops 1 and 2, got {unsupported_hops}.")
    if not cfg.hops.stmt_types:
        raise ValueError("hops.stmt_types must contain at least one statement type.")

    if cfg.intermediates.mode not in {"present_genes", "full_hgnc", "user_list"}:
        raise ValueError(
            "intermediates.mode must be 'present_genes', 'full_hgnc', or 'user_list'."
        )
    if cfg.intermediates.mode == "user_list" and not cfg.intermediates.file:
        raise ValueError("intermediates.file is required for mode='user_list'.")
