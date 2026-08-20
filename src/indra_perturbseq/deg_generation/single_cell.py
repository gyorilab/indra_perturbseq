"""Scanpy DEG generation for raw single-cell inputs."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from indra_perturbseq.deg_generation.common import (
    empty_deg_frame,
    safe_source_name,
    standardize_scanpy_deg,
    write_deg_csv,
)
from indra_perturbseq.indra_pipeline.config import PipelineConfig
from indra_perturbseq.indra_pipeline.inputs import load_gene_list, load_raw_sources

logger = logging.getLogger(__name__)


def generate_single_cell_degs(cfg: PipelineConfig) -> tuple[Path, list[str]]:
    """Generate per-source DEG CSVs from a raw AnnData file."""
    try:
        import scanpy as sc
    except ImportError as exc:
        raise RuntimeError(
            "Raw single-cell DEG generation requires the optional 'scanpy' "
            "dependency. Install with `pip install -e .[single-cell]`."
        ) from exc

    input_cfg = cfg.input
    out_dir = Path(input_cfg.deg_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading AnnData: %s", input_cfg.adata_path)
    adata = sc.read_h5ad(input_cfg.adata_path)
    adata = adata.copy()

    if input_cfg.symbol_column:
        if input_cfg.symbol_column not in adata.var.columns:
            raise ValueError(
                f"adata.var missing symbol_column '{input_cfg.symbol_column}'. "
                f"Available: {adata.var.columns.tolist()}"
            )
        adata.var_names = (
            adata.var[input_cfg.symbol_column]
            .astype(str)
            .str.strip()
            .values
        )

    perturb_col = input_cfg.perturbation_column
    if perturb_col not in adata.obs.columns:
        raise ValueError(
            f"adata.obs missing perturbation_column '{perturb_col}'. "
            f"Available: {adata.obs.columns.tolist()}"
        )

    labels = adata.obs[perturb_col].astype(str).str.strip()
    if input_cfg.tss_suffix:
        labels = labels.str.replace(input_cfg.tss_suffix, "", regex=False)

    controls = {str(label).strip() for label in input_cfg.control_labels}
    adata.obs["_indra_perturbation"] = labels
    adata.obs["_indra_group"] = labels.where(~labels.isin(controls), "control")

    requested = load_raw_sources(cfg) if (cfg.sources.values or cfg.sources.file) else []
    if requested:
        keep_groups = set(requested) | controls
        adata = adata[adata.obs["_indra_perturbation"].isin(keep_groups)].copy()

    if input_cfg.normalize_log1p:
        sc.pp.normalize_total(adata, target_sum=input_cfg.target_sum)
        sc.pp.log1p(adata)

    if input_cfg.endothelial_genes_path:
        genes = load_gene_list(
            input_cfg.endothelial_genes_path,
            input_cfg.endothelial_gene_column,
        )
        mask = pd.Index(adata.var_names.astype(str)).isin(genes)
        if int(mask.sum()) == 0:
            raise RuntimeError(
                "No overlap between AnnData var_names and endothelial gene list. "
                "Use input.symbol_column if var_names are not gene symbols."
            )
        adata = adata[:, mask].copy()

    sources = requested or sorted(
        label for label in adata.obs["_indra_group"].dropna().unique()
        if label != "control"
    )
    counts = adata.obs["_indra_perturbation"].value_counts().to_dict()
    runnable = [
        source for source in sources
        if counts.get(source, 0) >= input_cfg.min_cells
    ]

    if not runnable:
        for source in sources:
            write_deg_csv(empty_deg_frame(), out_dir, source)
        logger.warning("No perturbations met min_cells=%d.", input_cfg.min_cells)
        return out_dir, [safe_source_name(source) for source in sources]

    logger.info(
        "Running Scanpy DEG (%s) for %d perturbations vs control",
        input_cfg.deg_method,
        len(runnable),
    )
    sc.tl.rank_genes_groups(
        adata,
        groupby="_indra_group",
        reference="control",
        method=input_cfg.deg_method,
        pts=True,
    )
    completed = set(adata.uns["rank_genes_groups"]["names"].dtype.names)

    written: list[str] = []
    for source in sources:
        if source not in completed:
            df = empty_deg_frame()
        else:
            df = standardize_scanpy_deg(
                sc.get.rank_genes_groups_df(adata, group=source),
                source=source,
            )
        write_deg_csv(df, out_dir, source)
        written.append(safe_source_name(source))

    logger.info("Wrote %d single-cell DEG CSVs to %s", len(written), out_dir)
    return out_dir, written
