"""Bulk RNA DEG generation backends for raw YAML pipeline inputs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from indra_perturbseq.deg_generation.common import (
    safe_source_name,
    standardize_pydeseq2_deg,
    write_deg_csv,
)
from indra_perturbseq.indra_pipeline.config import PipelineConfig
from indra_perturbseq.indra_pipeline.inputs import read_table

logger = logging.getLogger(__name__)


def generate_bulk_degs(cfg: PipelineConfig) -> tuple[Path, list[str]]:
    """Generate per-source DEG CSVs from a bulk RNA count matrix."""
    backend = cfg.input.deg_backend.lower()
    if backend == "pydeseq2":
        return generate_pydeseq2_degs(cfg)
    if backend == "ttest":
        return generate_ttest_degs(cfg)
    raise ValueError("input.deg_backend must be 'pydeseq2' or 'ttest'.")


def generate_pydeseq2_degs(cfg: PipelineConfig) -> tuple[Path, list[str]]:
    """Generate per-source DEG CSVs with PyDESeq2."""
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.default_inference import DefaultInference
        from pydeseq2.ds import DeseqStats
    except ImportError as exc:
        raise RuntimeError(
            "Bulk RNA DEG generation with PyDESeq2 requires the optional "
            "'pydeseq2' dependency. Install with `pip install -e .[bulk]`."
        ) from exc

    counts, metadata = _load_bulk_inputs(cfg)
    out_dir = Path(cfg.input.deg_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for source, sub_counts, sub_meta in _iter_bulk_comparisons(counts, metadata, cfg):
        inference = DefaultInference(n_cpus=cfg.input.n_cpus)
        dds = DeseqDataSet(
            counts=sub_counts,
            metadata=sub_meta,
            design="~_indra_condition",
            refit_cooks=True,
            inference=inference,
        )
        dds.deseq2()
        stats = DeseqStats(
            dds,
            contrast=["_indra_condition", source, "control"],
            inference=inference,
            quiet=True,
        )
        stats.summary()
        result = standardize_pydeseq2_deg(stats.results_df, source=source)
        result["condition"] = source
        result["comparison_name"] = f"{source}_vs_control"
        result["n_perturbed_samples"] = int((sub_meta["_indra_condition"] == source).sum())
        result["n_control_samples"] = int((sub_meta["_indra_condition"] == "control").sum())
        write_deg_csv(result, out_dir, source)
        written.append(safe_source_name(source))

    logger.info("Wrote %d PyDESeq2 DEG CSVs to %s", len(written), out_dir)
    return out_dir, written


def generate_ttest_degs(cfg: PipelineConfig) -> tuple[Path, list[str]]:
    """Generate per-source DEG CSVs with Welch t-tests and BH correction."""
    try:
        from scipy.stats import ttest_ind
        from statsmodels.stats.multitest import multipletests
    except ImportError as exc:
        raise RuntimeError(
            "Bulk RNA t-test DEG generation requires optional dependencies "
            "`scipy` and `statsmodels`. Install with `pip install -e .[bulk-ttest]`."
        ) from exc

    counts, metadata = _load_bulk_inputs(cfg)
    out_dir = Path(cfg.input.deg_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for source, sub_counts, sub_meta in _iter_bulk_comparisons(counts, metadata, cfg):
        log_counts = np.log2(sub_counts.astype(float) + 1.0)
        perturbed = log_counts.loc[sub_meta["_indra_condition"] == source]
        control = log_counts.loc[sub_meta["_indra_condition"] == "control"]
        lfc = perturbed.mean(axis=0) - control.mean(axis=0)
        _stat, pvals = ttest_ind(perturbed, control, axis=0, equal_var=False)
        pvals = np.where(np.isnan(pvals), 1.0, pvals)
        _reject, padj, _alphac_sidak, _alphac_bonf = multipletests(pvals, method="fdr_bh")
        result = standardize_pydeseq2_deg(
            pd.DataFrame({
                "target": log_counts.columns,
                "log2FoldChange": lfc.to_numpy(),
                "pvalue": pvals,
                "padj": padj,
                "deg_method": "ttest",
            }),
            source=source,
        )
        write_deg_csv(result, out_dir, source)
        written.append(safe_source_name(source))

    logger.info("Wrote %d t-test DEG CSVs to %s", len(written), out_dir)
    return out_dir, written


def _load_bulk_inputs(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = read_table(cfg.input.counts_path)
    metadata = read_table(cfg.input.metadata_path)

    sample_col = cfg.input.sample_column
    if sample_col not in metadata.columns:
        raise ValueError(
            f"Bulk metadata missing sample_column '{sample_col}'. "
            f"Columns: {metadata.columns.tolist()}"
        )
    metadata = metadata.copy()
    metadata[sample_col] = metadata[sample_col].astype(str)
    metadata = metadata.set_index(sample_col, drop=False)

    counts = _orient_counts(counts, metadata.index.astype(str).tolist(), cfg)
    counts = counts.apply(pd.to_numeric, errors="coerce").fillna(0)
    counts = counts.loc[:, counts.sum(axis=0) > 0]
    return counts, metadata


def _orient_counts(
    counts: pd.DataFrame,
    samples: list[str],
    cfg: PipelineConfig,
) -> pd.DataFrame:
    orientation = cfg.input.counts_orientation
    sample_set = set(samples)
    if orientation == "samples_by_genes":
        sample_col = cfg.input.sample_column
        if sample_col in counts.columns:
            out = counts.set_index(sample_col)
        else:
            out = counts.set_index(counts.columns[0])
        out.index = out.index.astype(str)
        return out.loc[[s for s in samples if s in out.index]]

    if orientation != "genes_by_samples":
        raise ValueError("input.counts_orientation must be 'genes_by_samples' or 'samples_by_genes'.")

    gene_col = _gene_column(counts, cfg.input.count_gene_column)
    matrix = counts.set_index(gene_col)
    matrix.index = matrix.index.astype(str)
    sample_cols = [col for col in matrix.columns if str(col) in sample_set]
    if not sample_cols:
        raise ValueError("No count-matrix sample columns match metadata sample IDs.")
    out = matrix[sample_cols].T
    out.index = out.index.astype(str)
    return out


def _gene_column(counts: pd.DataFrame, configured: str | None) -> str:
    if configured and configured in counts.columns:
        return configured
    for candidate in ("gene", "Gene", "Symbol", "symbol", "target", "names"):
        if candidate in counts.columns:
            return candidate
    return counts.columns[0]


def _iter_bulk_comparisons(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    cfg: PipelineConfig,
):
    src_col = cfg.input.source_column
    cond_col = cfg.input.condition_column
    for col in (src_col, cond_col):
        if col not in metadata.columns:
            raise ValueError(f"Bulk metadata missing required column '{col}'.")

    controls = metadata[metadata[cond_col].astype(str) == str(cfg.input.control_label)].copy()
    if controls.empty:
        raise ValueError(f"No bulk control samples found for control_label={cfg.input.control_label!r}.")

    sources = [
        str(s).strip()
        for s in metadata[src_col].dropna().astype(str).unique()
        if str(s).strip() and str(s).strip() != str(cfg.input.control_label)
    ]
    if cfg.sources.values or cfg.sources.file:
        from indra_perturbseq.indra_pipeline.inputs import load_raw_sources

        requested = set(load_raw_sources(cfg))
        sources = [source for source in sources if source in requested]

    for source in sources:
        perturbed = metadata[
            (metadata[src_col].astype(str) == source)
            & (metadata[cond_col].astype(str) != str(cfg.input.control_label))
        ].copy()
        if len(perturbed) < cfg.input.min_replicates:
            logger.warning("Skipping %s: only %d perturbed samples.", source, len(perturbed))
            continue
        if len(controls) < cfg.input.min_replicates:
            raise ValueError(
                f"Only {len(controls)} control samples found; "
                f"min_replicates={cfg.input.min_replicates}."
            )

        sub_meta = pd.concat([controls, perturbed], axis=0)
        sub_meta = sub_meta.loc[[idx for idx in sub_meta.index if idx in counts.index]].copy()
        sub_meta["_indra_condition"] = [
            "control" if str(v) == str(cfg.input.control_label) else source
            for v in sub_meta[cond_col]
        ]
        sub_counts = counts.loc[sub_meta.index]
        if sub_counts.empty:
            raise ValueError(f"No count samples matched metadata for source {source}.")
        yield source, sub_counts, sub_meta
