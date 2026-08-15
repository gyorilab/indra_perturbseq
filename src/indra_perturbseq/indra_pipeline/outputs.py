"""Output writing, run summaries, and plots for the INDRA pipeline."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

from indra_perturbseq.indra_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

ONEHOP_COLUMNS = [
    "source", "target", "stmt_type", "DEGs-group", "logfoldchange", "pval",
    "belief", "evidence_count", "source_counts", "stmt_hash", "indra_url",
    "evidence_text", "pmids", "mesh_terms", "source_raw", "target_raw",
    "pvalue_column",
]

TWOHOP_COLUMNS = [
    "source", "intermediate", "target", "stmt_type_1", "stmt_type_2",
    "DEGs-group", "logfoldchange", "pval", "belief_1", "belief_2",
    "evidence_count_1", "evidence_count_2", "source_counts_1",
    "source_counts_2", "stmt_hash_1", "stmt_hash_2", "indra_url_1",
    "indra_url_2", "evidence_text_1", "evidence_text_2", "pmids_1",
    "pmids_2", "mesh_terms_1", "mesh_terms_2", "source_raw", "target_raw",
    "pvalue_column",
]


def _ordered(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    extras = [c for c in out.columns if c not in columns]
    return out[columns + extras]


def _unique_hash_count(df: pd.DataFrame, columns: list[str]) -> int:
    hashes: set[str] = set()
    for col in columns:
        if col not in df.columns:
            continue
        values = df[col].dropna().astype(str).str.strip()
        hashes.update(v for v in values if v)
    return len(hashes)


def build_summary(
    cfg: PipelineConfig,
    stats: dict[str, object],
    onehop: pd.DataFrame,
    twohop: pd.DataFrame,
) -> pd.DataFrame:
    """Build compact key/value run summary."""
    rows = [
        ("run_name", cfg.run.name),
        ("output_dir", cfg.run.output_dir),
        ("mesh_enabled", cfg.mesh.enabled),
        ("evidence_enabled", cfg.evidence.enabled),
    ]
    rows.extend((k, v) for k, v in stats.items())
    rows.extend([
        ("onehop_rows", len(onehop)),
        ("onehop_unique_targets", onehop["target"].nunique() if "target" in onehop else 0),
        ("onehop_unique_source_target_pairs",
         onehop[["source", "target"]].drop_duplicates().shape[0]
         if {"source", "target"}.issubset(onehop.columns) else 0),
        ("onehop_unique_statement_hashes", _unique_hash_count(onehop, ["stmt_hash"])),
        ("twohop_rows", len(twohop)),
        ("twohop_unique_targets", twohop["target"].nunique() if "target" in twohop else 0),
        ("twohop_unique_intermediates",
         twohop["intermediate"].nunique() if "intermediate" in twohop else 0),
        ("twohop_unique_source_target_pairs",
         twohop[["source", "target"]].drop_duplicates().shape[0]
         if {"source", "target"}.issubset(twohop.columns) else 0),
        ("twohop_unique_statement_hashes",
         _unique_hash_count(twohop, ["stmt_hash_1", "stmt_hash_2"])),
    ])
    return pd.DataFrame(rows, columns=["metric", "value"])


def write_outputs(
    cfg: PipelineConfig,
    onehop: pd.DataFrame,
    twohop: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, str]:
    """Write CSV outputs and return their paths."""
    out_dir = Path(cfg.run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    onehop_path = out_dir / "1hop_results.csv"
    twohop_path = out_dir / "2hop_results.csv"
    summary_path = out_dir / "run_summary.csv"

    _ordered(onehop, ONEHOP_COLUMNS).to_csv(onehop_path, index=False)
    _ordered(twohop, TWOHOP_COLUMNS).to_csv(twohop_path, index=False)
    summary.to_csv(summary_path, index=False)

    logger.info("Wrote %s", onehop_path)
    logger.info("Wrote %s", twohop_path)
    logger.info("Wrote %s", summary_path)
    return {
        "onehop": str(onehop_path),
        "twohop": str(twohop_path),
        "summary": str(summary_path),
    }


def write_plots(cfg: PipelineConfig, onehop: pd.DataFrame, twohop: pd.DataFrame) -> list[str]:
    """Write configured plots when enabled and possible."""
    if not cfg.plots.enabled:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Skipping plots: matplotlib is not installed.")
        return []

    plot_dir = Path(cfg.run.output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    requested = set(cfg.plots.include)

    for label, df in (("1hop", onehop), ("2hop", twohop)):
        if df.empty:
            continue
        work = df.copy()
        work["pval"] = pd.to_numeric(work.get("pval"), errors="coerce")
        work["logfoldchange"] = pd.to_numeric(work.get("logfoldchange"), errors="coerce")

        if "pval_histogram" in requested and work["pval"].notna().any():
            fig, ax = plt.subplots(figsize=(7, 5))
            work["pval"].dropna().plot(kind="hist", bins=50, ax=ax)
            ax.set_xlabel("p-value")
            ax.set_title(f"{label} p-value distribution")
            path = plot_dir / f"{label}_pval_histogram.png"
            fig.tight_layout()
            fig.savefig(path, dpi=200)
            plt.close(fig)
            written.append(str(path))

        if "logfc_histogram" in requested and work["logfoldchange"].notna().any():
            fig, ax = plt.subplots(figsize=(7, 5))
            work["logfoldchange"].dropna().plot(kind="hist", bins=50, ax=ax)
            ax.set_xlabel("log fold change")
            ax.set_title(f"{label} logFC distribution")
            path = plot_dir / f"{label}_logfc_histogram.png"
            fig.tight_layout()
            fig.savefig(path, dpi=200)
            plt.close(fig)
            written.append(str(path))

        if {"logfc_vs_pval_scatter", "logfc_vs_pvalue_scatter"} & requested:
            scatter = work[["logfoldchange", "pval"]].dropna()
            scatter = scatter[scatter["pval"] > 0]
            if not scatter.empty:
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(
                    scatter["logfoldchange"],
                    [-math.log10(p) for p in scatter["pval"]],
                    s=12,
                    alpha=0.65,
                )
                ax.set_xlabel("log fold change")
                ax.set_ylabel("-log10(p-value)")
                ax.set_title(f"{label} logFC vs p-value")
                path = plot_dir / f"{label}_logfc_vs_pval_scatter.png"
                fig.tight_layout()
                fig.savefig(path, dpi=200)
                plt.close(fig)
                written.append(str(path))

    return written
