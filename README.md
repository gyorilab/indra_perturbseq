# indra_perturbseq

Config-driven INDRA pipeline for explaining Perturb-seq source-target effects through
1-hop and 2-hop INDRA graph paths.

## What Is Retained

This branch is intentionally focused on the current INDRA pipeline:

```text
src/indra_perturbseq/
  deg_generation/      # optional raw single-cell and bulk RNA DEG generation
  indra_pipeline/      # config, input loading, path search, enrichment, outputs
  services/            # db.indra.bio and Neo4j adapters used by enrichment
  deg.py               # DEG column/path helpers for pipeline inputs
  evidence.py          # optional statement evidence enrichment
  graph.py             # INDRA graph loading and node helpers
  hgnc.py              # source/target gene normalization
  mesh.py              # optional MeSH enrichment
  runtime.py           # CLI logging helpers
  statements.py        # INDRA statement URL formatting
```

Legacy preprocessing, evaluation, postprocessing, standalone plotting, permutation,
and older hop-specific CLIs have been removed from this branch.

## Installation

```bash
pip install -e .
```

Raw input backends are optional:

```bash
pip install -e ".[single-cell]"  # Scanpy / AnnData
pip install -e ".[bulk]"         # PyDESeq2
pip install -e ".[all]"          # all optional DEG backends
```

## Run

```bash
indra-perturbseq --config configs/indra_pipeline_example.yaml
```

The shorter alias is also installed:

```bash
indra-ps-pipeline --config configs/indra_pipeline_example.yaml
```

Before starting a run, validate that the YAML is structurally valid and that
referenced input paths exist:

```bash
indra-perturbseq --config configs/indra_pipeline_example.yaml --validate-only
```

You can also run the repository script directly during development:

```bash
python scripts/run_indra_pipeline.py --config configs/indra_pipeline_example.yaml
```

Use `--output-dir` to override `run.output_dir` from the config.

For raw expression configs, generate DEG CSVs and stop:

```bash
indra-perturbseq --config configs/raw_perturbseq_scanpy_example.yaml --deg-only
```

Reuse an existing `input.deg_output_dir` instead of regenerating DEGs:

```bash
indra-perturbseq --config configs/raw_bulk_rna_pydeseq2_example.yaml --skip-deg
```

## Configuration

The pipeline accepts YAML or JSON config files. See
`configs/indra_pipeline_example.yaml`.

Supported input modes:

- `input.kind: paired_table` for a table with source and target columns.
- `targets.mode: table` for one shared target table with configured sources.
- `targets.mode: deg_dir` for per-source DEG files named `<SOURCE>_vs_control.csv`.
- `input.kind: raw_perturbseq` or `raw_scrna` for Scanpy DEG generation from `.h5ad`.
- `input.kind: raw_bulk_rna` for bulk RNA DEG generation from counts plus metadata.

Raw input modes first write standardized DEG CSVs to `input.deg_output_dir`, then
continue through the same `targets.mode: deg_dir` pathway as precomputed DEG runs.
Each generated file uses the canonical columns:

- `names`
- `logfoldchanges`
- `pvals`
- `pvals_adj`

Example configs:

- `configs/indra_pipeline_example.yaml`
- `configs/precomputed_deg_dir_example.yaml`
- `configs/raw_perturbseq_scanpy_example.yaml`
- `configs/raw_bulk_rna_pydeseq2_example.yaml`

Supported hops:

- `1`: direct source-to-target paths.
- `2`: source-to-intermediate-to-target paths.

## Outputs

Each run writes:

- `1hop_results.csv`
- `2hop_results.csv`
- `run_summary.csv`

When `plots.enabled: true`, the pipeline also writes visualization artifacts under
`<output_dir>/plots/`. Currently supported plot names are:

- `pval_histogram`
- `logfc_histogram`
- `logfc_vs_pval_scatter`

## Tests

```bash
pytest
```
