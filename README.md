# indra_perturbseq

Config-driven INDRA pipeline for explaining Perturb-seq source-target effects through
1-hop and 2-hop INDRA graph paths.

The package contains the reusable YAML pipeline, raw DEG generation, INDRA graph
path search, enrichment, outputs, and visualizations. Legacy preprocessing,
evaluation, postprocessing, permutation, and older one-off CLIs are out of scope.

## Installation

```bash
pip install -e .
```

Raw input backends are optional:

```bash
pip install -e ".[single-cell]"
pip install -e ".[bulk]"
pip install -e ".[all]"
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
