# indra_perturbseq

Using INDRA to explain Perturb-seq data through multi-hop pathway analysis.

## Installation

```bash
pip install -e .
```

## Package Structure

```
src/indra_perturbseq/
├── __init__.py           # Package init, logging config
├── graph.py              # Load INDRA network export (.pkl), node helpers
├── hgnc.py               # HGNC symbol normalization
├── deg.py                # DEG file loading, p-value column detection
├── statements.py         # Statement selection (best_statement, iter_incdec)
├── evidence.py           # Evidence fetching from db.indra.bio and Neo4j
├── mesh.py               # MeSH term annotation via INDRA CoGEx
├── gene_lists.py         # Gene list loading (endothelial, Karen sources)
├── permutation.py        # Label-permuted network view for null models
├── pipelines/
│   ├── onehop.py         # 1-hop pathway extraction
│   ├── twohop.py         # 2-hop pathway extraction
│   ├── threehop.py       # 3-hop pathway extraction (INDRA pathfinding)
│   ├── multihop.py       # Unified multi-hop CLI with waterfall exclusion
│   └── permuted.py       # Permuted-network pathfinding
├── evaluation/
│   ├── tpr_fpr.py        # TPR/FPR computation across thresholds
│   ├── false_positives.py # FP path export on negative target sets
│   └── outliers.py       # Hub/outlier gene evaluation (TP53, CDKN1A)
└── postprocessing/
    └── postprocess.py    # MeSH filtering + (source,target) deduplication
```

## CLI Usage

After installation, the following commands are available:

```bash
# 1-hop pipeline
indra-ps-1hop \
    --graph-pkl path/to/graph.pkl \
    --perturbations-csv path/to/target_validation_expanded.csv \
    --de-dir path/to/de_results_per_gene \
    --out-csv-main output/1hop_main.csv \
    --out-csv-self output/1hop_self.csv

# 2-hop pipeline
indra-ps-2hop \
    --graph-pkl path/to/graph.pkl \
    --genes-csv path/to/target_validation_expanded.csv \
    --de-dir path/to/de_results_per_gene \
    --endothelial-list path/to/endothelial_present_plus_manual.csv \
    --mesh-reference path/to/comprehensive_mesh_list.csv \
    --out-csv-main output/2hop_main.csv \
    --out-csv-self output/2hop_self.csv

# 3-hop pipeline
indra-ps-3hop \
    --graph-pkl path/to/graph.pkl \
    --genes-csv path/to/target_validation_expanded.csv \
    --de-dir path/to/de_results_per_gene \
    --endothelial-list path/to/endothelial_present_plus_manual.csv \
    --mesh-reference path/to/comprehensive_mesh_list.csv \
    --out-csv-raw output/3hop_raw.csv \
    --out-csv-main output/3hop_main.csv \
    --out-csv-self output/3hop_self.csv

# Unified multi-hop (1+2+3 with waterfall)
indra-ps-multihop \
    --graph-pkl path/to/graph.pkl \
    --endo-list path/to/endothelial_present_plus_manual.csv \
    --deg-dir path/to/de_results_per_gene \
    --out-dir output/ \
    --genes CCM2 KLF2 MAP2K5

# Permuted-network evaluation
indra-ps-permuted \
    --graph-pkl path/to/graph.pkl \
    --genes-csv path/to/target_validation_expanded.csv \
    --de-dir path/to/de_results_per_gene \
    --seed 42 \
    --out-1hop-csv output/permuted_1hop.csv \
    --out-2hop-csv output/permuted_2hop.csv

# TPR/FPR computation
indra-ps-tpr-fpr \
    --paths-csv output/3hop_raw.csv \
    --tv-path path/to/target_validation_expanded.csv \
    --de-dir path/to/de_results_per_gene \
    --out-csv output/tpr_fpr_3hop.csv

# False-positive path export
indra-ps-fp-eval \
    --graph-pkl path/to/graph.pkl \
    --genes-csv path/to/target_validation_expanded.csv \
    --de-dir path/to/de_results_per_gene \
    --out-1hop-csv output/fp_1hop.csv \
    --out-2hop-csv output/fp_2hop.csv
```

All commands support `--help` for full argument documentation.

## Python API

```python
from indra_perturbseq.graph import load_graph
from indra_perturbseq.hgnc import normalize_hgnc_symbol
from indra_perturbseq.pipelines.twohop import run_2hop_for_gene

graph, elapsed = load_graph("path/to/graph.pkl")
symbol = normalize_hgnc_symbol("TP53")
```
