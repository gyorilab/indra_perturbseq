from pathlib import Path
from contextlib import redirect_stdout
import io
import pickle

import networkx as nx
import pandas as pd

from indra_perturbseq.indra_pipeline.cli import main as cli_main
from indra_perturbseq.indra_pipeline.config import load_config
from indra_perturbseq.deg_generation.common import (
    standardize_pydeseq2_deg,
    standardize_scanpy_deg,
)
from indra_perturbseq.indra_pipeline.inputs import (
    resolve_intermediate_universe,
    resolve_source,
)
from indra_perturbseq.indra_pipeline.path_search import (
    run_1hop,
    run_1hop_source_target_table,
    run_2hop,
    run_2hop_source_target_table,
)
from indra_perturbseq.indra_pipeline.inputs import SourceRecord, TargetData
from indra_perturbseq.indra_pipeline.source_targets import (
    build_source_target_table,
    resolve_source_target_table,
)
from indra_perturbseq.indra_pipeline.runner import run_pipeline


def _graph():
    g = nx.DiGraph()
    g.add_node("FSHB", ns="HGNC", id="3964")
    g.add_node("FSH", ns="FPLX", id="FSH")
    g.add_node("P12345", ns="UP", id="P12345")
    g.add_node("MID", ns="HGNC", id="1")
    g.add_node("TGT", ns="HGNC", id="2")
    g.add_edge("FSH", "TGT", statements=[
        {
            "stmt_type": "IncreaseAmount",
            "belief": 0.8,
            "evidence_count": 2,
            "stmt_hash": 101,
            "source_counts": {"reach": 2},
        },
        {
            "stmt_type": "Activation",
            "belief": 0.7,
            "evidence_count": 1,
            "stmt_hash": 102,
            "source_counts": {"sparser": 1},
        },
    ])
    g.add_edge("FSH", "MID", statements=[
        {"stmt_type": "Activation", "belief": 0.9, "evidence_count": 3, "stmt_hash": 201}
    ])
    g.add_edge("MID", "TGT", statements=[
        {"stmt_type": "DecreaseAmount", "belief": 0.6, "evidence_count": 4, "stmt_hash": 301}
    ])
    g.add_edge("FSH", "P12345", statements=[
        {"stmt_type": "Activation", "belief": 0.5, "evidence_count": 1, "stmt_hash": 401}
    ])
    return g


def test_config_defaults(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
graph:
  pkl_path: graph.pkl
sources:
  values: [FSH]
targets:
  mode: table
  path: targets.csv
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.mesh.terms_path is None
    assert not cfg.mesh.enabled
    assert cfg.hops.representative_statement_only is False


def test_raw_single_cell_config_parses(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
graph:
  pkl_path: graph.pkl
input:
  kind: raw_scrna
  adata_path: matrix.h5ad
  perturbation_column: Gene
  control_labels: [negative-control, safe-targeting]
  deg_output_dir: outputs/deg
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.input.kind == "raw_scrna"
    assert cfg.input.adata_path == "matrix.h5ad"
    assert cfg.input.control_labels == ["negative-control", "safe-targeting"]


def test_raw_bulk_config_parses(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
graph:
  pkl_path: graph.pkl
input:
  kind: raw_bulk_rna
  counts_path: counts.csv
  metadata_path: metadata.csv
  sample_column: sample_id
  source_column: perturbation_gene
  condition_column: condition
  control_label: control
  deg_backend: pydeseq2
  deg_output_dir: outputs/deg
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.input.kind == "raw_bulk_rna"
    assert cfg.input.deg_backend == "pydeseq2"
    assert cfg.input.counts_orientation == "genes_by_samples"


def test_deg_standardization_scanpy_like_output():
    raw = pd.DataFrame({
        "names": ["TGT"],
        "logfoldchanges": [1.25],
        "pvals": [0.001],
        "pvals_adj": [0.01],
        "scores": [3.0],
    })
    out = standardize_scanpy_deg(raw, source="FSH")
    assert out.loc[0, "names"] == "TGT"
    assert out.loc[0, "logfoldchanges"] == 1.25
    assert out.loc[0, "pvals"] == 0.001
    assert out.loc[0, "pvals_adj"] == 0.01
    assert out.loc[0, "source"] == "FSH"
    assert "scores" in out.columns


def test_deg_standardization_pydeseq2_like_output():
    raw = pd.DataFrame({
        "target": ["TGT"],
        "log2FoldChange": [1.5],
        "pvalue": [0.002],
        "padj": [0.02],
        "baseMean": [100.0],
    })
    out = standardize_pydeseq2_deg(raw, source="FSH")
    assert out.loc[0, "names"] == "TGT"
    assert out.loc[0, "logfoldchanges"] == 1.5
    assert out.loc[0, "pvals"] == 0.002
    assert out.loc[0, "pvals_adj"] == 0.02
    assert out.loc[0, "source"] == "FSH"
    assert out.loc[0, "deg_method"] == "PyDESeq2"
    assert "baseMean" in out.columns


def test_deg_standardization_pydeseq2_index_output():
    raw = pd.DataFrame({
        "log2FoldChange": [1.5],
        "pvalue": [0.002],
        "padj": [0.02],
    }, index=pd.Index(["TGT"], name=None))
    out = standardize_pydeseq2_deg(raw, source="FSH")
    assert out.loc[0, "names"] == "TGT"
    assert out.loc[0, "pvals_adj"] == 0.02


def test_source_resolution_hgnc_and_fplx():
    g = _graph()
    assert resolve_source(g, "FSHB").node == "FSHB"
    assert resolve_source(g, "FSH").node == "FSH"
    assert resolve_source(g, "FSH").namespace == "FPLX"


def test_path_search_returns_all_statements():
    g = _graph()
    source = SourceRecord(raw="FSH", node="FSH", found=True)
    targets = TargetData(pd.DataFrame([{
        "target": "TGT",
        "logfoldchange": 1.5,
        "pval": 0.001,
        "DEGs-group": "pos",
    }]))
    stmt_types = ["IncreaseAmount", "DecreaseAmount", "Activation", "Inhibition"]

    onehop = run_1hop(g, [source], {"FSH": targets}, stmt_types)
    assert len(onehop) == 2
    assert set(onehop["stmt_hash"]) == {101, 102}

    twohop = run_2hop(g, [source], {"FSH": targets}, {"MID"}, stmt_types)
    assert len(twohop) == 1
    assert twohop.iloc[0]["stmt_hash_1"] == 201
    assert twohop.iloc[0]["stmt_hash_2"] == 301


def test_intermediate_modes_full_hgnc_and_present_genes(tmp_path):
    from indra_perturbseq.indra_pipeline.config import PipelineConfig

    g = _graph()
    cfg = PipelineConfig()
    cfg.intermediates.mode = "full_hgnc"
    assert resolve_intermediate_universe(g, cfg, set()) == {"FSHB", "MID", "TGT"}

    cfg.intermediates.mode = "present_genes"
    assert resolve_intermediate_universe(g, cfg, {"MID", "FSH"}) == {"MID"}


def test_paired_table_column_mapping_and_resolution(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    input_path = tmp_path / "pairs.csv"
    input_path.write_text(
        "\n".join([
            "source_gene,target_gene,pval_adj,logfoldchange,condition",
            "FSH,TGT,0.01,1.5,stimulated",
            "FSH,P12345,0.02,0.7,unstimulated",
            "FSH,TGT,0.20,2.0,filtered",
        ]),
        encoding="utf-8",
    )
    cfg_path.write_text(
        f"""
graph:
  pkl_path: graph.pkl
input:
  kind: paired_table
  path: {input_path}
  columns:
    source: source_gene
    target: target_gene
    pvalue: pval_adj
    logfc: logfoldchange
    metadata: [condition]
  significance:
    threshold: 0.05
    already_significant: false
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    table = build_source_target_table(cfg)
    assert len(table.rows) == 2
    assert set(table.rows["pvalue"]) == {0.01, 0.02}

    resolved = resolve_source_target_table(_graph(), table)
    assert set(resolved.rows["target_node"]) == {"TGT", "P12345"}
    assert set(resolved.rows["target_ns"]) == {"HGNC", "UP"}


def test_paired_table_path_search_preserves_metadata_and_uniprot():
    g = _graph()
    table = pd.DataFrame([
        {
            "source_raw": "FSH",
            "target_raw": "TGT",
            "source_node": "FSH",
            "target_node": "TGT",
            "pvalue": 0.01,
            "logfoldchange": 1.5,
            "condition": "stimulated",
        },
        {
            "source_raw": "FSH",
            "target_raw": "P12345",
            "source_node": "FSH",
            "target_node": "P12345",
            "pvalue": 0.02,
            "logfoldchange": 0.7,
            "condition": "unstimulated",
        },
    ])
    stmt_types = ["IncreaseAmount", "DecreaseAmount", "Activation", "Inhibition"]
    onehop = run_1hop_source_target_table(g, table, stmt_types)
    assert set(onehop["target"]) == {"TGT", "P12345"}
    assert "condition" in onehop.columns

    twohop = run_2hop_source_target_table(g, table.iloc[:1], {"MID"}, stmt_types)
    assert len(twohop) == 1
    assert twohop.iloc[0]["condition"] == "stimulated"


def test_run_pipeline_writes_tables_summary_and_plots(tmp_path):
    graph_path = tmp_path / "graph.pkl"
    with graph_path.open("wb") as fh:
        pickle.dump(_graph(), fh)

    targets_path = tmp_path / "targets.csv"
    targets_path.write_text(
        "\n".join([
            "names,pval,logfoldchange,DEGs-group",
            "TGT,0.001,1.5,pos",
            "P12345,0.002,0.7,pos",
            "IGNORED,0.20,0.1,ns",
        ]),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
run:
  name: smoke
  output_dir: {out_dir}
graph:
  pkl_path: {graph_path}
sources:
  values: [FSH]
targets:
  mode: table
  path: {targets_path}
  gene_column: names
  pval_column: pval
  logfc_column: logfoldchange
  deg_group_column: DEGs-group
  p_threshold: 0.01
intermediates:
  mode: full_hgnc
evidence:
  enabled: false
plots:
  enabled: true
  include: [pval_histogram, logfc_histogram, logfc_vs_pval_scatter]
""",
        encoding="utf-8",
    )

    result = run_pipeline(load_config(cfg_path))
    paths = result["paths"]
    assert Path(paths["onehop"]).exists()
    assert Path(paths["twohop"]).exists()
    assert Path(paths["summary"]).exists()
    assert len(paths["plots"]) == 6
    assert all(Path(path).exists() for path in paths["plots"])

    summary = pd.read_csv(paths["summary"]).set_index("metric")["value"].to_dict()
    assert str(summary["sources_found"]) == "1"
    assert str(summary["onehop_rows"]) == "2"
    assert str(summary["twohop_rows"]) == "1"


def test_raw_single_cell_runner_generates_deg_then_runs_paths(tmp_path):
    graph_path = tmp_path / "graph.pkl"
    with graph_path.open("wb") as fh:
        pickle.dump(_graph(), fh)

    adata_path = tmp_path / "matrix.h5ad"
    adata_path.write_text("placeholder", encoding="utf-8")
    deg_dir = tmp_path / "deg"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
run:
  output_dir: {tmp_path / "out"}
graph:
  pkl_path: {graph_path}
input:
  kind: raw_perturbseq
  adata_path: {adata_path}
  deg_output_dir: {deg_dir}
intermediates:
  mode: full_hgnc
evidence:
  enabled: false
plots:
  enabled: false
""",
        encoding="utf-8",
    )

    from indra_perturbseq.deg_generation import single_cell

    original = single_cell.generate_single_cell_degs

    def fake_generate(cfg):
        Path(cfg.input.deg_output_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "names": ["TGT"],
            "logfoldchanges": [1.0],
            "pvals": [0.001],
            "pvals_adj": [0.01],
        }).to_csv(Path(cfg.input.deg_output_dir) / "FSH_vs_control.csv", index=False)
        return Path(cfg.input.deg_output_dir), ["FSH"]

    single_cell.generate_single_cell_degs = fake_generate
    try:
        result = run_pipeline(load_config(cfg_path))
    finally:
        single_cell.generate_single_cell_degs = original

    assert Path(result["paths"]["onehop"]).exists()
    assert len(result["onehop"]) == 2
    assert len(result["twohop"]) == 1
    summary = result["summary"].set_index("metric")["value"].to_dict()
    assert summary["raw_input_kind"] == "raw_perturbseq"
    assert summary["deg_source_count"] == 1


def test_raw_bulk_runner_generates_deg_then_runs_paths(tmp_path):
    graph_path = tmp_path / "graph.pkl"
    with graph_path.open("wb") as fh:
        pickle.dump(_graph(), fh)

    counts_path = tmp_path / "counts.csv"
    counts_path.write_text("gene,c1,k1\nTGT,10,30\nMID,4,9\n", encoding="utf-8")
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "\n".join([
            "sample_id,perturbation_gene,condition",
            "c1,control,control",
            "k1,FSH,knockdown",
        ]),
        encoding="utf-8",
    )
    deg_dir = tmp_path / "deg"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
run:
  output_dir: {tmp_path / "out"}
graph:
  pkl_path: {graph_path}
input:
  kind: raw_bulk_rna
  counts_path: {counts_path}
  metadata_path: {metadata_path}
  deg_output_dir: {deg_dir}
intermediates:
  mode: full_hgnc
evidence:
  enabled: false
plots:
  enabled: false
""",
        encoding="utf-8",
    )

    from indra_perturbseq.deg_generation import bulk

    original = bulk.generate_bulk_degs

    def fake_generate(cfg):
        Path(cfg.input.deg_output_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "names": ["TGT"],
            "logfoldchanges": [1.0],
            "pvals": [0.001],
            "pvals_adj": [0.01],
            "deg_method": ["PyDESeq2"],
        }).to_csv(Path(cfg.input.deg_output_dir) / "FSH_vs_control.csv", index=False)
        return Path(cfg.input.deg_output_dir), ["FSH"]

    bulk.generate_bulk_degs = fake_generate
    try:
        result = run_pipeline(load_config(cfg_path))
    finally:
        bulk.generate_bulk_degs = original

    assert len(result["onehop"]) == 2
    assert len(result["twohop"]) == 1
    summary = result["summary"].set_index("metric")["value"].to_dict()
    assert summary["raw_input_kind"] == "raw_bulk_rna"
    assert summary["deg_source_count"] == 1


def test_cli_deg_only_generates_deg_and_stops(tmp_path):
    adata_path = tmp_path / "matrix.h5ad"
    adata_path.write_text("placeholder", encoding="utf-8")
    deg_dir = tmp_path / "deg"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
input:
  kind: raw_scrna
  adata_path: {adata_path}
  deg_output_dir: {deg_dir}
""",
        encoding="utf-8",
    )

    from indra_perturbseq.deg_generation import single_cell

    original = single_cell.generate_single_cell_degs

    def fake_generate(cfg):
        Path(cfg.input.deg_output_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "names": ["TGT"],
            "logfoldchanges": [1.0],
            "pvals": [0.001],
            "pvals_adj": [0.01],
        }).to_csv(Path(cfg.input.deg_output_dir) / "FSH_vs_control.csv", index=False)
        return Path(cfg.input.deg_output_dir), ["FSH"]

    single_cell.generate_single_cell_degs = fake_generate
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            status = cli_main(["--config", str(cfg_path), "--deg-only"])
    finally:
        single_cell.generate_single_cell_degs = original

    assert status == 0
    assert "DEG directory" in out.getvalue()
    assert (deg_dir / "FSH_vs_control.csv").exists()


def test_cli_skip_deg_reuses_existing_deg_dir(tmp_path):
    graph_path = tmp_path / "graph.pkl"
    with graph_path.open("wb") as fh:
        pickle.dump(_graph(), fh)

    deg_dir = tmp_path / "deg"
    deg_dir.mkdir()
    pd.DataFrame({
        "names": ["TGT"],
        "logfoldchanges": [1.0],
        "pvals": [0.001],
        "pvals_adj": [0.01],
    }).to_csv(deg_dir / "FSH_vs_control.csv", index=False)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
run:
  output_dir: {tmp_path / "out"}
graph:
  pkl_path: {graph_path}
input:
  kind: raw_perturbseq
  adata_path: {tmp_path / "not_used.h5ad"}
  deg_output_dir: {deg_dir}
intermediates:
  mode: full_hgnc
evidence:
  enabled: false
plots:
  enabled: false
""",
        encoding="utf-8",
    )

    out = io.StringIO()
    with redirect_stdout(out):
        status = cli_main(["--config", str(cfg_path), "--skip-deg"])

    assert status == 0
    assert (tmp_path / "out" / "1hop_results.csv").exists()


def test_cli_validate_only_accepts_existing_paths(tmp_path):
    graph_path = tmp_path / "graph.pkl"
    with graph_path.open("wb") as fh:
        pickle.dump(_graph(), fh)

    targets_path = tmp_path / "targets.csv"
    targets_path.write_text("names,pval\nTGT,0.001\n", encoding="utf-8")

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
graph:
  pkl_path: {graph_path}
sources:
  values: [FSH]
targets:
  mode: table
  path: {targets_path}
  gene_column: names
  pval_column: pval
evidence:
  enabled: false
plots:
  enabled: false
""",
        encoding="utf-8",
    )

    out = io.StringIO()
    with redirect_stdout(out):
        status = cli_main(["--config", str(cfg_path), "--validate-only"])

    assert status == 0
    assert "Config OK" in out.getvalue()


def test_cli_validate_only_rejects_missing_paths(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
graph:
  pkl_path: {tmp_path / "missing.pkl"}
sources:
  values: [FSH]
targets:
  mode: table
  path: {tmp_path / "missing_targets.csv"}
""",
        encoding="utf-8",
    )

    assert cli_main(["--config", str(cfg_path), "--validate-only"]) == 1
