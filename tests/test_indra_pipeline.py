from pathlib import Path

import networkx as nx
import pandas as pd

from indra_perturbseq.indra_pipeline.config import load_config
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
