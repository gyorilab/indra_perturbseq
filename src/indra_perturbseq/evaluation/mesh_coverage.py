"""MeSH coverage analysis for INDRA causal-path statements.
Computes what percentage of INDRA statements (deduplicated by.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _has_pmids(pmid_str: str | None) -> bool:
    if not pmid_str or str(pmid_str).strip().lower() in ("", "nan"):
        return False
    return any(
        tok.strip().isdigit()
        for tok in re.split(r"[;,\s]+", str(pmid_str))
    )


def _has_mesh(mesh_str: str | None) -> bool:
    if not mesh_str or str(mesh_str).strip().lower() in ("", "nan"):
        return False
    return bool(str(mesh_str).strip())


def load_gene_set(path: str, gene_column: str = "gene") -> set[str]:
    """Load a set of upper-cased gene symbols from a CSV.

    Parameters
    ----------
    path : str
        CSV path.
    gene_column : str
        Column name.

    Returns
    -------
    set[str]
    """
    genes: set[str] = set()
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            g = row.get(gene_column, "").strip().upper()
            if g:
                genes.add(g)
    logger.info("Loaded gene set: %d genes from %s", len(genes), path)
    return genes


def _all_in_set(gene_list: list[str], gene_set: set[str]) -> bool:
    return all(g.upper() in gene_set for g in gene_list if g and g.strip())


# ------------------------------------------------------------------
# Statement accumulator
# ------------------------------------------------------------------

class StatementAccumulator:
    """Deduplicated accumulator for (from_gene, stmt_type, to_gene) triples.

    Parameters
    ----------
    None

    Attributes
    ----------
    statements : dict
        Mapping from statement key to ``{"pmids": bool, "mesh": bool}``.
    """

    def __init__(self) -> None:
        self.statements: dict[tuple[str, str, str], dict[str, bool]] = {}

    def register(
        self,
        from_gene: str,
        stmt_type: str,
        to_gene: str,
        pmid_str: str | None,
        mesh_str: str | None,
    ) -> None:
        """Register a statement occurrence, OR-merging PMID/MeSH flags.

        Parameters
        ----------
        from_gene, stmt_type, to_gene : str
            Statement triple components.
        pmid_str, mesh_str : str or None
            Raw PMID and MeSH annotation strings.
        """
        key = (from_gene.strip().upper(), stmt_type.strip(),
               to_gene.strip().upper())
        hp = _has_pmids(pmid_str)
        hm = _has_mesh(mesh_str)
        if key in self.statements:
            self.statements[key]["pmids"] |= hp
            self.statements[key]["mesh"] |= hm
        else:
            self.statements[key] = {"pmids": hp, "mesh": hm}

    def summary(self) -> dict[str, int | float]:
        """Return aggregate counts and percentages.

        Returns
        -------
        dict
            Keys: ``n_total``, ``n_pmids``, ``n_no_pmids``, ``n_hit``,
            ``n_miss``, ``pct_pmids``, ``pct_mesh``.
        """
        n_total = len(self.statements)
        n_pmids = sum(1 for v in self.statements.values() if v["pmids"])
        n_hit = sum(
            1 for v in self.statements.values() if v["pmids"] and v["mesh"]
        )
        return {
            "n_total": n_total,
            "n_pmids": n_pmids,
            "n_no_pmids": n_total - n_pmids,
            "n_hit": n_hit,
            "n_miss": n_pmids - n_hit,
            "pct_pmids": 100.0 * n_pmids / n_total if n_total else 0.0,
            "pct_mesh": 100.0 * n_hit / n_pmids if n_pmids else 0.0,
        }


def _register_into(
    acc: StatementAccumulator,
    hop_acc: StatementAccumulator,
    from_gene: str,
    stmt_type: str,
    to_gene: str,
    pmid_str: str | None,
    mesh_str: str | None,
) -> None:
    acc.register(from_gene, stmt_type, to_gene, pmid_str, mesh_str)
    hop_acc.register(from_gene, stmt_type, to_gene, pmid_str, mesh_str)


# ------------------------------------------------------------------
# Per-hop file processors
# ------------------------------------------------------------------

def process_1hop(
    filepath: str,
    gene_set: set[str],
    global_acc: StatementAccumulator,
) -> StatementAccumulator:
    """Process a 1-hop CSV and accumulate statements.

    Parameters
    ----------
    filepath : str
    gene_set : set[str]
        RNA-present gene universe (upper-cased).
    global_acc : StatementAccumulator
        Global accumulator across all hops.

    Returns
    -------
    StatementAccumulator
        Hop-local accumulator.
    """
    hop_acc = StatementAccumulator()
    total = rna_kept = 0
    with open(filepath, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            total += 1
            src = row.get("source", "").strip()
            tgt = row.get("target", "").strip()
            if not _all_in_set([src, tgt], gene_set):
                continue
            rna_kept += 1
            _register_into(
                global_acc, hop_acc,
                src, row.get("stmt_type", "").strip(), tgt,
                row.get("pmids", ""),
                row.get("Annotated MeSH terms", ""),
            )
    logger.info("1-hop: %d rows -> %d RNA-present (%.1f%% kept)",
                total, rna_kept,
                100.0 * rna_kept / total if total else 0.0)
    return hop_acc


def process_2hop(
    filepath: str,
    gene_set: set[str],
    global_acc: StatementAccumulator,
) -> tuple[StatementAccumulator, StatementAccumulator]:
    """Process a 2-hop CSV and accumulate statements per hop position.

    Parameters
    ----------
    filepath : str
    gene_set : set[str]
    global_acc : StatementAccumulator

    Returns
    -------
    tuple[StatementAccumulator, StatementAccumulator]
        (hop1_acc, hop2_acc)
    """
    h1, h2 = StatementAccumulator(), StatementAccumulator()
    total = rna_kept = 0
    with open(filepath, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            total += 1
            src = row.get("source", "").strip()
            mid = row.get("intermediate", "").strip()
            tgt = row.get("target", "").strip()
            if not _all_in_set([src, mid, tgt], gene_set):
                continue
            rna_kept += 1
            for a, b, st_col, p_col, m_col, hacc in [
                (src, mid, "stmt_type_1", "pmids_hop1",
                 "Annotated MeSH terms hop1", h1),
                (mid, tgt, "stmt_type_2", "pmids_hop2",
                 "Annotated MeSH terms hop2", h2),
            ]:
                _register_into(
                    global_acc, hacc,
                    a, row.get(st_col, "").strip(), b,
                    row.get(p_col, ""), row.get(m_col, ""),
                )
    logger.info("2-hop: %d rows -> %d RNA-present (%.1f%% kept)",
                total, rna_kept,
                100.0 * rna_kept / total if total else 0.0)
    return h1, h2


def process_3hop(
    filepath: str,
    gene_set: set[str],
    global_acc: StatementAccumulator,
) -> tuple[StatementAccumulator, StatementAccumulator, StatementAccumulator]:
    """Process a 3-hop CSV and accumulate statements per hop position.

    Parameters
    ----------
    filepath : str
    gene_set : set[str]
    global_acc : StatementAccumulator

    Returns
    -------
    tuple[StatementAccumulator, StatementAccumulator, StatementAccumulator]
        (hop1_acc, hop2_acc, hop3_acc)
    """
    h1, h2, h3 = (
        StatementAccumulator(), StatementAccumulator(), StatementAccumulator(),
    )
    total = rna_kept = 0
    with open(filepath, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            total += 1
            src = row.get("source", "").strip()
            mid1 = row.get("intermediate_1", "").strip()
            mid2 = row.get("intermediate_2", "").strip()
            tgt = row.get("target", "").strip()
            if not _all_in_set([src, mid1, mid2, tgt], gene_set):
                continue
            rna_kept += 1
            for a, b, st_col, p_col, m_col, hacc in [
                (src, mid1, "stmt_type_1", "pmids_hop1",
                 "Annotated MeSH terms (hop1)", h1),
                (mid1, mid2, "stmt_type_2", "pmids_hop2",
                 "Annotated MeSH terms (hop2)", h2),
                (mid2, tgt, "stmt_type_3", "pmids_hop3",
                 "Annotated MeSH terms (hop3)", h3),
            ]:
                _register_into(
                    global_acc, hacc,
                    a, row.get(st_col, "").strip(), b,
                    row.get(p_col, ""), row.get(m_col, ""),
                )
    logger.info("3-hop: %d rows -> %d RNA-present (%.1f%% kept)",
                total, rna_kept,
                100.0 * rna_kept / total if total else 0.0)
    return h1, h2, h3


def _log_hop_stats(label: str, acc: StatementAccumulator) -> None:
    s = acc.summary()
    logger.info(
        "%s: %d stmts | with PMIDs: %d (%.1f%%) | "
        "MeSH of PMID-backed: %d (%.1f%%)",
        label, s["n_total"], s["n_pmids"], s["pct_pmids"],
        s["n_hit"], s["pct_mesh"],
    )


def _merge_accumulators(
    *accs: StatementAccumulator,
) -> StatementAccumulator:
    merged = StatementAccumulator()
    for acc in accs:
        for key, val in acc.statements.items():
            if key in merged.statements:
                merged.statements[key]["pmids"] |= val["pmids"]
                merged.statements[key]["mesh"] |= val["mesh"]
            else:
                merged.statements[key] = dict(val)
    return merged


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="MeSH coverage analysis for INDRA causal-path statements.",
    )
    ap.add_argument("--gene-list-csv", required=True,
                    help="RNA-present gene list CSV")
    ap.add_argument("--gene-column", default="gene")
    ap.add_argument("--hop1-csv", default="",
                    help="1-hop causal-path CSV")
    ap.add_argument("--hop2-csv", default="",
                    help="2-hop causal-path CSV")
    ap.add_argument("--hop3-csv", default="",
                    help="3-hop causal-path CSV")
    args = ap.parse_args(argv)

    gene_set = load_gene_set(args.gene_list_csv, gene_column=args.gene_column)
    global_acc = StatementAccumulator()

    if args.hop1_csv:
        h1 = process_1hop(args.hop1_csv, gene_set, global_acc)
        _log_hop_stats("1-hop", h1)

    if args.hop2_csv:
        h2_1, h2_2 = process_2hop(args.hop2_csv, gene_set, global_acc)
        _log_hop_stats("2-hop (hop 1)", h2_1)
        _log_hop_stats("2-hop (hop 2)", h2_2)
        _log_hop_stats("2-hop (combined)", _merge_accumulators(h2_1, h2_2))

    if args.hop3_csv:
        h3_1, h3_2, h3_3 = process_3hop(args.hop3_csv, gene_set, global_acc)
        _log_hop_stats("3-hop (hop 1)", h3_1)
        _log_hop_stats("3-hop (hop 2)", h3_2)
        _log_hop_stats("3-hop (hop 3)", h3_3)
        _log_hop_stats("3-hop (combined)",
                        _merge_accumulators(h3_1, h3_2, h3_3))

    gs = global_acc.summary()
    logger.info(
        "GLOBAL: %d unique stmts | with PMIDs: %d (%.1f%%) | "
        "MeSH of PMID-backed: %d / %d (%.1f%%)",
        gs["n_total"], gs["n_pmids"], gs["pct_pmids"],
        gs["n_hit"], gs["n_pmids"], gs["pct_mesh"],
    )


if __name__ == "__main__":
    main()
