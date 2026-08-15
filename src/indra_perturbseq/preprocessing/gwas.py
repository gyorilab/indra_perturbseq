"""Combine, deduplicate, and segregate GWAS pathway datasets.

Provides two subcommands:

``dedup``
    Combine two pathway CSVs, apply strict hop-2 quality filtering,
    score rows, and deduplicate on ``(source, target)``.  Outputs a
    combined CSV plus GWAS-as-source and GWAS-as-target subsets.

``segregate``
    Split a pathway CSV by membership in predefined gene sets (8-gene,
    17-gene, residual), annotating which genes from each set appear in
    the path.
"""
from __future__ import annotations

import argparse
import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gene sets
# ---------------------------------------------------------------------------
GWAS_GENES: set[str] = {
    "ARHGEF26", "BCAR1", "BMP1", "CALCRL", "CCM2", "CDKN1A", "CDKN2B",
    "CFDP1", "COL4A1", "COL4A2", "EDN1", "EXOC3L2", "FBN2", "FGD6",
    "FLT1", "FURIN", "GDPD5", "GGT5", "GOSR2", "IBTK", "JCAD", "LAMB2",
    "LOX", "MORF4L1", "N4BP2L2", "NOS3", "PALLD", "PECAM1", "PGF",
    "PLPP3", "PRDM16", "PREX1", "PRKAR1A", "SCUBE1", "SERPINH1",
    "SH3PXD2A", "SLK", "SMAD3", "SPRY4", "SVIL", "SWAP70", "TFPI",
    "TLNRD1", "TSPAN14", "ZEB2",
}

GENES_8: set[str] = {
    "PRDM16", "PLPP3", "NOS3", "JCAD", "FLT1", "EDN1", "PECAM1",
    "ARHGEF26",
}

GENES_17: set[str] = {
    "CALCRL", "CCM2", "CDKN1A", "EXOC3L2", "GDPD5", "GGT5", "IBTK",
    "N4BP2L2", "PREX1", "PRKAR1A", "SCUBE1", "SLK", "SPRY4", "SVIL",
    "TFPI", "TLNRD1", "TSPAN14",
}

GENES_420: set[str] = {
    "AAK1", "ABCG5", "ABCG8", "ABHD2", "AC003986.6", "AC105384.1",
    "ACTA2", "ACTN4", "ACTRT2", "ACVR2A", "ACVRL1", "ADAM19", "ADAMTS3",
    "ADAMTS7", "AEBP1", "AFAP1L2", "AFF4", "AGAP5", "AHI1", "AIDA",
    "AKAP12", "AL592148.3", "ANGPTL4", "ANKRD13B", "ANTXR2", "ANXA11",
    "AP000318.2", "AP002989.1", "APOA1", "APOA5", "APOB", "APOE", "APOM",
    "ARAP3", "ARHGAP21", "ARHGAP26", "ARHGAP42", "ARHGEF12", "ARHGEF26",
    "ARNT", "ARNTL", "ARVCF", "ATF6B", "ATP1B1", "ATP2B1", "ATXN7L2",
    "AXL", "B4GALT5", "BACH1", "BAG6", "BASP1", "BCAR1", "BCAS3",
    "BCKDHA", "BMP1", "BMPR1B", "C1QTNF1", "C4A", "CABIN1", "CALCRL",
    "CAND1", "CARF", "CAV1", "CBS", "CBX5", "CCDC3", "CCDC97", "CCM2",
    "CD36", "CDC123", "CDH13", "CDK8", "CDKN1A", "CDKN2A", "CDKN2B",
    "CDKN2BAS", "CEL", "CELSR2", "CETP", "CFDP1", "CHRNB4", "CLDN5",
    "COBLL1", "COL4A1", "COL4A2", "COL6A3", "COPRS", "CORO6", "CRYAB",
    "CSF1", "CTD-3253I12.1", "CTSH", "CXCL12", "CYP17A1", "CYP46A1",
    "DAAM2", "DAB2IP", "DCUN1D3", "DENND4C", "DHX36", "DHX38", "DMAC2",
    "DOCK5", "DOCK9", "DPYD", "DST", "EDN1", "EDNRA", "EFCAB13",
    "EGFLAM", "EHBP1L1", "EIF2B2", "EPAS1", "ESYT3", "EXOC3L2", "EZR",
    "F10", "FAM114A1", "FAM117B", "FAM177B", "FBN2", "FCHO1", "FER",
    "FES", "FGD5", "FGD6", "FGF5", "FHL3", "FHL5", "FIGN", "FLOT1",
    "FLT1", "FN1", "FNDC3B", "FOCAD", "FOXC1", "FURIN", "GAS8", "GATA6",
    "GATAD2A", "GDPD5", "GEM", "GFOD1", "GGCX", "GGT5", "GIGYF1",
    "GIGYF2", "GNA12", "GNAS", "GOSR2", "GRK4", "GUCY1A1", "GUCY1A3",
    "HDAC9", "HDGFL1", "HEY2", "HHAT", "HHIPL1", "HIPK2", "HIVEP2",
    "HLA-C", "HLA-DQB1", "HMHB1", "HNF1A", "HOMER3", "HSD17B1",
    "HSD17B12", "HTRA1", "HTT", "IBTK", "ICA1L", "IGFBP7", "IL6R",
    "IL6ST", "ILK", "INPP5B", "IRS1", "ITGA1", "ITGB3", "ITIH4", "JCAD",
    "JUN", "KANK1", "KCNE2", "KCNK5", "KCTD8", "KIAA0040", "KLF2",
    "KPTN", "LAMA4", "LAMB1", "LAMB2", "LDLR", "LIMS2", "LINC00189",
    "LINC00310", "LINC01312", "LIPA", "LIPC", "LMAN1", "LMOD1", "LOX",
    "LOXL4", "LPA", "LPIN3", "LPL", "LRP1", "LRRC10B", "LSM2",
    "MAD2L1", "MAGI3", "MAN2A2", "MAP1S", "MAP3K1", "MAP3K3", "MAP3K7CL",
    "MAP9", "MAT2A", "MC4R", "MCAM", "MCF2L", "MECOM", "MED1", "MESD",
    "MFGE8", "MGP", "MIA3", "MLH3", "MORF4L1", "MRAS", "MRPS6", "MRVI1",
    "MSH5", "MTAP", "MTUS1", "MYH11", "MYL2", "MYLK", "MYO9B",
    "N4BP2L2", "NBEAL1", "NCOA6", "NEK8", "NEK9", "NF2", "NFIB", "NGF",
    "NIPBL", "NISCH", "NLRC4", "NME9", "NOB1", "NOS3", "NOTCH1", "NR2F2",
    "NR3C1", "NRP1", "NT5C2", "NUPR1", "OPRL1", "PAFAH1B1", "PALLD",
    "PARP12", "PCNX3", "PCSK9", "PDE1A", "PDE1C", "PDE3A", "PDE5A",
    "PDGFD", "PDGFRA", "PECAM1", "PGF", "PHACTR1", "PHB", "PHETA1",
    "PHLPP2", "PID1", "PLCE1", "PLCG1", "PLCG2", "PLG", "PLPP3", "PLTP",
    "PMAIP1", "PNPLA3", "POLK", "PPAP2B", "PPARD", "PPP1R12A", "PRDM16",
    "PREX1", "PRIM2", "PRKAR1A", "PRKCE", "PRL", "PROCR", "PRRT1",
    "PSMA4", "PSMA5", "PSORS1C1", "PSRC1", "R3HCC1L", "R3HDM1", "RAC1",
    "RASGEF1B", "RCOR3", "RDX", "RELA", "REST", "RGS19", "RHOB",
    "RIIAD1", "RP1-257A7.4", "RP1-257A7.5", "RP11-298D21.1",
    "RP11-298D21.3", "RP11-543N12.1", "RP11-588K22.2", "RP11-752L20.5",
    "RP11-755F10.1", "RRBP1", "RUNX1", "SARS", "SCAMP1-AS1", "SCARB1",
    "SCUBE1", "SDCCAG3", "SEMA5A", "SEPT11", "SERPINA1", "SERPINH1",
    "SH3PXD2A", "SHROOM3", "SKI", "SKIV2L", "SLC18A1", "SLC22A1",
    "SLC22A3", "SLC22A4", "SLC22A5", "SLC2A12", "SLC5A3", "SLK", "SMAD1",
    "SMAD3", "SMAD7", "SMG6", "SMTN", "SNF8", "SORT1", "SPC24", "SPRY4",
    "SREBF1", "ST3GAL4", "ST5", "STAG1", "STARD13", "STAT3", "STX4",
    "SUMO1", "SUMO2", "SVIL", "SWAP70", "TAF1A", "TARID", "TBC1D7",
    "TBX2", "TBX20", "TBX3", "TCF21", "TCF7L2", "TENT5A", "TFAP2B",
    "TFPI", "TGFB1", "THOC5", "TIE1", "TIMP3", "TIPARP", "TLNRD1",
    "TMEM133", "TNFAIP8", "TNKS", "TNS1", "TRIB1", "TSPAN11", "TSPAN14",
    "TTC32", "TWIST1", "TWISTNB", "TXNRD3", "UBC", "UBE2H", "UFL1",
    "UMPS", "UNC119B", "USP34", "VAMP5", "VEGFA", "VWF", "WASF1",
    "WASF2", "WIPI1", "WT1", "WWOX", "WWP2", "ZBTB38", "ZC3HC1", "ZEB2",
    "ZFHX3", "ZFPM2", "ZNF100", "ZNF335", "ZNF43", "ZNF462", "ZNF532",
    "ZNF589", "ZNF652", "ZNF831",
}

_URL_RE = re.compile(
    r"^https://db\.indra\.bio/statements/from_hash/-?\d+\?format=html$",
)


# ---------------------------------------------------------------------------
# Quality helpers
# ---------------------------------------------------------------------------
def _is_nonempty(x: object) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and pd.isna(x):
        return False
    s = str(x).strip()
    return s != "" and s.lower() not in {"nan", "none"}


def _evidence_ok_strict(text: object) -> bool:
    if not _is_nonempty(text):
        return False
    s = str(text).strip()
    return bool(re.search(r"(^|\n)\s*1\)\s+", s)) or s.startswith("Evidence from:")


def _is_valid_indra_url(url: object) -> bool:
    if not _is_nonempty(url):
        return False
    return bool(_URL_RE.fullmatch(str(url).strip()))


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
        return default if pd.isna(v) else v
    except (TypeError, ValueError):
        return default


def _compute_gwas_in_path(row: pd.Series) -> str:
    hits = []
    for col in ("source", "intermediate", "target"):
        val = str(row.get(col, "")).strip()
        if val in GWAS_GENES:
            hits.append(val)
    return ", ".join(sorted(set(hits)))


# ---------------------------------------------------------------------------
# Dedup subcommand
# ---------------------------------------------------------------------------
def _ensure_schema(df: pd.DataFrame, template_cols: list[str]) -> pd.DataFrame:
    """Ensure *df* has at least the columns present in *template_cols*."""
    df = df.copy()
    df = df.drop(columns=["pval_col_used"], errors="ignore")

    for col in template_cols:
        if col not in df.columns:
            df[col] = ""

    if "logfoldchange" in df.columns:
        df["logfoldchange"] = pd.to_numeric(df["logfoldchange"], errors="coerce")

    if "GWAS_genes_in_path" not in df.columns:
        df["GWAS_genes_in_path"] = ""
    missing = ~df["GWAS_genes_in_path"].apply(_is_nonempty)
    if missing.any():
        df.loc[missing, "GWAS_genes_in_path"] = df.loc[missing].apply(
            _compute_gwas_in_path, axis=1,
        )
    return df


def _hop2_strict_ok(row: pd.Series, require_pmids: bool) -> bool:
    if not _is_nonempty(row.get("hop2_hash", "")):
        return False
    if not _evidence_ok_strict(row.get("evidence_text_hop2", "")):
        return False
    if require_pmids and not _is_nonempty(row.get("pmids_hop2", "")):
        return False
    return True


def _row_quality_score(row: pd.Series) -> float:
    url1 = _is_valid_indra_url(row.get("hop1_indra_url", ""))
    url2 = _is_valid_indra_url(row.get("hop2_indra_url", ""))

    ev1_ok = _evidence_ok_strict(row.get("evidence_text_hop1", ""))
    ev2_ok = _evidence_ok_strict(row.get("evidence_text_hop2", ""))

    pm1 = _is_nonempty(row.get("pmids_hop1", ""))
    pm2 = _is_nonempty(row.get("pmids_hop2", ""))

    completeness = sum(1 for c in row.index if _is_nonempty(row.get(c, "")))
    abs_lfc = abs(_safe_float(row.get("logfoldchange", 0.0)))

    score = 0.0
    score += 20.0 * (url1 + url2)
    score += 10.0 * (ev1_ok + ev2_ok)
    score += 4.0 * (pm1 + pm2)
    score += 0.5 * completeness
    score += 1.0 * abs_lfc
    return score


def deduplicate_gwas(
    primary_csv: str,
    secondary_csv: str,
    *,
    require_hop2_pmids: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combine two GWAS pathway CSVs and deduplicate on ``(source, target)``.

    Parameters
    ----------
    primary_csv
        Path to the primary (GWAS-URL-enriched) CSV.
    secondary_csv
        Path to the secondary (one-stop) CSV.
    require_hop2_pmids
        If ``True``, rows without hop-2 PMIDs are dropped.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(combined_deduped, gwas_sources, gwas_targets)``.
    """
    df_a = pd.read_csv(primary_csv, low_memory=False)
    df_b = pd.read_csv(secondary_csv, low_memory=False)

    df_a = df_a.drop(columns=["pval_col_used"], errors="ignore")
    df_b = df_b.drop(columns=["pval_col_used"], errors="ignore")
    a_cols = list(df_a.columns)

    logger.info("Primary CSV: %d rows, %d cols", len(df_a), len(df_a.columns))
    logger.info("Secondary CSV: %d rows, %d cols", len(df_b), len(df_b.columns))

    df_a = _ensure_schema(df_a, a_cols)
    df_b = _ensure_schema(df_b, a_cols)

    combined = pd.concat([df_a, df_b], ignore_index=True)
    before = len(combined)

    combined["_hop2_ok"] = combined.apply(
        lambda r: _hop2_strict_ok(r, require_hop2_pmids), axis=1,
    )
    combined = combined[combined["_hop2_ok"]].copy()
    logger.info(
        "Strict hop2 filter: %d -> %d rows (%.1f%%)",
        before, len(combined), len(combined) / max(before, 1) * 100,
    )

    if combined.empty:
        raise RuntimeError(
            "No rows survived strict hop2 filtering. "
            "Consider disabling --require-hop2-pmids."
        )

    combined["_score"] = combined.apply(_row_quality_score, axis=1)
    combined["_abs_lfc"] = combined["logfoldchange"].apply(
        lambda x: abs(_safe_float(x)),
    )
    combined = combined.sort_values(
        ["_score", "_abs_lfc"], ascending=[False, False], kind="mergesort",
    )
    combined = combined.drop_duplicates(subset=["source", "target"], keep="first")
    combined = combined.drop(columns=["_hop2_ok", "_score", "_abs_lfc"], errors="ignore")

    keep = [c for c in a_cols if c in combined.columns]
    combined = combined[keep].copy()
    logger.info("Deduplicated to %d unique (source, target) pairs", len(combined))

    gwas_src = combined[combined["source"].astype(str).isin(GWAS_GENES)].copy()
    gwas_tgt = combined[combined["target"].astype(str).isin(GWAS_GENES)].copy()
    logger.info("GWAS-as-source: %d rows", len(gwas_src))
    logger.info("GWAS-as-target: %d rows", len(gwas_tgt))

    return combined, gwas_src, gwas_tgt


# ---------------------------------------------------------------------------
# Segregate subcommand
# ---------------------------------------------------------------------------
def _genes_in_path(row: pd.Series, gene_set: set[str]) -> str:
    hits: list[str] = []
    for col in ("source", "intermediate", "target"):
        val = str(row.get(col, "")).strip()
        if val in gene_set:
            hits.append(val)
    return ", ".join(sorted(set(hits)))


def segregate_gwas(
    input_csv: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a GWAS CSV by 8-gene, 17-gene, and residual sets.

    Parameters
    ----------
    input_csv
        Path to the input GWAS pathway CSV.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(df_8gene, df_17gene, df_residual)``.
    """
    df = pd.read_csv(input_csv, low_memory=False)
    logger.info("Loaded %d rows from %s", len(df), input_csv)

    mask_8 = (
        df["source"].isin(GENES_8)
        | df["intermediate"].isin(GENES_8)
        | df["target"].isin(GENES_8)
    )
    mask_17 = (
        df["source"].isin(GENES_17)
        | df["intermediate"].isin(GENES_17)
        | df["target"].isin(GENES_17)
    )
    mask_residual = ~(mask_8 | mask_17)

    df_8 = df[mask_8].copy()
    df_8["Genes_in_path_8gene_set"] = df_8.apply(
        lambda r: _genes_in_path(r, GENES_8), axis=1,
    )
    df_8["Genes_in_path_420gene_set"] = df_8.apply(
        lambda r: _genes_in_path(r, GENES_420), axis=1,
    )

    df_17 = df[mask_17].copy()
    df_17["Genes_in_path_17gene_set"] = df_17.apply(
        lambda r: _genes_in_path(r, GENES_17), axis=1,
    )
    df_17["Genes_in_path_420gene_set"] = df_17.apply(
        lambda r: _genes_in_path(r, GENES_420), axis=1,
    )

    df_res = df[mask_residual].copy()
    df_res["Genes_in_path_420gene_set"] = df_res.apply(
        lambda r: _genes_in_path(r, GENES_420), axis=1,
    )

    logger.info(
        "Segregation: 8-gene=%d, 17-gene=%d, residual=%d",
        len(df_8), len(df_17), len(df_res),
    )
    return df_8, df_17, df_res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    """CLI entry point with ``dedup`` and ``segregate`` subcommands."""
    parser = argparse.ArgumentParser(
        description="Combine, deduplicate, and segregate GWAS pathway datasets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- dedup --
    dd = sub.add_parser("dedup", help="Combine + strict deduplicate two GWAS CSVs.")
    dd.add_argument(
        "--primary-csv", required=True,
        help="Primary GWAS CSV (e.g. gwas_endothelial_paths_url_enriched.csv).",
    )
    dd.add_argument(
        "--secondary-csv", required=True,
        help="Secondary CSV (e.g. one-stop 2hop output).",
    )
    dd.add_argument("--output-combined", required=True, help="Combined deduped output.")
    dd.add_argument("--output-gwas-sources", required=True, help="GWAS-as-source subset.")
    dd.add_argument("--output-gwas-targets", required=True, help="GWAS-as-target subset.")
    dd.add_argument(
        "--require-hop2-pmids", action="store_true",
        help="Also require pmids_hop2 to be non-empty.",
    )

    # -- segregate --
    seg = sub.add_parser("segregate", help="Split GWAS CSV by gene groups.")
    seg.add_argument("--input", required=True, help="Input GWAS CSV.")
    seg.add_argument("--output-8gene", required=True, help="8-gene subset output.")
    seg.add_argument("--output-17gene", required=True, help="17-gene subset output.")
    seg.add_argument("--output-residual", required=True, help="Residual subset output.")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "dedup":
        combined, gwas_src, gwas_tgt = deduplicate_gwas(
            primary_csv=args.primary_csv,
            secondary_csv=args.secondary_csv,
            require_hop2_pmids=args.require_hop2_pmids,
        )
        combined.to_csv(args.output_combined, index=False)
        gwas_src.to_csv(args.output_gwas_sources, index=False)
        gwas_tgt.to_csv(args.output_gwas_targets, index=False)
        logger.info("Saved dedup outputs")

    elif args.command == "segregate":
        df_8, df_17, df_res = segregate_gwas(args.input)
        df_8.to_csv(args.output_8gene, index=False)
        df_17.to_csv(args.output_17gene, index=False)
        df_res.to_csv(args.output_residual, index=False)
        logger.info("Saved segregation outputs")


if __name__ == "__main__":
    main()
