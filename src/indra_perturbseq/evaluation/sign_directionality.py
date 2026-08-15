"""Annotate pathway CSVs with directional consistency under knockdown logic.

Rules implemented:
- Any pathway with a non-directional statement type is ``Undeterminable``.
- Any directional-only pathway with at least one ``msigdb`` evidence source is
  ``Undeterminable``.
- For the remaining directional-only pathways:
  - ``IncreaseAmount`` / ``Activation`` => +1
  - ``DecreaseAmount`` / ``Inhibition`` => -1
  - pathway sign = product of all hop signs
  - a pathway is ``Consistent`` when:
    - product == -1 and logfoldchange > 0
    - or product == +1 and logfoldchange < 0
  - otherwise it is ``Inconsistent``.

If a CSV does not carry evidence text columns, this module attempts to recover
``msigdb`` provenance from other local evidence-bearing CSVs using statement
hashes. If that provenance cannot be resolved locally, the pathway is marked
``Undeterminable`` conservatively.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

logger = logging.getLogger(__name__)

POSITIVE_STMT_TYPES = frozenset({"IncreaseAmount", "Activation"})
NEGATIVE_STMT_TYPES = frozenset({"DecreaseAmount", "Inhibition"})
ALLOWED_STMT_TYPES = POSITIVE_STMT_TYPES | NEGATIVE_STMT_TYPES
MSIGDB_TERMS = ("msigdb", "misgdb")
DIRECTIONAL_CONSISTENCY_COL = "directional consistency"


def _norm_gene(value: object) -> str | None:
    s = str(value).strip().upper()
    return s or None


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        x = float(s)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def _is_msigdb_text(value: object) -> bool:
    text = str(value or "").lower()
    return any(term in text for term in MSIGDB_TERMS)


def _find_stmt_cols(fieldnames: list[str]) -> list[str]:
    return [name for name in fieldnames if name.startswith("stmt_type")]


def _find_evidence_cols(fieldnames: list[str]) -> list[str]:
    return [name for name in fieldnames if name.startswith("evidence_text_hop")]


def _find_hash_cols(fieldnames: list[str]) -> list[str]:
    cols: list[str] = []
    if "stmt_hash" in fieldnames:
        cols.append("stmt_hash")
    cols.extend(
        name
        for name in fieldnames
        if name.startswith("hop") and name.endswith("_hash")
    )
    return cols


def _sign_for_stmt(stmt_type: str) -> int:
    if stmt_type in POSITIVE_STMT_TYPES:
        return 1
    return -1


def _classify_pair(
    any_consistent: bool,
    any_undeterminable: bool,
    any_inconsistent: bool,
) -> str:
    if any_consistent:
        return "Consistent"
    if any_undeterminable:
        return "Undeterminable"
    if any_inconsistent:
        return "Inconsistent"
    return "Undeterminable"


def _empty_row_counter() -> dict[str, int]:
    return {
        "total_rows": 0,
        "determinable_rows": 0,
        "consistent_rows": 0,
        "inconsistent_rows": 0,
        "undeterminable_rows": 0,
        "undeterminable_nondirectional_rows": 0,
        "undeterminable_msigdb_rows": 0,
        "undeterminable_unresolved_provenance_rows": 0,
        "undeterminable_missing_logfc_rows": 0,
    }


def _finalize_row_counter(counter: dict[str, int]) -> dict[str, float | int]:
    determinable = counter["determinable_rows"]
    if determinable > 0:
        pct_consistent = counter["consistent_rows"] / determinable * 100.0
        pct_inconsistent = counter["inconsistent_rows"] / determinable * 100.0
    else:
        pct_consistent = float("nan")
        pct_inconsistent = float("nan")
    return {
        **counter,
        "pct_consistent_of_determinable": pct_consistent,
        "pct_inconsistent_of_determinable": pct_inconsistent,
    }


def _summarize_pairs(pair_flags: dict[tuple[str, str], dict[str, bool]]) -> dict[str, float | int]:
    total = len(pair_flags)
    consistent = 0
    inconsistent = 0
    undeterminable = 0
    for flags in pair_flags.values():
        cls = _classify_pair(
            flags["any_consistent"],
            flags["any_undeterminable"],
            flags["any_inconsistent"],
        )
        if cls == "Consistent":
            consistent += 1
        elif cls == "Inconsistent":
            inconsistent += 1
        else:
            undeterminable += 1

    determinable = consistent + inconsistent
    if determinable > 0:
        pct_consistent = consistent / determinable * 100.0
        pct_inconsistent = inconsistent / determinable * 100.0
    else:
        pct_consistent = float("nan")
        pct_inconsistent = float("nan")

    return {
        "total_pairs": total,
        "determinable_pairs": determinable,
        "consistent_pairs": consistent,
        "inconsistent_pairs": inconsistent,
        "undeterminable_pairs": undeterminable,
        "pct_consistent_of_determinable_pairs": pct_consistent,
        "pct_inconsistent_of_determinable_pairs": pct_inconsistent,
    }


def _build_hash_msigdb_lookup(search_root: Path) -> dict[str, bool]:
    lookup: dict[str, bool] = {}
    csv.field_size_limit(sys.maxsize)

    for path in search_root.rglob("*.csv"):
        try:
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                fieldnames = reader.fieldnames or []
                evidence_cols = _find_evidence_cols(fieldnames)
                if not evidence_cols:
                    continue

                for row in reader:
                    if "stmt_hash" in fieldnames and "evidence_text_hop1" in fieldnames:
                        stmt_hash = str(row.get("stmt_hash", "")).strip()
                        if stmt_hash:
                            lookup[stmt_hash] = lookup.get(stmt_hash, False) or _is_msigdb_text(
                                row.get("evidence_text_hop1")
                            )

                    for hop in range(1, 5):
                        hash_col = f"hop{hop}_hash"
                        evidence_col = f"evidence_text_hop{hop}"
                        if hash_col not in fieldnames or evidence_col not in fieldnames:
                            continue
                        stmt_hash = str(row.get(hash_col, "")).strip()
                        if stmt_hash:
                            lookup[stmt_hash] = lookup.get(stmt_hash, False) or _is_msigdb_text(
                                row.get(evidence_col)
                            )
        except Exception as exc:
            logger.debug("Skipping %s while building provenance lookup: %s", path, exc)

    logger.info("Recovered msigdb provenance for %s statement hashes", f"{len(lookup):,}")
    return lookup


def _classify_row(
    row: dict[str, str],
    fieldnames: list[str],
    hash_msigdb_lookup: dict[str, bool],
    ignore_missing_provenance: bool,
) -> tuple[str, str]:
    stmt_types = [str(row.get(col, "")).strip() for col in _find_stmt_cols(fieldnames)]
    stmt_types = [stmt_type for stmt_type in stmt_types if stmt_type]
    if not stmt_types or not all(stmt_type in ALLOWED_STMT_TYPES for stmt_type in stmt_types):
        return "Undeterminable", "nondirectional_stmt_type"

    evidence_cols = _find_evidence_cols(fieldnames)
    if evidence_cols:
        if any(_is_msigdb_text(row.get(col)) for col in evidence_cols):
            return "Undeterminable", "msigdb_evidence"
    else:
        hash_cols = _find_hash_cols(fieldnames)
        stmt_hashes = [str(row.get(col, "")).strip() for col in hash_cols]
        stmt_hashes = [stmt_hash for stmt_hash in stmt_hashes if stmt_hash]
        if stmt_hashes:
            if any(hash_msigdb_lookup.get(stmt_hash, False) for stmt_hash in stmt_hashes):
                return "Undeterminable", "msigdb_evidence"
            if (
                not ignore_missing_provenance
                and not all(stmt_hash in hash_msigdb_lookup for stmt_hash in stmt_hashes)
            ):
                return "Undeterminable", "unresolved_msigdb_provenance"
        elif not ignore_missing_provenance:
            return "Undeterminable", "unresolved_msigdb_provenance"

    logfc = _to_float(row.get("logfoldchange"))
    if logfc is None:
        return "Undeterminable", "missing_logfc"

    product = 1
    for stmt_type in stmt_types:
        product *= _sign_for_stmt(stmt_type)

    if (product == -1 and logfc > 0.0) or (product == 1 and logfc < 0.0):
        return "Consistent", "determinable"
    return "Inconsistent", "determinable"


def _annotation_output_path(input_path: Path, annotated_dir: Path | None, cwd: Path) -> Path:
    if annotated_dir is None:
        return input_path

    try:
        relative = input_path.resolve().relative_to(cwd.resolve())
    except ValueError:
        relative = Path(input_path.name)
    return annotated_dir / relative


def _annotate_csv(
    csv_path: Path,
    output_path: Path,
    hash_msigdb_lookup: dict[str, bool],
    ignore_missing_provenance: bool,
    row_counter: dict[str, int],
    pair_flags: dict[tuple[str, str], dict[str, bool]],
) -> None:
    logger.info("Annotating %s -> %s", csv_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        fieldnames = list(reader.fieldnames or [])
        base_fields = [field for field in fieldnames if field != DIRECTIONAL_CONSISTENCY_COL]
        output_fields = base_fields + [DIRECTIONAL_CONSISTENCY_COL]

        if output_path.resolve() == csv_path.resolve():
            tmp_ctx = NamedTemporaryFile(
                "w",
                newline="",
                encoding="utf-8",
                delete=False,
                dir=str(csv_path.parent),
            )
        else:
            tmp_ctx = output_path.open("w", newline="", encoding="utf-8")

        with tmp_ctx as dst:
            writer = csv.DictWriter(dst, fieldnames=output_fields)
            writer.writeheader()

            for row in reader:
                label, reason = _classify_row(
                    row,
                    fieldnames,
                    hash_msigdb_lookup,
                    ignore_missing_provenance=ignore_missing_provenance,
                )
                out_row = {field: row.get(field, "") for field in base_fields}
                out_row[DIRECTIONAL_CONSISTENCY_COL] = label
                writer.writerow(out_row)

                src_gene = _norm_gene(row.get("source"))
                tgt_gene = _norm_gene(row.get("target"))
                if not src_gene or not tgt_gene or src_gene == tgt_gene:
                    continue

                row_counter["total_rows"] += 1
                if label == "Consistent":
                    row_counter["determinable_rows"] += 1
                    row_counter["consistent_rows"] += 1
                elif label == "Inconsistent":
                    row_counter["determinable_rows"] += 1
                    row_counter["inconsistent_rows"] += 1
                else:
                    row_counter["undeterminable_rows"] += 1
                    if reason == "nondirectional_stmt_type":
                        row_counter["undeterminable_nondirectional_rows"] += 1
                    elif reason == "msigdb_evidence":
                        row_counter["undeterminable_msigdb_rows"] += 1
                    elif reason == "unresolved_msigdb_provenance":
                        row_counter["undeterminable_unresolved_provenance_rows"] += 1
                    elif reason == "missing_logfc":
                        row_counter["undeterminable_missing_logfc_rows"] += 1

                pair = (src_gene, tgt_gene)
                flags = pair_flags[pair]
                if label == "Consistent":
                    flags["any_consistent"] = True
                elif label == "Inconsistent":
                    flags["any_inconsistent"] = True
                else:
                    flags["any_undeterminable"] = True

    if output_path.resolve() == csv_path.resolve():
        Path(dst.name).replace(output_path)


def _write_pair_detail(
    output_csv: Path,
    pair_flags: dict[tuple[str, str], dict[str, bool]],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (src, tgt), flags in pair_flags.items():
        rows.append(
            (
                src,
                tgt,
                _classify_pair(
                    flags["any_consistent"],
                    flags["any_undeterminable"],
                    flags["any_inconsistent"],
                ),
                flags["any_consistent"],
                flags["any_undeterminable"],
                flags["any_inconsistent"],
            )
        )
    rows.sort()

    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "source",
                "target",
                "pair_directional_consistency",
                "any_consistent_path",
                "any_undeterminable_path_without_consistent_override",
                "any_inconsistent_path_without_consistent_override",
            ]
        )
        writer.writerows(rows)
    logger.info("Wrote pair detail CSV: %s", output_csv)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Annotate pathway CSVs with directional consistency and summarize "
            "determinable vs undeterminable pathways under full-path sign logic."
        ),
    )
    ap.add_argument("--hop1-csv", default=None)
    ap.add_argument("--hop2-csv", default=None)
    ap.add_argument("--hop3-csv", default=None)
    ap.add_argument(
        "--csv",
        action="append",
        default=[],
        help="Generic pathway CSVs to annotate and summarize.",
    )
    ap.add_argument(
        "--extra-csv",
        action="append",
        default=[],
        help="Optional additional pathway CSVs to annotate and summarize.",
    )
    ap.add_argument(
        "--annotated-dir",
        default=None,
        help=(
            "Optional output directory for annotated CSV copies. "
            "If omitted, input CSVs are updated in place."
        ),
    )
    ap.add_argument(
        "--ignore-missing-provenance",
        action="store_true",
        help=(
            "When a row lacks evidence text and its statement hashes cannot be "
            "resolved to msigdb provenance locally, continue with sign-based "
            "classification instead of marking it Undeterminable."
        ),
    )
    ap.add_argument(
        "--provenance-search-root",
        default=".",
        help="Root directory to scan for evidence-bearing CSVs when recovering msigdb provenance.",
    )
    ap.add_argument(
        "--output-json",
        default="outputs/tables/sign_directionality_summary.json",
        help="Summary JSON output path.",
    )
    ap.add_argument(
        "--output-pair-csv",
        default=None,
        help="Optional pair-level classification CSV output path.",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    csv.field_size_limit(sys.maxsize)

    csv_specs: list[tuple[str, Path]] = []
    if args.hop1_csv or args.hop2_csv or args.hop3_csv:
        missing_hop_args = [
            name
            for name, value in (
                ("--hop1-csv", args.hop1_csv),
                ("--hop2-csv", args.hop2_csv),
                ("--hop3-csv", args.hop3_csv),
            )
            if not value
        ]
        if missing_hop_args:
            raise SystemExit(
                f"Hop mode requires all of --hop1-csv/--hop2-csv/--hop3-csv; missing {missing_hop_args}"
            )
        csv_specs.extend(
            [
                ("1-hop", Path(args.hop1_csv)),
                ("2-hop", Path(args.hop2_csv)),
                ("3-hop", Path(args.hop3_csv)),
            ]
        )
        for extra_csv in args.extra_csv:
            extra_path = Path(extra_csv)
            csv_specs.append((extra_path.stem, extra_path))
    else:
        if args.extra_csv:
            raise SystemExit("--extra-csv requires hop mode with --hop1-csv/--hop2-csv/--hop3-csv")
        if not args.csv:
            raise SystemExit("Provide either hop CSVs or at least one --csv input")
        csv_specs.extend((Path(csv_path).stem, Path(csv_path)) for csv_path in args.csv)

    annotated_dir = Path(args.annotated_dir) if args.annotated_dir else None
    provenance_search_root = Path(args.provenance_search_root)
    hash_msigdb_lookup = _build_hash_msigdb_lookup(provenance_search_root)

    per_file_row_stats: dict[str, dict[str, float | int]] = {}
    per_file_pair_flags: dict[str, dict[tuple[str, str], dict[str, bool]]] = {}
    combined_pair_flags: dict[tuple[str, str], dict[str, bool]] = defaultdict(
        lambda: {
            "any_consistent": False,
            "any_undeterminable": False,
            "any_inconsistent": False,
        }
    )
    combined_rows = _empty_row_counter()
    cwd = Path.cwd()

    for label, csv_path in csv_specs:
        row_counter = _empty_row_counter()
        file_pair_flags: dict[tuple[str, str], dict[str, bool]] = defaultdict(
            lambda: {
                "any_consistent": False,
                "any_undeterminable": False,
                "any_inconsistent": False,
            }
        )
        output_path = _annotation_output_path(csv_path, annotated_dir, cwd)
        _annotate_csv(
            csv_path=csv_path,
            output_path=output_path,
            hash_msigdb_lookup=hash_msigdb_lookup,
            ignore_missing_provenance=args.ignore_missing_provenance,
            row_counter=row_counter,
            pair_flags=file_pair_flags,
        )

        per_file_row_stats[label] = _finalize_row_counter(row_counter)
        per_file_pair_flags[label] = file_pair_flags

        for key in combined_rows:
            combined_rows[key] += row_counter[key]

        for pair, flags in file_pair_flags.items():
            combined = combined_pair_flags[pair]
            combined["any_consistent"] = combined["any_consistent"] or flags["any_consistent"]
            combined["any_undeterminable"] = (
                combined["any_undeterminable"] or flags["any_undeterminable"]
            )
            combined["any_inconsistent"] = combined["any_inconsistent"] or flags["any_inconsistent"]

    summary = {
        "rule": {
            "directional_stmt_types": sorted(ALLOWED_STMT_TYPES),
            "positive_stmt_types": sorted(POSITIVE_STMT_TYPES),
            "negative_stmt_types": sorted(NEGATIVE_STMT_TYPES),
            "undeterminable_if": [
                "any non-directional stmt_type in the pathway",
                "any hop evidence source contains msigdb",
                "msigdb provenance cannot be resolved locally for hash-only rows",
                "logfoldchange is missing or non-finite",
            ],
            "ignore_missing_provenance": bool(args.ignore_missing_provenance),
            "consistency_condition": {
                "product_eq_-1": "logfoldchange > 0",
                "product_eq_+1": "logfoldchange < 0",
            },
            "pair_consistency_rule": (
                "Priority order: any Consistent pathway => pair is Consistent; "
                "else any Undeterminable pathway => pair is Undeterminable; "
                "else pair is Inconsistent"
            ),
        },
        "pathway_level": {
            "per_file": per_file_row_stats,
            "combined": _finalize_row_counter(combined_rows),
        },
        "pair_level": {
            "per_file": {
                label: _summarize_pairs(flags)
                for label, flags in per_file_pair_flags.items()
            },
            "combined": _summarize_pairs(combined_pair_flags),
        },
        "annotated_csvs": {
            label: str(_annotation_output_path(path, annotated_dir, cwd))
            for label, path in csv_specs
        },
        "provenance_search_root": str(provenance_search_root),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote summary JSON: %s", out_json)
    logger.info("Combined pathway stats: %s", summary["pathway_level"]["combined"])

    if args.output_pair_csv:
        _write_pair_detail(Path(args.output_pair_csv), combined_pair_flags)


if __name__ == "__main__":
    main()
