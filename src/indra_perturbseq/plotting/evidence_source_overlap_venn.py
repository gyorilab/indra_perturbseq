"""Plot a 2-set pair-level overlap: DB-only vs DB+Reader.

Input is the JSON produced by ``evaluation/evidence_source_overlap.py``.
This script intentionally supports one publication use case only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

try:
    from matplotlib_venn import venn2
except ModuleNotFoundError:
    venn2 = None

from indra_perturbseq.utils.evidence_overlap_helpers import (
    build_db_only_vs_db_reader_counts,
)


def _plot(
    collapsed: dict[str, int],
    output_png: Path,
    title: str,
    *,
    no_title: bool = False,
) -> None:
    db_only_only_pairs = collapsed["db_only_only_pairs"]
    db_plus_reader_only_pairs = collapsed["db_plus_reader_only_pairs"]
    db_only_and_db_plus_reader_pairs = collapsed["db_only_and_db_plus_reader_pairs"]
    total_explained_pairs = collapsed["total_explained_pairs"]

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.6, 9.6))
    ax.set_position([0.10, 0.16, 0.80, 0.60])
    fig.text(
        0.5,
        0.955,
        f"Total explained gene-pairs: {total_explained_pairs:,}",
        ha="center",
        va="top",
        fontsize=28,
        fontweight="bold",
    )
    if not no_title:
        fig.text(
            0.5,
            0.775,
            title,
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
        )
    ax.set_xlim(0.0, 11.0)
    ax.set_ylim(0.0, 6.8)
    ax.set_aspect("equal")

    left = Circle((3.35, 3.15), 1.95, color="#ff858a", alpha=0.90, ec="none")
    right = Circle((6.75, 3.10), 3.20, color="#93c98d", alpha=0.84, ec="none")
    ax.add_patch(left)
    ax.add_patch(right)

    ax.text(
        2.15,
        3.00,
        f"{db_only_only_pairs:,}\npairs",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )
    ax.text(
        4.65,
        3.05,
        f"{db_only_and_db_plus_reader_pairs:,}\npairs",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )
    ax.text(
        7.75,
        3.00,
        f"{db_plus_reader_only_pairs:,}\npairs",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )
    ax.text(2.00, 0.45, "DB-only", ha="center", va="center", fontsize=22, fontweight="bold")
    ax.text(7.30, 0.15, "DB+Reader", ha="center", va="center", fontsize=22, fontweight="bold")
    ax.axis("off")

    fig.text(
        0.5,
        0.055,
        "DB-only set = gene pairs with at least one pathway fully supported by databases\n"
        "DB+Reader set = gene pairs with at least one pathway that requires reader-extracted support",
        ha="center",
        va="bottom",
        fontsize=11,
        linespacing=1.35,
    )
    plt.savefig(output_png, dpi=300)
    plt.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Plot 2-set DB-only vs DB+Reader overlap from JSON counts."
    )
    ap.add_argument("--input-json", required=True, help="Path to overlap JSON")
    ap.add_argument("--output-png", required=True, help="Output figure path")
    ap.add_argument("--title", default="Gene-Pair Overlap")
    ap.add_argument("--no-title", action="store_true")
    args = ap.parse_args(argv)

    input_json = Path(args.input_json)
    payload = json.loads(input_json.read_text())
    counts = payload.get("combined_all_hops")
    if not isinstance(counts, dict):
        raise ValueError(f"{input_json} missing 'combined_all_hops'")

    collapsed = build_db_only_vs_db_reader_counts(counts)
    _plot(collapsed=collapsed, output_png=Path(args.output_png), title=args.title, no_title=args.no_title)


if __name__ == "__main__":
    main()
