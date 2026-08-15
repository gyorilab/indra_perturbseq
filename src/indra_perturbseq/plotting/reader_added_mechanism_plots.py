"""Plot additional mechanisms introduced by reader-derived evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_deltas(pair_csv: Path) -> list[int]:
    deltas: list[int] = []
    with pair_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"source", "target", "all_paths", "db_only_paths"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"{pair_csv} must include columns {sorted(required)}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            src = str(row.get("source", "")).strip()
            tgt = str(row.get("target", "")).strip()
            if not src or not tgt or src.upper() == tgt.upper():
                continue
            all_paths = int(row.get("all_paths", "0"))
            db_only_paths = int(row.get("db_only_paths", "0"))
            deltas.append(all_paths - db_only_paths)
    return deltas


def _plot_histogram(
    deltas: list[int],
    output_png: Path,
    title: str,
    highlight_zero: bool,
    *,
    no_title: bool = False,
) -> dict[str, float]:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(deltas, dtype=int)
    max_delta = int(arr.max()) if len(arr) else 0
    bins = range(0, max_delta + 2)
    mean_delta = float(arr.mean()) if len(arr) else float("nan")
    median_delta = float(np.median(arr)) if len(arr) else float("nan")
    view_max = int(np.percentile(arr, 99)) if len(arr) else 0
    view_max = max(view_max, 10)

    plt.figure(figsize=(12, 6.75))
    if highlight_zero:
        zeros = arr[arr == 0]
        positives = arr[arr > 0]
        plt.hist(zeros, bins=bins, color="#9ca3af", edgecolor="white", label="Δ = 0")
        plt.hist(positives, bins=bins, color="#0f766e", edgecolor="white", label="Δ > 0")
    else:
        plt.hist(arr, bins=bins, color="#1f77b4", edgecolor="white", alpha=0.95, label="Pairs")

    plt.axvline(mean_delta, color="black", linestyle="--", linewidth=1.8, label=f"Mean = {mean_delta:.2f}")
    plt.axvline(median_delta, color="#d97706", linestyle=":", linewidth=2.0, label=f"Median = {median_delta:.2f}")

    plt.xlabel("Additional causal paths per pair (Δ paths)")
    plt.ylabel("Number of gene pairs")
    if not no_title:
        plt.title(title)
    plt.xlim(-0.5, view_max + 0.5)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "n_pairs": int(len(arr)),
        "min_delta": int(arr.min()) if len(arr) else 0,
        "max_delta": max_delta,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "n_zero_delta_pairs": int(np.sum(arr == 0)),
        "n_positive_delta_pairs": int(np.sum(arr > 0)),
    }


def _plot_tail_focus(
    deltas: list[int],
    output_png: Path,
    *,
    no_title: bool = False,
) -> dict[str, float]:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(deltas, dtype=int)
    if len(arr) == 0:
        raise ValueError("No delta values provided for tail plotting")

    tail_start = 10
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    max_delta = int(arr.max())
    marker = int(round(p99))
    right_pad = max(25, int(round(max_delta * 0.04)))

    xs = np.arange(0, max_delta + 1, dtype=int)
    counts = np.bincount(arr, minlength=max_delta + 1)
    mask_tail = xs >= tail_start

    ccdf_counts = np.array([int(np.sum(arr >= x)) for x in xs], dtype=int)
    ccdf_prob = ccdf_counts / len(arr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    axes[0].bar(
        xs[mask_tail],
        counts[mask_tail],
        width=1.0,
        color="#0ea5e9",
        edgecolor="white",
        linewidth=0.3,
    )
    axes[0].set_yscale("log")
    axes[0].set_xlim(tail_start - 1, max_delta + right_pad)
    axes[0].set_xlabel("Additional causal paths per pair (Δ)")
    axes[0].set_ylabel("Number of gene pairs (log scale)")
    if not no_title:
        axes[0].set_title(f"Tail frequency (Δ ≥ {tail_start})")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].axvline(
        p99,
        color="#7c3aed",
        linestyle="--",
        linewidth=1.6,
        label=f"Delta at 99th percentile = {p99:.0f}",
    )
    axes[0].axvline(
        max_delta,
        color="#ef4444",
        linestyle=":",
        linewidth=3.0,
        label=f"Largest delta = {max_delta}",
    )
    axes[0].legend(frameon=True)

    axes[1].plot(xs, ccdf_prob, color="#1d4ed8", linewidth=2.0)
    axes[1].set_yscale("log")
    axes[1].set_xlim(0, max_delta + right_pad)
    axes[1].set_xlabel("Additional causal paths per pair (Δ)")
    axes[1].set_ylabel("P(Δ ≥ x) [log scale]")
    if not no_title:
        axes[1].set_title("Tail persistence (CCDF)")
    axes[1].grid(True, alpha=0.25)
    axes[1].axvline(
        marker,
        color="#7c3aed",
        linestyle="--",
        linewidth=1.4,
        label=f"Delta at 99th percentile = {marker}",
    )
    axes[1].axvline(
        max_delta,
        color="#ef4444",
        linestyle=":",
        linewidth=3.0,
        label=f"Largest delta = {max_delta}",
    )
    axes[1].legend(frameon=True, loc="upper center")

    if not no_title:
        fig.suptitle("Additional mechanisms per pair: tail-focused view", fontsize=12)
    fig.tight_layout()
    plt.savefig(output_png, dpi=320, bbox_inches="tight")
    plt.close()

    freq_png = output_png.with_name(f"{output_png.stem}_frequency{output_png.suffix}")
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.bar(
        xs[mask_tail],
        counts[mask_tail],
        width=1.0,
        color="#0ea5e9",
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_yscale("log")
    ax.set_xlim(tail_start - 1, max_delta + right_pad)
    ax.set_xlabel("Additional causal paths per pair (Δ)")
    ax.set_ylabel("Number of gene pairs (log scale)")
    if not no_title:
        ax.set_title("Tail Frequency of Additional Mechanisms per Gene Pair")
    ax.grid(axis="y", alpha=0.25)
    ax.axvline(
        p99,
        color="#7c3aed",
        linestyle="--",
        linewidth=1.6,
        label=f"Delta at 99th percentile = {p99:.0f}",
    )
    ax.axvline(
        max_delta,
        color="#ef4444",
        linestyle=":",
        linewidth=3.0,
        label=f"Largest delta = {max_delta}",
    )
    ax.legend(frameon=True, loc="upper center")
    fig.tight_layout()
    plt.savefig(freq_png, dpi=320, bbox_inches="tight")
    plt.close()

    ccdf_png = output_png.with_name(f"{output_png.stem}_ccdf{output_png.suffix}")
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.plot(xs, ccdf_prob, color="#1d4ed8", linewidth=2.0)
    ax.set_yscale("log")
    ax.set_xlim(0, max_delta + right_pad)
    ax.set_xlabel("Additional causal paths per pair (Δ)")
    ax.set_ylabel("P(Δ ≥ x) [log scale]")
    if not no_title:
        ax.set_title("Tail Persistence of Additional Mechanisms per Gene Pair")
    ax.grid(True, alpha=0.25)
    ax.axvline(
        marker,
        color="#7c3aed",
        linestyle="--",
        linewidth=1.4,
        label=f"Delta at 99th percentile = {marker}",
    )
    ax.axvline(
        max_delta,
        color="#ef4444",
        linestyle=":",
        linewidth=3.0,
        label=f"Largest delta = {max_delta}",
    )
    ax.legend(frameon=True, loc="upper center")
    fig.tight_layout()
    plt.savefig(ccdf_png, dpi=320, bbox_inches="tight")
    plt.close()

    n_tail_start = int(np.sum(arr >= tail_start))
    n_marker = int(np.sum(arr >= marker))
    return {
        "n_pairs_total": int(len(arr)),
        "max_delta": max_delta,
        "p95": p95,
        "p99": p99,
        "tail_start": tail_start,
        "tail_marker": marker,
        "tail_frequency_png": str(freq_png),
        "tail_ccdf_png": str(ccdf_png),
        "n_pairs_delta_ge_tail_start": n_tail_start,
        "pct_pairs_delta_ge_tail_start": float(n_tail_start / len(arr) * 100.0),
        "n_pairs_delta_ge_tail_marker": n_marker,
        "pct_pairs_delta_ge_tail_marker": float(n_marker / len(arr) * 100.0),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Plot additional reader-added mechanisms from per-pair CSV.")
    ap.add_argument("--input-pair-csv", required=True)
    ap.add_argument("--output-png", required=True)
    ap.add_argument(
        "--output-summary-json",
        default="outputs/tables/additional_paths_hist_summary.json",
        help="Output summary JSON",
    )
    ap.add_argument(
        "--title",
        default="Additional Causal Paths Introduced by Reader-Derived Evidence",
    )
    ap.add_argument("--highlight-zero", action="store_true")
    ap.add_argument(
        "--tail-focus",
        action="store_true",
        help="Also generate a tail-focused figure at <output-png stem>_tail_focus.png.",
    )
    ap.add_argument("--no-title", action="store_true")
    args = ap.parse_args(argv)

    deltas = _load_deltas(Path(args.input_pair_csv))
    summary = _plot_histogram(
        deltas=deltas,
        output_png=Path(args.output_png),
        title=args.title,
        highlight_zero=args.highlight_zero,
        no_title=args.no_title,
    )

    if args.tail_focus:
        out_png = Path(args.output_png)
        tail_png = out_png.with_name(f"{out_png.stem}_tail_focus{out_png.suffix or '.png'}")
        summary["tail_focus"] = _plot_tail_focus(deltas=deltas, output_png=tail_png, no_title=args.no_title)

    out_json = Path(args.output_summary_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
