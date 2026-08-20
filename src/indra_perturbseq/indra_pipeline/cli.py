"""Command-line entry point for the INDRA Perturb-seq pipeline."""

from __future__ import annotations

import argparse
import logging

from indra_perturbseq.indra_pipeline.config import load_config, validate_config_paths
from indra_perturbseq.indra_pipeline.runner import run_pipeline
from indra_perturbseq.runtime import add_log_level_arg, configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Run the INDRA pipeline.")
    parser.add_argument("--config", required=True, help="YAML or JSON pipeline config.")
    parser.add_argument("--output-dir", default=None, help="Override run.output_dir.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the config and referenced input paths without running.",
    )
    parser.add_argument(
        "--deg-only",
        action="store_true",
        help="Generate DEG CSVs for raw input configs and stop.",
    )
    parser.add_argument(
        "--skip-deg",
        action="store_true",
        help="Reuse input.deg_output_dir for raw input configs instead of regenerating DEGs.",
    )
    add_log_level_arg(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline from a YAML or JSON config."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    configure_logging(args.log_level)

    try:
        cfg = load_config(
            args.config,
            output_dir=args.output_dir,
            require_graph=not args.deg_only,
        )
        validate_config_paths(
            cfg,
            skip_deg=args.skip_deg,
            deg_only=args.deg_only,
        )
        if args.validate_only:
            print(f"Config OK: {args.config}")
            return 0
        result = run_pipeline(cfg, deg_only=args.deg_only, skip_deg=args.skip_deg)
    except Exception as exc:
        if args.log_level.upper() == "DEBUG":
            logging.exception("Pipeline failed")
        else:
            logging.error("%s", exc)
        return 1

    if args.deg_only:
        deg = result["deg"]
        print(f"DEG directory: {deg['deg_output_dir']}")
        print(f"DEG files:     {deg['deg_source_count']}")
        return 0

    paths = result["paths"]
    print(f"1-hop results: {paths['onehop']}")
    print(f"2-hop results: {paths['twohop']}")
    print(f"Run summary:   {paths['summary']}")
    if paths.get("plots"):
        print(f"Plots:         {len(paths['plots'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
