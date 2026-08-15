#!/usr/bin/env python3
"""Run the flexible INDRA pipeline from a YAML or JSON config."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from indra_perturbseq.indra_pipeline.config import load_config
from indra_perturbseq.indra_pipeline.runner import run_pipeline
from indra_perturbseq.runtime import add_log_level_arg, configure_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the flexible INDRA pipeline.")
    parser.add_argument("--config", required=True, help="YAML or JSON pipeline config.")
    parser.add_argument("--output-dir", default=None, help="Override run.output_dir.")
    add_log_level_arg(parser)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    configure_logging(args.log_level)

    cfg = load_config(args.config, output_dir=args.output_dir)
    result = run_pipeline(cfg)
    paths = result["paths"]
    print(f"1-hop results: {paths['onehop']}")
    print(f"2-hop results: {paths['twohop']}")
    print(f"Run summary:   {paths['summary']}")
    if paths.get("plots"):
        print(f"Plots:         {len(paths['plots'])} files")


if __name__ == "__main__":
    main()
