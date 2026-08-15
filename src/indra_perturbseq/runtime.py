"""Shared runtime helpers for CLI modules.
This module provides shared utilities used across the INDRA Perturb-seq codebase.
"""

from __future__ import annotations

import argparse
import logging

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def add_log_level_arg(parser: argparse.ArgumentParser, default: str = "INFO") -> None:
    """Add a standard ``--log-level`` argument to a parser."""
    parser.add_argument(
        "--log-level",
        default=default,
        choices=_LOG_LEVELS,
        help="Logging verbosity",
    )


def configure_logging(log_level: str) -> None:
    """Configure root logging with a consistent format."""
    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))
