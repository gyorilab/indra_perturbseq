"""INDRA Perturb-seq: explaining Perturb-seq data via INDRA pathway analysis."""

import logging

__version__ = "0.1.0"

logging.basicConfig(
    format="%(levelname)s: [%(asctime)s] %(name)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("indra_cogex").setLevel(logging.WARNING)
logging.getLogger("indra_cogex.client.queries").setLevel(logging.WARNING)
