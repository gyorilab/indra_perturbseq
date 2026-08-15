"""Generate HTML evidence reports from INDRA statement data.

Queries INDRA for statements matching CSV rows, enriches the CSV with
statement hashes and URLs, and produces an ``HtmlAssembler`` report.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from indra.assemblers.html.assembler import HtmlAssembler
from indra_cogex.client.neo4j_client import Neo4jClient
from indra_cogex.client.queries import get_statements
from indra_perturbseq.services.neo4j import get_neo4j_client

logger = logging.getLogger(__name__)

INDRA_URL_TEMPLATE = "https://db.indra.bio/statements/from_hash/{hash}?format=html"


def _find_matching_statement(stmts: list, target_belief: float,
                             target_ev_count: int):
    """Return the statement best matching *target_belief* / *target_ev_count*.

    Parameters
    ----------
    stmts : list
        INDRA statements returned by ``get_statements``.
    target_belief : float
        Expected belief score from the CSV row.
    target_ev_count : int
        Expected evidence count from the CSV row.

    Returns
    -------
    stmt or None
        Best-matching statement, or ``None`` if *stmts* is empty.
    """
    best, min_diff = None, float("inf")
    for stmt in stmts:
        if len(stmt.evidence) == target_ev_count:
            diff = abs(stmt.belief - target_belief)
            if diff < min_diff:
                min_diff = diff
                best = stmt
    if best is not None:
        return best
    for stmt in stmts:
        diff = abs(stmt.belief - target_belief)
        if diff < min_diff:
            min_diff = diff
            best = stmt
    return best


def enrich_and_assemble(
    df: pd.DataFrame,
    client: Neo4jClient | None = None,
    title: str = "INDRA Evidence Report",
) -> tuple[pd.DataFrame, list]:
    """Enrich *df* with INDRA hashes/URLs and collect statements for HTML.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``source``, ``intermediate``, ``target``,
        ``stmt_type_1``, ``stmt_type_2``, ``belief_1``, ``belief_2``,
        ``evidence_1``, ``evidence_2``.
    client : Neo4jClient or None
        Neo4j client instance. Created on the fly when ``None``.
    title : str
        Title embedded in the HTML report.

    Returns
    -------
    tuple[pd.DataFrame, list]
        The enriched dataframe and the deduplicated statement list.
    """
    if client is None:
        client = get_neo4j_client()

    df = df.copy()
    hop1_hashes: list[str] = []
    hop1_urls: list[str] = []
    hop2_hashes: list[str] = []
    hop2_urls: list[str] = []
    all_stmts: list = []

    for idx, row in df.iterrows():
        for hop, (agent, other, stype, belief, evcnt, hashes, urls) in enumerate((
            (row["source"], row["intermediate"], row["stmt_type_1"],
             float(row["belief_1"]), int(row["evidence_1"]),
             hop1_hashes, hop1_urls),
            (row["intermediate"], row["target"], row["stmt_type_2"],
             float(row["belief_2"]), int(row["evidence_2"]),
             hop2_hashes, hop2_urls),
        ), start=1):
            try:
                stmts = get_statements(
                    agent=agent,
                    other_agent=other,
                    rel_types=stype,
                    evidence_limit=50,
                    client=client,
                )
            except Exception as exc:
                logger.warning(
                    "Statement lookup failed for hop %d (%s -> %s): %s",
                    hop,
                    agent,
                    other,
                    exc,
                )
                stmts = []
            matched = _find_matching_statement(stmts, belief, evcnt)
            if matched:
                h = matched.get_hash()
                hashes.append(h)
                urls.append(INDRA_URL_TEMPLATE.format(hash=h))
                all_stmts.append(matched)
            else:
                hashes.append("")
                urls.append("")

        if idx % 25 == 0:
            logger.info("Processed %d / %d rows", idx, len(df))

    df["hop1_hash"] = hop1_hashes
    df["hop1_indra_url"] = hop1_urls
    df["hop2_hash"] = hop2_hashes
    df["hop2_indra_url"] = hop2_urls

    seen: set[int] = set()
    unique_stmts = []
    for s in all_stmts:
        h = s.get_hash()
        if h not in seen:
            unique_stmts.append(s)
            seen.add(h)

    return df, unique_stmts


def build_html(stmts: list, title: str = "INDRA Evidence Report") -> str:
    """Assemble an HTML string from a list of INDRA statements.

    Parameters
    ----------
    stmts : list
        Deduplicated INDRA statements.
    title : str
        Page title.

    Returns
    -------
    str
        HTML content.
    """
    ha = HtmlAssembler(statements=stmts, title=title)
    ha.make_model(grouping_level="statement")
    return ha.model


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Enrich a 2-hop CSV with INDRA hashes/URLs and generate "
                    "an HTML evidence report.",
    )
    ap.add_argument("--input", required=True, help="Input CSV with 2-hop data.")
    ap.add_argument("--output-csv", required=True,
                    help="Path for the URL-enriched CSV.")
    ap.add_argument("--output-html", required=True,
                    help="Path for the HTML evidence report.")
    ap.add_argument("--title", default="INDRA Evidence Report",
                    help="Title for the HTML page.")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.input, low_memory=False)
    logger.info("Loaded %d rows from %s", len(df), args.input)

    df_enriched, stmts = enrich_and_assemble(df, title=args.title)
    df_enriched.to_csv(args.output_csv, index=False)
    logger.info("Enriched CSV -> %s", args.output_csv)

    ha = HtmlAssembler(statements=stmts, title=args.title)
    ha.make_model(grouping_level="statement")
    ha.save_model(args.output_html)
    logger.info("HTML report -> %s", args.output_html)


if __name__ == "__main__":
    main()
