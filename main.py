"""
Automaton Auditor Swarm — CLI entry point.

Usage:
    python main.py <repo_url> [--pdf <path_to_pdf>]

Example:
    python main.py https://github.com/user/repo --pdf reports/interim_report.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

from src.graph import run_detective_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Automaton Auditor detective graph against a target repo."
    )
    parser.add_argument("repo_url", help="HTTPS URL of the GitHub repository to audit.")
    parser.add_argument("--pdf", default=None, help="Path to the PDF report (optional).")
    parser.add_argument(
        "--thread-id",
        default="audit_session_1",
        help="Thread ID for checkpointing (default: audit_session_1).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    print(f"\n🔍 Automaton Auditor Swarm — Detective Phase")
    print(f"   Target repo : {args.repo_url}")
    print(f"   PDF report  : {args.pdf or 'not provided'}")
    print(f"   Thread ID   : {args.thread_id}\n")

    final_state = run_detective_graph(
        repo_url=args.repo_url,
        pdf_path=args.pdf,
        thread_id=args.thread_id,
    )

    # Pretty-print collected evidence
    evidences = final_state.get("evidences", {})
    print(f"\n📋 Evidence Summary ({len(evidences)} items):\n")
    for dim_id, evidence in sorted(evidences.items()):
        status = "✅" if evidence.found else "❌"
        print(f"  {status} {dim_id}")
        print(f"     Location   : {evidence.location}")
        print(f"     Confidence : {evidence.confidence:.0%}")
        if evidence.content:
            preview = evidence.content[:150].replace("\n", " ")
            print(f"     Preview    : {preview}...")
        print()

    if final_state.get("error"):
        print(f"⚠️  Error: {final_state['error']}")

    print("✅ Detective phase complete. Judicial layer will be added in the final submission.")


if __name__ == "__main__":
    main()
