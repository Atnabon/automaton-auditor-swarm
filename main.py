"""
Automaton Auditor Swarm — CLI entry point.

Usage:
    python main.py <repo_url> [--pdf <path_to_pdf>]

Example:
    python main.py https://github.com/user/repo --pdf reports/final_report.pdf

LangSmith Tracing:
    Set LANGCHAIN_API_KEY in a .env file or environment variable.
    Traces are sent to the 'automaton-auditor-swarm' project automatically.
    Share the trace link from https://smith.langchain.com for the submission.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Enable LangSmith tracing — required for the LangSmith Trace submission
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "automaton-auditor-swarm")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

from src.graph import run_auditor_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Automaton Auditor Swarm — full pipeline."
    )
    parser.add_argument("repo_url", help="HTTPS URL of the GitHub repository to audit.")
    parser.add_argument("--pdf", default=None, help="Path to the PDF report.")
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

    print("\n" + "=" * 70)
    print("  AUTOMATON AUDITOR SWARM — Full Pipeline Execution")
    print("=" * 70)
    print(f"  Target repo : {args.repo_url}")
    print(f"  PDF report  : {args.pdf or 'not provided'}")
    print(f"  Thread ID   : {args.thread_id}")
    langsmith_key = os.environ.get("LANGCHAIN_API_KEY", "")
    langsmith_status = "ENABLED" if langsmith_key else "DISABLED (set LANGCHAIN_API_KEY)"
    print(f"  LangSmith   : {langsmith_status}")
    print(f"  Project     : {os.environ.get('LANGCHAIN_PROJECT', 'N/A')}")
    print("=" * 70 + "\n")

    # --- Run the full auditor graph ---
    try:
        final_state = run_auditor_graph(
            repo_url=args.repo_url,
            pdf_path=args.pdf,
            thread_id=args.thread_id,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("Pipeline execution failed: %s", exc)
        print(f"\n  ❌ Pipeline execution failed: {exc}")
        print("  Check the LangSmith trace for detailed error information.")
        sys.exit(1)

    # --- Display Evidence Summary ---
    evidences = final_state.get("evidences", {})
    print(f"\n{'=' * 70}")
    print(f"  DETECTIVE LAYER — Evidence Summary ({len(evidences)} items)")
    print(f"{'=' * 70}\n")
    for dim_id, evidence in sorted(evidences.items()):
        status = "FOUND" if evidence.found else "NOT FOUND"
        icon = "✅" if evidence.found else "❌"
        print(f"  {icon} {dim_id} [{status}]")
        print(f"     Detective   : {evidence.detective}")
        print(f"     Location    : {evidence.location}")
        print(f"     Confidence  : {evidence.confidence:.0%}")
        if evidence.content:
            preview = evidence.content[:150].replace("\n", " ")
            print(f"     Preview     : {preview}...")
        print()

    # --- Display Judicial Opinions ---
    opinions = final_state.get("opinions", [])
    if opinions:
        print(f"{'=' * 70}")
        print(f"  JUDICIAL LAYER — Judge Opinions ({len(opinions)} opinions)")
        print(f"{'=' * 70}\n")

        # Group by dimension
        from collections import defaultdict
        opinions_by_dim = defaultdict(list)
        for op in opinions:
            opinions_by_dim[op.dimension_id].append(op)

        for dim_id, dim_opinions in sorted(opinions_by_dim.items()):
            print(f"  --- {dim_id} ---")
            for op in dim_opinions:
                print(f"    {op.judge:12s} | Score: {op.score}/5 | {op.argument[:100]}...")
            print()

    # --- Display Final Report ---
    report = final_state.get("report")
    if report and report.markdown:
        print(f"{'=' * 70}")
        print(f"  CHIEF JUSTICE — Final Audit Report")
        print(f"{'=' * 70}\n")

        # Show executive summary
        print(report.executive_summary)
        print()

        # Show per-criterion scores
        if report.verdicts:
            print("  Per-Criterion Scores:")
            total = 0
            for v in report.verdicts:
                total += v.final_score
                dissent_marker = " [DISSENT]" if v.dissent_summary else ""
                print(f"    {v.dimension_id:40s} : {v.final_score}/5{dissent_marker}")
            print(f"    {'─' * 50}")
            print(f"    {'TOTAL':40s} : {total}/{len(report.verdicts) * 5}")
            print()

        # Report file location
        report_path = os.path.join(os.getcwd(), "reports", "audit_report.md")
        if os.path.isfile(report_path):
            print(f"  📄 Full report written to: {report_path}")
        print()

    if final_state.get("error"):
        print(f"  ⚠️  Error: {final_state['error']}")

    # --- Report artifacts ---
    report_dir = os.path.join(os.getcwd(), "reports")
    report_files = [
        ("Self-Audit", "self_audit_report.md"),
        ("Peer-Audit", "peer_audit_report.md"),
        ("Peer-Received", "peer_received_report.md"),
        ("Generated", "audit_report.md"),
    ]
    print(f"{'=' * 70}")
    print(f"  REPORT ARTIFACTS")
    print(f"{'=' * 70}\n")
    for label, filename in report_files:
        path = os.path.join(report_dir, filename)
        status = "✅" if os.path.isfile(path) else "❌"
        print(f"  {status} {label:15s}: reports/{filename}")
    print()

    # --- LangSmith trace info ---
    if os.environ.get("LANGCHAIN_API_KEY"):
        project = os.environ.get("LANGCHAIN_PROJECT", "automaton-auditor-swarm")
        print(f"  🔗 LangSmith trace: https://smith.langchain.com/o/default/projects/p/{project}")
        print(f"     (Find the latest run in the '{project}' project)")
    else:
        print(f"  ⚠️  LangSmith tracing disabled — set LANGCHAIN_API_KEY to enable.")
    print()

    print(f"{'=' * 70}")
    print("  ✅ Automaton Auditor Swarm — Pipeline Complete")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
