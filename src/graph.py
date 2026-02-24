"""
LangGraph StateGraph for the Automaton Auditor Swarm.

Implements the detective fan-out / fan-in pattern:

    START
      │
      ├──► RepoInvestigator  ──┐
      │                        │
      └──► DocAnalyst        ──┤
                               │
                    EvidenceAggregator  (fan-in sync point)
                               │
                    [Judges — not yet wired]
                               │
                              END

The judicial layer (Prosecutor, Defense, TechLead → ChiefJustice) will be
added in the final submission.  The graph compiles and runs with a
MemorySaver checkpointer for crash recovery.
"""

from __future__ import annotations

import logging
from typing import Optional

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.state import AgentState
from src.nodes.detectives import (
    doc_analyst,
    evidence_aggregator,
    repo_investigator,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------


def build_detective_graph(checkpointer: Optional[MemorySaver] = None) -> StateGraph:
    """Build and compile the detective-phase StateGraph.

    Architecture:
        Fan-Out: START → [repo_investigator, doc_analyst] (parallel)
        Fan-In:  [repo_investigator, doc_analyst] → evidence_aggregator
        Terminal: evidence_aggregator → END

    The Annotated reducers in AgentState (`operator.ior` for evidences,
    `operator.add` for opinions) ensure that parallel detective writes
    merge safely without overwriting each other.

    Args:
        checkpointer: Optional MemorySaver for crash recovery.

    Returns:
        Compiled LangGraph application.
    """
    builder = StateGraph(AgentState)

    # --- Register nodes -----------------------------------------------------
    builder.add_node("repo_investigator", repo_investigator)
    builder.add_node("doc_analyst", doc_analyst)
    builder.add_node("evidence_aggregator", evidence_aggregator)

    # --- Fan-out: START dispatches to both detectives in parallel -----------
    #     LangGraph executes nodes that share the same source in parallel
    #     when using `add_edge` from a common predecessor.
    builder.add_edge(START, "repo_investigator")
    builder.add_edge(START, "doc_analyst")

    # --- Fan-in: both detectives converge at the aggregator -----------------
    builder.add_edge("repo_investigator", "evidence_aggregator")
    builder.add_edge("doc_analyst", "evidence_aggregator")

    # --- Terminal -----------------------------------------------------------
    builder.add_edge("evidence_aggregator", END)

    # --- Compile with optional checkpointer ---------------------------------
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = builder.compile(checkpointer=checkpointer)
    logger.info("Detective graph compiled successfully.")
    return graph


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_detective_graph(
    repo_url: str,
    pdf_path: Optional[str] = None,
    thread_id: str = "audit_session_1",
) -> dict:
    """Run the detective graph against a target repository and optional PDF.

    Args:
        repo_url: HTTPS URL of the GitHub repository to audit.
        pdf_path: Local path to the PDF report (optional for interim).
        thread_id: Unique thread ID for checkpointing.

    Returns:
        The final AgentState dict after execution.
    """
    graph = build_detective_graph()

    initial_state = {
        "repo_url": repo_url,
        "pdf_path": pdf_path,
        "evidences": {},
        "opinions": [],
        "report": None,
        "error": None,
    }

    config = {"configurable": {"thread_id": thread_id}}

    logger.info("Starting detective graph for %s", repo_url)
    final_state = graph.invoke(initial_state, config)
    logger.info(
        "Detective graph complete — %d evidence items collected.",
        len(final_state.get("evidences", {})),
    )
    return final_state
