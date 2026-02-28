"""
LangGraph StateGraph for the Automaton Auditor Swarm.

Implements two distinct parallel fan-out / fan-in patterns:

    START
      │
      ├──► RepoInvestigator  ──┐
      │                        │
      ├──► DocAnalyst        ──┤
      │                        │
      └──► VisionInspector   ──┤
                               │
                    EvidenceAggregator  (fan-in sync point #1)
                               │
                    [route_after_aggregation]
                       ├── error ──► END
                       └── ok    ──► Judicial Fan-Out
                                      │
                   ┌──────────────────┼──────────────────┐
                   │                  │                   │
              Prosecutor          Defense           TechLead
                   │                  │                   │
                   └──────────────────┼──────────────────┘
                                      │
                              JudicialSynchronizer  (fan-in sync point #2)
                                      │
                               ChiefJustice
                                      │
                                     END

The graph compiles and runs with a MemorySaver checkpointer for crash
recovery and LangSmith tracing.
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
    vision_inspector,
)
from src.nodes.judges import (
    prosecutor_node,
    defense_node,
    tech_lead_node,
)
from src.nodes.justice import chief_justice_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional Routing Functions
# ---------------------------------------------------------------------------


def route_to_judges(state: AgentState) -> list[str]:
    """Fan-out to all three judges in parallel after evidence aggregation.

    Conditional routing logic handles multiple error states:
    1. Fatal error (e.g. clone failure) → skip judicial phase entirely → END
    2. Missing evidence → not enough data to judge → END
    3. Normal case → dispatch all three judges in parallel

    Returns a list of node names — LangGraph will dispatch them concurrently.
    """
    # Error state: clone failure or fatal upstream error
    if state.get("error"):
        logger.warning("Fatal error detected; skipping judicial phase: %s", state["error"])
        return ["__end__"]

    # Missing evidence check: need at least 2 evidence dimensions to proceed
    evidences = state.get("evidences", {})
    if len(evidences) < 2:
        logger.warning(
            "Insufficient evidence collected (%d items); skipping judicial phase.",
            len(evidences),
        )
        return ["__end__"]

    return ["prosecutor", "defense", "tech_lead"]


def route_after_judicial_sync(state: AgentState) -> str:
    """Route after judicial synchronizer — validates judge outputs before synthesis.

    Handles malformed or missing judge output by checking that at least one
    valid JudicialOpinion exists. If judges produced no usable opinions,
    routes to END to prevent the Chief Justice from operating on empty data.
    """
    opinions = state.get("opinions", [])
    if not opinions:
        logger.warning("No judicial opinions produced; skipping Chief Justice synthesis.")
        return "__end__"

    # Check that we have opinions from at least 2 of 3 judges
    judges_seen = {o.judge for o in opinions}
    if len(judges_seen) < 2:
        logger.warning(
            "Only %d judge(s) produced opinions (%s); proceeding with partial synthesis.",
            len(judges_seen), judges_seen,
        )
    return "chief_justice"


# ---------------------------------------------------------------------------
# Judicial synchronization node
# ---------------------------------------------------------------------------


def judicial_synchronizer(state: AgentState) -> dict:
    """Fan-in synchronization point after all three judges complete.

    Validates that opinions have been collected from all three personas
    before dispatching to the ChiefJustice.
    """
    opinions = state.get("opinions", [])
    judges_seen = set(o.judge for o in opinions)
    expected = {"Prosecutor", "Defense", "TechLead"}
    missing = expected - judges_seen

    if missing:
        logger.warning(
            "Judicial synchronizer: missing opinions from %s. Proceeding with %d opinions.",
            missing, len(opinions),
        )
    else:
        logger.info(
            "Judicial synchronizer: all three judges reported. Total opinions: %d",
            len(opinions),
        )
    return {}


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------


def build_auditor_graph(checkpointer: Optional[MemorySaver] = None) -> StateGraph:
    """Build and compile the full Automaton Auditor StateGraph.

    Architecture:
        Fan-Out #1: START → [repo_investigator, doc_analyst, vision_inspector] (parallel)
        Fan-In #1:  [repo_investigator, doc_analyst, vision_inspector] → evidence_aggregator
        Routing:    evidence_aggregator →[conditional]→ judicial layer | END
        Fan-Out #2: [prosecutor, defense, tech_lead] (parallel judges)
        Fan-In #2:  [prosecutor, defense, tech_lead] → judicial_synchronizer
        Synthesis:  judicial_synchronizer → chief_justice → END

    The Annotated reducers in AgentState (`operator.ior` for evidences,
    `operator.add` for opinions) ensure that parallel writes merge safely.

    Args:
        checkpointer: Optional MemorySaver for crash recovery.

    Returns:
        Compiled LangGraph application.
    """
    builder = StateGraph(AgentState)

    # --- Register detective nodes ---
    builder.add_node("repo_investigator", repo_investigator)
    builder.add_node("doc_analyst", doc_analyst)
    builder.add_node("vision_inspector", vision_inspector)
    builder.add_node("evidence_aggregator", evidence_aggregator)

    # --- Register judicial nodes ---
    builder.add_node("prosecutor", prosecutor_node)
    builder.add_node("defense", defense_node)
    builder.add_node("tech_lead", tech_lead_node)
    builder.add_node("judicial_synchronizer", judicial_synchronizer)

    # --- Register synthesis node ---
    builder.add_node("chief_justice", chief_justice_node)

    # === Fan-Out #1: START dispatches to all detectives in parallel ===
    builder.add_edge(START, "repo_investigator")
    builder.add_edge(START, "doc_analyst")
    builder.add_edge(START, "vision_inspector")

    # === Fan-In #1: All detectives converge at the aggregator ===
    builder.add_edge("repo_investigator", "evidence_aggregator")
    builder.add_edge("doc_analyst", "evidence_aggregator")
    builder.add_edge("vision_inspector", "evidence_aggregator")

    # === Conditional routing: skip judges on fatal error ===
    builder.add_conditional_edges(
        "evidence_aggregator",
        route_to_judges,
        ["prosecutor", "defense", "tech_lead", "__end__"],
    )

    # === Fan-In #2: All judges converge at the synchronizer ===
    builder.add_edge("prosecutor", "judicial_synchronizer")
    builder.add_edge("defense", "judicial_synchronizer")
    builder.add_edge("tech_lead", "judicial_synchronizer")

    # === Conditional routing after judicial sync: skip synthesis if no opinions ===
    builder.add_conditional_edges(
        "judicial_synchronizer",
        route_after_judicial_sync,
        {"chief_justice": "chief_justice", "__end__": END},
    )
    builder.add_edge("chief_justice", END)

    # --- Compile with optional checkpointer ---
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = builder.compile(checkpointer=checkpointer)
    logger.info("Full auditor graph compiled successfully.")
    return graph


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_auditor_graph(
    repo_url: str,
    pdf_path: Optional[str] = None,
    thread_id: str = "audit_session_1",
) -> dict:
    """Run the full auditor graph against a target repository and optional PDF.

    Args:
        repo_url: HTTPS URL of the GitHub repository to audit.
        pdf_path: Local path to the PDF report (optional).
        thread_id: Unique thread ID for checkpointing.

    Returns:
        The final AgentState dict after execution.
    """
    graph = build_auditor_graph()

    initial_state = {
        "repo_url": repo_url,
        "pdf_path": pdf_path,
        "evidences": {},
        "opinions": [],
        "report": None,
        "error": None,
    }

    config = {"configurable": {"thread_id": thread_id}}

    logger.info("Starting full auditor graph for %s", repo_url)
    final_state = graph.invoke(initial_state, config)
    logger.info(
        "Auditor graph complete — %d evidence items, %d opinions.",
        len(final_state.get("evidences", {})),
        len(final_state.get("opinions", [])),
    )
    return final_state
