"""
Judge nodes for the Automaton Auditor Swarm.

Each judge (Prosecutor, Defense, TechLead) receives the aggregated evidence
and produces a JudicialOpinion per rubric dimension using
`.with_structured_output(JudicialOpinion)`.

The three judges run in parallel (fan-out) and their opinions are collected
via the `operator.add` reducer on `state["opinions"]`.

Persona prompts are fundamentally distinct and conflicting:
  - Prosecutor: adversarial, looks for gaps, security flaws, laziness
  - Defense: forgiving, rewards effort, intent, creative workarounds
  - TechLead: pragmatic, focuses on architectural soundness & maintainability
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_ollama import ChatOllama
from pydantic import ValidationError

from src.state import AgentState, Evidence, JudicialOpinion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rubric dimensions each judge evaluates
# ---------------------------------------------------------------------------

_JUDGE_DIMENSIONS = [
    "git_forensic_analysis",
    "state_management_rigor",
    "graph_orchestration",
    "safe_tool_engineering",
    "structured_output_enforcement",
    "judicial_nuance",
    "chief_justice_synthesis",
    "theoretical_depth",
    "report_accuracy",
    "swarm_visual",
]

# ---------------------------------------------------------------------------
# Persona System Prompts — fundamentally distinct and conflicting
# ---------------------------------------------------------------------------

PROSECUTOR_SYSTEM_PROMPT = """You are the PROSECUTOR in an automated code audit tribunal.

Your role is ADVERSARIAL. You are relentless, skeptical, and unforgiving. You actively
search for gaps, security flaws, laziness, shortcuts, and architectural fraud.

Your evaluation philosophy:
- Assume the developer took shortcuts unless PROVEN otherwise by irrefutable evidence.
- If you find ANY security violation (os.system(), unsandboxed git clone, no input
  sanitization), this is an automatic failure — score 1 or 2 maximum.
- If the code is linear when it claims to be parallel, charge "Orchestration Fraud."
- If judges return freeform text without Pydantic validation, charge "Hallucination Liability."
- Missing features are NOT "planned" — they are MISSING. Stubs get no credit.
- Buzzwords without supporting code are "Keyword Dropping" — penalize heavily.

Use confrontational language: "The engineer FAILED to...", "This is a CLEAR VIOLATION of...",
"There is NO EVIDENCE of...", "The claim is UNSUBSTANTIATED."

Your score tendency: LOW (1-3). Only give 4-5 if evidence is truly irrefutable.

For each rubric dimension, evaluate the evidence and produce a structured opinion with:
- score (1-5): 1=The Vibe Coder, 3=Competent Orchestrator, 5=Master Thinker
- argument: Your adversarial reasoning citing specific evidence
- cited_evidence: List of dimension_ids you relied on
"""

DEFENSE_SYSTEM_PROMPT = """You are the DEFENSE ATTORNEY in an automated code audit tribunal.

Your role is FORGIVING and EMPATHETIC. You actively look for effort, intent, creative
workarounds, and partial implementations that show genuine understanding.

Your evaluation philosophy:
- Recognize that building a multi-agent system is HARD. Partial progress deserves credit.
- If the StateGraph fails to compile due to a minor edge error but the AST parsing logic
  is sophisticated, argue: "The engineer achieved deep code comprehension but tripped on
  framework syntax." Request boosting the score.
- If the Chief Justice uses an LLM prompt instead of deterministic rules but the judge
  personas are distinct and genuinely disagree, argue: "Role separation was successful,
  yielding true dialectical tension, even if synthesis lacks strict structural rigor."
  Request partial credit (3 or 4).
- Stubs with detailed docstrings show INTENT and UNDERSTANDING even if not implemented.
- Every genuine effort deserves acknowledgment. Find the silver lining.

Use encouraging language: "Despite the limitation, the engineer DEMONSTRATED...",
"Partial credit is WARRANTED because...", "The intent is CLEAR and the foundation is SOLID."

Your score tendency: HIGH (3-5). Only give 1-2 if there is truly zero effort.

For each rubric dimension, evaluate the evidence and produce a structured opinion with:
- score (1-5): 1=The Vibe Coder, 3=Competent Orchestrator, 5=Master Thinker
- argument: Your defense reasoning citing specific evidence showing effort
- cited_evidence: List of dimension_ids you relied on
"""

TECH_LEAD_SYSTEM_PROMPT = """You are the TECH LEAD in an automated code audit tribunal.

Your role is PRAGMATIC and ARCHITECTURAL. You focus on whether the code is sound,
maintainable, modular, and practically viable for production use.

Your evaluation philosophy:
- "Pydantic Rigor vs. Dict Soups": State definitions and JSON outputs MUST use typed
  structures (BaseModel). If standard Python dicts are used for complex nested state,
  rule: "Technical Debt" — score 3 max (functionally executes but is brittle).
- "Sandboxed Tooling": System-level interactions (cloning, parsing) MUST be wrapped in
  error handlers and temp directories. If os.system('git clone <url>') puts code in the
  live working directory, rule: "Security Negligence" — overrides all effort points.
- Evaluate whether the architecture is modular enough that a new developer could:
  (a) add a fourth judge persona, (b) add a new rubric dimension, (c) swap the LLM
  provider — WITHOUT rewriting more than one file.
- Consider error handling: does the code fail gracefully or crash silently?
- Consider testability: could unit tests be written for individual nodes?

Use measured, technical language: "The architecture is modular enough to...",
"This pattern introduces technical debt because...", "The separation of concerns is..."

Your score tendency: BALANCED (2-4). Give 5 only for genuinely excellent architecture.
Give 1 only for complete absence of engineering discipline.

For each rubric dimension, evaluate the evidence and produce a structured opinion with:
- score (1-5): 1=The Vibe Coder, 3=Competent Orchestrator, 5=Master Thinker
- argument: Your technical assessment citing specific architectural patterns
- cited_evidence: List of dimension_ids you relied on
"""

# ---------------------------------------------------------------------------
# LLM instance for judges
# ---------------------------------------------------------------------------

_judge_llm = ChatOllama(model="minimax-m2.5:cloud", temperature=0)


def _format_evidence_for_judge(evidences: Dict[str, Evidence]) -> str:
    """Format all collected evidence into a readable string for the judge prompt."""
    lines = []
    for dim_id, ev in sorted(evidences.items()):
        status = "FOUND" if ev.found else "NOT FOUND"
        lines.append(f"\n--- Evidence: {dim_id} [{status}] ---")
        lines.append(f"  Detective: {ev.detective}")
        lines.append(f"  Goal: {ev.goal}")
        lines.append(f"  Location: {ev.location}")
        lines.append(f"  Confidence: {ev.confidence:.0%}")
        lines.append(f"  Rationale: {ev.rationale}")
        if ev.content:
            # Truncate very long content
            content_preview = ev.content[:1500]
            if len(ev.content) > 1500:
                content_preview += "\n  ... [truncated]"
            lines.append(f"  Content:\n{content_preview}")
    return "\n".join(lines)


def _invoke_judge_for_dimension(
    llm,
    system_prompt: str,
    judge_name: str,
    dimension_id: str,
    evidence_text: str,
    max_retries: int = 3,
) -> JudicialOpinion:
    """Invoke the LLM for a single judge on a single dimension with retry logic.

    Uses .with_structured_output(JudicialOpinion) to guarantee typed JSON.
    Retries up to max_retries times on ValidationError or parse failure.
    Falls back to a score of 0 with error message on final failure.
    """
    structured_llm = llm.with_structured_output(JudicialOpinion)

    user_message = (
        f"Evaluate rubric dimension '{dimension_id}' based on the following forensic evidence.\n\n"
        f"EVIDENCE COLLECTED BY DETECTIVES:\n{evidence_text}\n\n"
        f"DIMENSION TO EVALUATE: {dimension_id}\n\n"
        f"You MUST set dimension_id to '{dimension_id}' and judge to '{judge_name}'.\n"
        f"Provide your score (1-5), argument, and list of cited_evidence dimension_ids."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for attempt in range(max_retries):
        try:
            opinion: JudicialOpinion = structured_llm.invoke(messages)
            # Validate and fix dimension_id / judge if the LLM got them wrong
            if opinion.dimension_id != dimension_id:
                opinion = opinion.model_copy(update={"dimension_id": dimension_id})
            if opinion.judge != judge_name:
                opinion = opinion.model_copy(update={"judge": judge_name})
            # Validate cited_evidence against known dimensions
            valid_citations = [
                c for c in opinion.cited_evidence
                if c in _JUDGE_DIMENSIONS or c == "repo_file_listing"
            ]
            if len(valid_citations) != len(opinion.cited_evidence):
                logger.warning(
                    "Judge %s cited unknown evidence for %s; stripped invalid citations.",
                    judge_name, dimension_id,
                )
                opinion = opinion.model_copy(update={"cited_evidence": valid_citations})
            logger.info(
                "Judge %s scored %s = %d (attempt %d)",
                judge_name, dimension_id, opinion.score, attempt + 1,
            )
            return opinion

        except (ValidationError, Exception) as exc:
            logger.warning(
                "Judge %s attempt %d/%d for %s failed: %s",
                judge_name, attempt + 1, max_retries, dimension_id, exc,
            )
            if attempt == max_retries - 1:
                # Final fallback: return a conservative opinion
                logger.error(
                    "Judge %s exhausted retries for %s; returning fallback opinion.",
                    judge_name, dimension_id,
                )
                return JudicialOpinion(
                    dimension_id=dimension_id,
                    judge=judge_name,
                    score=2,
                    argument=(
                        f"[FALLBACK] Judge {judge_name} could not produce a valid structured "
                        f"opinion after {max_retries} attempts. Error: {exc}. "
                        f"Defaulting to conservative score of 2."
                    ),
                    cited_evidence=[],
                )

    # Should not reach here, but just in case
    return JudicialOpinion(
        dimension_id=dimension_id,
        judge=judge_name,
        score=2,
        argument=f"[FALLBACK] Unreachable fallback for {judge_name} on {dimension_id}.",
        cited_evidence=[],
    )


def _run_judge(
    state: AgentState,
    system_prompt: str,
    judge_name: str,
) -> Dict[str, Any]:
    """Common logic for all three judge nodes.

    Iterates over each rubric dimension that has evidence, invokes the LLM
    with the judge-specific persona, and collects JudicialOpinion objects.
    """
    evidences = state.get("evidences", {})
    evidence_text = _format_evidence_for_judge(evidences)
    opinions: List[JudicialOpinion] = []

    # Evaluate each dimension that has evidence
    dimensions_to_evaluate = [
        d for d in _JUDGE_DIMENSIONS if d in evidences
    ]

    if not dimensions_to_evaluate:
        logger.warning("Judge %s has no evidence to evaluate.", judge_name)
        return {"opinions": []}

    for dimension_id in dimensions_to_evaluate:
        opinion = _invoke_judge_for_dimension(
            llm=_judge_llm,
            system_prompt=system_prompt,
            judge_name=judge_name,
            dimension_id=dimension_id,
            evidence_text=evidence_text,
        )
        opinions.append(opinion)

    logger.info(
        "Judge %s produced %d opinions across %d dimensions.",
        judge_name, len(opinions), len(dimensions_to_evaluate),
    )
    return {"opinions": opinions}


# ---------------------------------------------------------------------------
# Judge Node Functions (LangGraph nodes)
# ---------------------------------------------------------------------------


def prosecutor_node(state: AgentState) -> Dict[str, Any]:
    """The Prosecutor: adversarial judge looking for gaps, security flaws, laziness.

    Invokes LLM with PROSECUTOR_SYSTEM_PROMPT for each evidence dimension.
    Uses .with_structured_output(JudicialOpinion) with retry logic.
    """
    return _run_judge(state, PROSECUTOR_SYSTEM_PROMPT, "Prosecutor")


def defense_node(state: AgentState) -> Dict[str, Any]:
    """The Defense Attorney: rewards effort, intent, creative workarounds.

    Invokes LLM with DEFENSE_SYSTEM_PROMPT for each evidence dimension.
    Uses .with_structured_output(JudicialOpinion) with retry logic.
    """
    return _run_judge(state, DEFENSE_SYSTEM_PROMPT, "Defense")


def tech_lead_node(state: AgentState) -> Dict[str, Any]:
    """The Tech Lead: focuses on architectural soundness and maintainability.

    Invokes LLM with TECH_LEAD_SYSTEM_PROMPT for each evidence dimension.
    Uses .with_structured_output(JudicialOpinion) with retry logic.
    """
    return _run_judge(state, TECH_LEAD_SYSTEM_PROMPT, "TechLead")
