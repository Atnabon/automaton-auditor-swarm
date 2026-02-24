"""
Judge nodes for the Automaton Auditor Swarm.

STUB — Will be fully implemented for the final submission.

Each judge (Prosecutor, Defense, TechLead) receives the aggregated evidence
and produces a JudicialOpinion per rubric dimension using
`.with_structured_output(JudicialOpinion)`.

The three judges run in parallel (fan-out) and their opinions are collected
via the `operator.add` reducer on `state["opinions"]`.

Uses local Ollama model (minimax2.5) for inference.
"""

from __future__ import annotations

from typing import Any, Dict

from src.state import AgentState, JudicialOpinion


def prosecutor_node(state: AgentState) -> Dict[str, Any]:
    """The Prosecutor: adversarial judge looking for gaps, security flaws, laziness.

    TODO (Final Submission):
    - Load Prosecutor system prompt from rubric judicial_logic
    - For each evidence dimension, invoke LLM with:
        ChatOllama(model="minimax-m2.5:cloud").with_structured_output(JudicialOpinion)
    - Include retry logic for malformed outputs
    - Return {"opinions": [JudicialOpinion, ...]}
    """
    raise NotImplementedError("Prosecutor node — will be implemented for final submission.")


def defense_node(state: AgentState) -> Dict[str, Any]:
    """The Defense Attorney: rewards effort, intent, creative workarounds.

    TODO (Final Submission):
    - Load Defense system prompt from rubric judicial_logic
    - Mitigate harsh assessments where genuine effort is visible
    - Use ChatOllama(model="minimax-m2.5:cloud").with_structured_output(JudicialOpinion)
    - Return {"opinions": [JudicialOpinion, ...]}
    """
    raise NotImplementedError("Defense node — will be implemented for final submission.")


def tech_lead_node(state: AgentState) -> Dict[str, Any]:
    """The Tech Lead: focuses on architectural soundness and maintainability.

    TODO (Final Submission):
    - Load TechLead system prompt from rubric judicial_logic
    - Evaluate Pydantic rigor, sandboxing, modularity
    - Use ChatOllama(model="minimax-m2.5:cloud").with_structured_output(JudicialOpinion)
    - Return {"opinions": [JudicialOpinion, ...]}
    """
    raise NotImplementedError("TechLead node — will be implemented for final submission.")
