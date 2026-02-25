"""
State definitions for the Automaton Auditor Swarm.

Uses Pydantic BaseModel for structured validation and TypedDict for LangGraph
state management. Annotated reducers (operator.add, operator.ior) prevent data
overwrites during parallel fan-out execution of detectives and judges.
"""

from __future__ import annotations

import operator
from typing import Annotated, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Pydantic models for structured, validated data within the graph
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """A single piece of evidence collected by a Detective node.

    Each detective produces one or more Evidence objects per rubric dimension
    it is responsible for. The `dimension_id` links back to the rubric JSON
    so that judges can route evidence to the correct criterion.
    """

    dimension_id: str = Field(
        ...,
        description="Rubric dimension this evidence addresses (e.g. 'git_forensic_analysis').",
    )
    detective: Literal["RepoInvestigator", "DocAnalyst", "VisionInspector"] = Field(
        ...,
        description="Which detective produced this evidence.",
    )
    goal: str = Field(
        ...,
        description="What this evidence was trying to establish or verify.",
    )
    found: bool = Field(
        ...,
        description="Whether the expected artifact / pattern was located.",
    )
    content: Optional[str] = Field(
        None,
        description="Extracted snippet, commit log, AST fragment, or PDF excerpt.",
    )
    location: str = Field(
        ...,
        description="File path, git ref, or logical location inside the target.",
    )
    rationale: str = Field(
        ...,
        description="Detective's reasoning for why this evidence supports the finding.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 = no confidence, 1.0 = irrefutable).",
    )


class JudicialOpinion(BaseModel):
    """A structured opinion returned by a Judge node.

    Judges are invoked with `.with_structured_output(JudicialOpinion)` to
    guarantee the LLM returns valid, typed JSON matching this schema.
    """

    dimension_id: str = Field(
        ...,
        description="Which rubric dimension is being judged.",
    )
    judge: Literal["Prosecutor", "Defense", "TechLead"] = Field(
        ...,
        description="Persona of the judge issuing this opinion.",
    )
    score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Score from 1 (The Vibe Coder) to 5 (Master Thinker).",
    )
    argument: str = Field(
        ...,
        description="The judge's reasoning and justification for the score.",
    )
    cited_evidence: List[str] = Field(
        default_factory=list,
        description="List of evidence dimension_ids or snippets this opinion relies on.",
    )


class CriterionVerdict(BaseModel):
    """Final verdict for a single rubric dimension after Chief Justice synthesis."""

    dimension_id: str
    final_score: int = Field(..., ge=1, le=5)
    reasoning: str
    dissent_summary: Optional[str] = Field(
        None,
        description="Required when score variance across judges > 2.",
    )
    prosecutor_score: int
    defense_score: int
    tech_lead_score: int


class AuditReport(BaseModel):
    """The complete audit report produced by the Chief Justice."""

    target_repo: str
    target_pdf: Optional[str] = None
    verdicts: List[CriterionVerdict] = Field(default_factory=list)
    executive_summary: str = ""
    remediation_plan: str = ""
    markdown: str = Field(
        "",
        description="Fully rendered Markdown report ready for export.",
    )


# ---------------------------------------------------------------------------
# LangGraph state — uses Annotated reducers for safe parallel writes
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """Top-level state flowing through the LangGraph StateGraph.

    Reducers:
    - `evidences`: operator.ior  — merges dicts so parallel detectives can
      each write to their own keys without overwriting each other.
    - `opinions`: operator.add   — concatenates lists so parallel judges
      append their opinions rather than replacing the list.
    """

    # Inputs
    repo_url: str
    pdf_path: Optional[str]

    # Collected by detectives (keyed by dimension_id)
    evidences: Annotated[Dict[str, Evidence], operator.ior]

    # Issued by judges (appended in parallel)
    opinions: Annotated[List[JudicialOpinion], operator.add]

    # Produced by Chief Justice
    report: Optional[AuditReport]

    # Error tracking
    error: Optional[str]
