"""
Chief Justice synthesis node for the Automaton Auditor Swarm.

The Chief Justice receives all JudicialOpinion objects from the three judges
and synthesises a final verdict using deterministic Python rules (not an LLM):

  1. Rule of Security — security flaws cap score at 3
  2. Rule of Evidence — facts overrule opinions (fact_supremacy)
  3. Rule of Functionality — Tech Lead carries highest weight for architecture
  4. Variance Re-evaluation — score spread > 2 triggers re-evaluation
  5. Dissent Requirement — divergent opinions must be explained

Output: a structured Markdown audit report (AuditReport) written to file.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.state import AgentState, AuditReport, CriterionVerdict, Evidence, JudicialOpinion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dimension metadata for report rendering
# ---------------------------------------------------------------------------

_DIMENSION_NAMES = {
    "git_forensic_analysis": "Git Forensic Analysis",
    "state_management_rigor": "State Management Rigor",
    "graph_orchestration": "Graph Orchestration Architecture",
    "safe_tool_engineering": "Safe Tool Engineering",
    "structured_output_enforcement": "Structured Output Enforcement",
    "judicial_nuance": "Judicial Nuance and Dialectics",
    "chief_justice_synthesis": "Chief Justice Synthesis Engine",
    "theoretical_depth": "Theoretical Depth (Documentation)",
    "report_accuracy": "Report Accuracy (Cross-Reference)",
    "swarm_visual": "Architectural Diagram Analysis",
}

# Security-related dimensions
_SECURITY_DIMENSIONS = {"safe_tool_engineering"}

# Architecture-related dimensions where TechLead opinion carries highest weight
_ARCHITECTURE_DIMENSIONS = {"graph_orchestration", "state_management_rigor"}


# ---------------------------------------------------------------------------
# Deterministic Synthesis Rules
# ---------------------------------------------------------------------------


def _apply_security_override(
    dimension_id: str,
    prosecutor_opinion: Optional[JudicialOpinion],
    evidences: Dict[str, Evidence],
    proposed_score: int,
) -> tuple[int, Optional[str]]:
    """Rule of Security: confirmed security flaws cap the score at 3.

    If the Prosecutor identifies a security vulnerability AND the detective
    evidence confirms it (e.g., os.system found, no tempfile usage), the
    score is capped regardless of Defense arguments.
    """
    if dimension_id not in _SECURITY_DIMENSIONS:
        return proposed_score, None

    # Check if evidence confirms security issues
    evidence = evidences.get(dimension_id)
    if evidence and not evidence.found:
        # Evidence says security is NOT clean
        if prosecutor_opinion and prosecutor_opinion.score <= 2:
            capped = min(proposed_score, 3)
            if capped < proposed_score:
                return capped, (
                    f"SECURITY OVERRIDE: Prosecutor identified security violations "
                    f"(score={prosecutor_opinion.score}) and detective evidence confirms "
                    f"unsafe practices. Score capped at {capped}."
                )
    return proposed_score, None


def _apply_fact_supremacy(
    dimension_id: str,
    defense_opinion: Optional[JudicialOpinion],
    evidences: Dict[str, Evidence],
    proposed_score: int,
) -> tuple[int, Optional[str]]:
    """Rule of Evidence: forensic facts overrule judicial opinions.

    If the Defense claims something positive but detective evidence contradicts
    it, the Defense is overruled.
    """
    evidence = evidences.get(dimension_id)
    if not evidence:
        return proposed_score, None

    if defense_opinion and defense_opinion.score >= 4 and not evidence.found:
        # Defense gave high score but evidence says the feature is missing
        adjusted = min(proposed_score, defense_opinion.score - 1, 3)
        return adjusted, (
            f"FACT SUPREMACY: Defense scored {defense_opinion.score} but detective evidence "
            f"shows the expected artifact was NOT found (confidence={evidence.confidence:.0%}). "
            f"Defense claim overruled. Score adjusted to {adjusted}."
        )
    return proposed_score, None


def _apply_functionality_weight(
    dimension_id: str,
    tech_lead_opinion: Optional[JudicialOpinion],
    proposed_score: int,
) -> tuple[int, Optional[str]]:
    """Rule of Functionality: Tech Lead carries highest weight for architecture.

    For architecture-related dimensions, the Tech Lead's score is weighted
    more heavily than the simple median.
    """
    if dimension_id not in _ARCHITECTURE_DIMENSIONS:
        return proposed_score, None

    if tech_lead_opinion:
        # Weight TechLead at 50%, others split the remaining 50%
        weighted = round(proposed_score * 0.5 + tech_lead_opinion.score * 0.5)
        if weighted != proposed_score:
            return weighted, (
                f"FUNCTIONALITY WEIGHT: Tech Lead opinion (score={tech_lead_opinion.score}) "
                f"carries highest weight for architecture dimension. "
                f"Adjusted from {proposed_score} to {weighted}."
            )
    return proposed_score, None


def _compute_variance_dissent(
    opinions: List[JudicialOpinion],
) -> tuple[bool, Optional[str]]:
    """Check if score variance exceeds 2 and generate dissent summary."""
    if len(opinions) < 2:
        return False, None

    scores = [o.score for o in opinions]
    variance = max(scores) - min(scores)

    if variance > 2:
        dissent_parts = []
        for op in opinions:
            dissent_parts.append(
                f"  - {op.judge} (score={op.score}): {op.argument[:200]}"
            )
        dissent = (
            f"DISSENT (variance={variance}): Judges disagreed significantly.\n"
            + "\n".join(dissent_parts)
        )
        return True, dissent
    return False, None


def _synthesize_dimension(
    dimension_id: str,
    opinions: List[JudicialOpinion],
    evidences: Dict[str, Evidence],
) -> CriterionVerdict:
    """Apply all synthesis rules for a single rubric dimension.

    Rule application order (precedence):
    1. Security Override (highest priority)
    2. Fact Supremacy
    3. Functionality Weight
    4. Variance Re-evaluation (lowest, generates dissent)
    """
    prosecutor = next((o for o in opinions if o.judge == "Prosecutor"), None)
    defense = next((o for o in opinions if o.judge == "Defense"), None)
    tech_lead = next((o for o in opinions if o.judge == "TechLead"), None)

    scores = [o.score for o in opinions]

    # Start with median score
    sorted_scores = sorted(scores)
    if len(sorted_scores) >= 3:
        proposed_score = sorted_scores[1]  # median of 3
    elif len(sorted_scores) == 2:
        proposed_score = round(sum(sorted_scores) / 2)
    elif len(sorted_scores) == 1:
        proposed_score = sorted_scores[0]
    else:
        proposed_score = 1

    reasoning_parts = []

    # Rule 1: Security Override
    proposed_score, security_note = _apply_security_override(
        dimension_id, prosecutor, evidences, proposed_score
    )
    if security_note:
        reasoning_parts.append(security_note)

    # Rule 2: Fact Supremacy
    proposed_score, fact_note = _apply_fact_supremacy(
        dimension_id, defense, evidences, proposed_score
    )
    if fact_note:
        reasoning_parts.append(fact_note)

    # Rule 3: Functionality Weight
    proposed_score, func_note = _apply_functionality_weight(
        dimension_id, tech_lead, proposed_score
    )
    if func_note:
        reasoning_parts.append(func_note)

    # Rule 4: Variance re-evaluation
    has_dissent, dissent_summary = _compute_variance_dissent(opinions)
    if has_dissent and dissent_summary:
        reasoning_parts.append(dissent_summary)
        # Re-evaluate: if high variance, lean toward the median but note it
        reasoning_parts.append(
            f"After variance re-evaluation, final score remains at {proposed_score} "
            f"(median preserved due to irreconcilable disagreement)."
        )

    # Clamp final score
    proposed_score = max(1, min(5, proposed_score))

    # Build reasoning
    if not reasoning_parts:
        reasoning_parts.append(
            f"Consensus reached. All three judges aligned within acceptable variance."
        )

    return CriterionVerdict(
        dimension_id=dimension_id,
        final_score=proposed_score,
        reasoning="\n".join(reasoning_parts),
        dissent_summary=dissent_summary if has_dissent else None,
        prosecutor_score=prosecutor.score if prosecutor else 0,
        defense_score=defense.score if defense else 0,
        tech_lead_score=tech_lead.score if tech_lead else 0,
    )


# ---------------------------------------------------------------------------
# Markdown Report Renderer
# ---------------------------------------------------------------------------


def _render_markdown_report(
    report: AuditReport,
    evidences: Dict[str, Evidence],
    opinions: List[JudicialOpinion],
) -> str:
    """Render the AuditReport into a professional Markdown document."""
    lines = []
    lines.append("# Automaton Auditor — Final Audit Report")
    lines.append("")
    lines.append(f"**Target Repository:** {report.target_repo}")
    if report.target_pdf:
        lines.append(f"**Target PDF Report:** {report.target_pdf}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # --- Executive Summary ---
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(report.executive_summary)
    lines.append("")

    # --- Aggregate Score ---
    if report.verdicts:
        total = sum(v.final_score for v in report.verdicts)
        max_total = len(report.verdicts) * 5
        avg = total / len(report.verdicts)
        lines.append(f"**Aggregate Score: {total}/{max_total} ({avg:.1f}/5.0 average)**")
        lines.append("")

    # --- Criterion Breakdown ---
    lines.append("## Criterion Breakdown")
    lines.append("")

    for verdict in report.verdicts:
        dim_name = _DIMENSION_NAMES.get(verdict.dimension_id, verdict.dimension_id)
        lines.append(f"### {dim_name}")
        lines.append("")
        lines.append(f"**Final Score: {verdict.final_score}/5**")
        lines.append("")
        lines.append(
            f"| Judge | Score | Key Argument |"
        )
        lines.append(f"|-------|-------|--------------|")

        # Find individual opinions for this dimension
        dim_opinions = [o for o in opinions if o.dimension_id == verdict.dimension_id]
        for op in dim_opinions:
            arg_preview = op.argument[:200].replace("|", "/").replace("\n", " ")
            lines.append(f"| {op.judge} | {op.score}/5 | {arg_preview} |")

        lines.append("")
        lines.append(f"**Chief Justice Reasoning:**")
        lines.append(f"{verdict.reasoning}")
        lines.append("")

        if verdict.dissent_summary:
            lines.append(f"**Dissent:**")
            lines.append(f"{verdict.dissent_summary}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # --- Remediation Plan ---
    lines.append("## Remediation Plan")
    lines.append("")
    lines.append(report.remediation_plan)
    lines.append("")

    return "\n".join(lines)


def _generate_executive_summary(verdicts: List[CriterionVerdict], repo_url: str) -> str:
    """Generate an executive summary from the synthesized verdicts."""
    if not verdicts:
        return "No verdicts produced. The audit pipeline did not complete successfully."

    total = sum(v.final_score for v in verdicts)
    max_total = len(verdicts) * 5
    avg = total / len(verdicts)

    # Find strongest and weakest dimensions
    sorted_verdicts = sorted(verdicts, key=lambda v: v.final_score)
    weakest = sorted_verdicts[0]
    strongest = sorted_verdicts[-1]

    weak_name = _DIMENSION_NAMES.get(weakest.dimension_id, weakest.dimension_id)
    strong_name = _DIMENSION_NAMES.get(strongest.dimension_id, strongest.dimension_id)

    # Count dissents
    dissent_count = sum(1 for v in verdicts if v.dissent_summary)

    summary = (
        f"This audit evaluated the target repository ({repo_url}) across "
        f"{len(verdicts)} rubric dimensions using the Dialectical Synthesis framework. "
        f"Three judge personas (Prosecutor, Defense, Tech Lead) independently evaluated "
        f"forensic evidence collected by detective agents, and the Chief Justice "
        f"synthesized their conflicting opinions using deterministic rules.\n\n"
        f"**Overall Result: {total}/{max_total} ({avg:.1f}/5.0 average)**\n\n"
        f"The strongest dimension was **{strong_name}** (score: {strongest.final_score}/5). "
        f"The weakest dimension was **{weak_name}** (score: {weakest.final_score}/5). "
        f"{dissent_count} dimension(s) had significant judicial disagreement (variance > 2) "
        f"requiring explicit dissent summaries."
    )
    return summary


def _generate_remediation_plan(
    verdicts: List[CriterionVerdict],
    evidences: Dict[str, Evidence],
) -> str:
    """Generate a prioritized, file-level remediation plan."""
    # Sort by score (lowest first) for priority ordering
    sorted_verdicts = sorted(verdicts, key=lambda v: v.final_score)

    items = []
    priority = 1

    for verdict in sorted_verdicts:
        if verdict.final_score >= 5:
            continue  # No remediation needed for perfect scores

        dim_name = _DIMENSION_NAMES.get(verdict.dimension_id, verdict.dimension_id)
        evidence = evidences.get(verdict.dimension_id)

        # Determine the file/component to modify
        location = evidence.location if evidence else "Unknown"

        # Generate specific remediation advice based on dimension
        remediation = _get_remediation_advice(verdict.dimension_id, verdict, evidence)

        items.append(
            f"### Priority {priority}: {dim_name} (Current Score: {verdict.final_score}/5)\n\n"
            f"**Affected Component:** {location}\n\n"
            f"**Current Gap:** {remediation['gap']}\n\n"
            f"**Recommended Action:** {remediation['action']}\n\n"
            f"**Expected Impact:** Score improvement to {remediation['target_score']}/5\n"
        )
        priority += 1

    if not items:
        return "All dimensions scored at maximum. No remediation required."

    return "\n".join(items)


def _get_remediation_advice(
    dimension_id: str,
    verdict: CriterionVerdict,
    evidence: Optional[Evidence],
) -> Dict[str, str]:
    """Generate dimension-specific remediation advice."""
    advice_map = {
        "git_forensic_analysis": {
            "gap": "Commit history does not show iterative development progression.",
            "action": "Restructure commits to show clear progression: Environment Setup → Tool Engineering → Graph Orchestration. Use atomic, meaningful commit messages.",
            "target_score": "4",
        },
        "state_management_rigor": {
            "gap": "State definitions missing Pydantic models or Annotated reducers for parallel-safe writes.",
            "action": "Ensure AgentState uses TypedDict with Annotated[Dict, operator.ior] and Annotated[List, operator.add]. Define Evidence and JudicialOpinion as Pydantic BaseModel classes.",
            "target_score": "5",
        },
        "graph_orchestration": {
            "gap": "Graph lacks two distinct parallel fan-out/fan-in patterns or conditional error edges.",
            "action": "Wire StateGraph with START → [Detectives in parallel] → Aggregator → [Judges in parallel] → ChiefJustice → END. Add conditional edges for error handling (e.g., clone failure skips judicial phase).",
            "target_score": "5",
        },
        "safe_tool_engineering": {
            "gap": "Git operations may not use sandboxed temp directories or proper error handling.",
            "action": "Wrap all git clone operations in tempfile.TemporaryDirectory(). Replace any os.system() calls with subprocess.run() with capture_output=True, text=True, and timeout. Add input sanitization for repo URLs.",
            "target_score": "5",
        },
        "structured_output_enforcement": {
            "gap": "Judge LLM calls do not use .with_structured_output(JudicialOpinion).",
            "action": "Bind all judge LLM invocations with .with_structured_output(JudicialOpinion). Add retry logic (3 attempts) with ValidationError catching. Validate output schema before adding to state.",
            "target_score": "5",
        },
        "judicial_nuance": {
            "gap": "Judge personas lack distinct, conflicting system prompts.",
            "action": "Write fundamentally different system prompts: Prosecutor (adversarial, looks for gaps), Defense (forgiving, rewards effort), TechLead (pragmatic, focuses on architecture). Ensure prompts share less than 50% text.",
            "target_score": "5",
        },
        "chief_justice_synthesis": {
            "gap": "ChiefJustice uses LLM averaging instead of deterministic Python rules.",
            "action": "Implement hardcoded if/else rules: security_override (cap at 3), fact_supremacy (evidence overrules opinions), functionality_weight (TechLead highest for architecture). Add variance > 2 dissent summaries. Output structured Markdown report.",
            "target_score": "5",
        },
        "theoretical_depth": {
            "gap": "Report lacks substantive explanation of Dialectical Synthesis, Fan-In/Fan-Out, Metacognition, State Synchronization.",
            "action": "Add detailed architectural explanations connecting each concept to concrete implementation: tie Dialectical Synthesis to three judge personas, Fan-In/Fan-Out to specific graph edges, Metacognition to self-audit capability.",
            "target_score": "5",
        },
        "report_accuracy": {
            "gap": "File paths mentioned in report do not match actual repository contents.",
            "action": "Cross-reference all file paths in the report against the actual repository structure. Remove hallucinated paths. Ensure every claimed feature has supporting code evidence.",
            "target_score": "5",
        },
        "swarm_visual": {
            "gap": "Architectural diagram does not accurately represent parallel branches.",
            "action": "Create a LangGraph state machine diagram showing both detective and judicial fan-out/fan-in patterns with visually distinct parallel branches and synchronization points.",
            "target_score": "4",
        },
    }

    return advice_map.get(dimension_id, {
        "gap": "Dimension did not achieve maximum score.",
        "action": "Review the rubric criteria and address specific gaps identified by detective evidence.",
        "target_score": "4",
    })


# ---------------------------------------------------------------------------
# Chief Justice Node (LangGraph node function)
# ---------------------------------------------------------------------------


def chief_justice_node(state: AgentState) -> Dict[str, Any]:
    """Deterministic synthesis of judicial opinions into a final verdict.

    Groups opinions by dimension, applies synthesis rules in precedence order
    (Security Override > Fact Supremacy > Functionality Weight > Variance),
    generates a structured Markdown report, and writes it to file.
    """
    opinions: List[JudicialOpinion] = state.get("opinions", [])
    evidences: Dict[str, Evidence] = state.get("evidences", {})
    repo_url: str = state.get("repo_url", "unknown")
    pdf_path: Optional[str] = state.get("pdf_path")

    if not opinions:
        logger.warning("ChiefJustice received no opinions — producing empty report.")
        report = AuditReport(
            target_repo=repo_url,
            target_pdf=pdf_path,
            executive_summary="No judicial opinions were produced. The audit pipeline may have failed before the judicial phase.",
            remediation_plan="Ensure the full pipeline (detectives → judges → synthesis) executes without errors.",
        )
        return {"report": report}

    # --- Group opinions by dimension ---
    opinions_by_dim: Dict[str, List[JudicialOpinion]] = defaultdict(list)
    for op in opinions:
        opinions_by_dim[op.dimension_id].append(op)

    # --- Synthesize each dimension ---
    verdicts: List[CriterionVerdict] = []
    for dimension_id, dim_opinions in sorted(opinions_by_dim.items()):
        verdict = _synthesize_dimension(dimension_id, dim_opinions, evidences)
        verdicts.append(verdict)
        logger.info(
            "Dimension %s: final_score=%d (P=%d, D=%d, TL=%d) %s",
            dimension_id,
            verdict.final_score,
            verdict.prosecutor_score,
            verdict.defense_score,
            verdict.tech_lead_score,
            "[DISSENT]" if verdict.dissent_summary else "",
        )

    # --- Generate report components ---
    executive_summary = _generate_executive_summary(verdicts, repo_url)
    remediation_plan = _generate_remediation_plan(verdicts, evidences)

    report = AuditReport(
        target_repo=repo_url,
        target_pdf=pdf_path,
        verdicts=verdicts,
        executive_summary=executive_summary,
        remediation_plan=remediation_plan,
    )

    # --- Render Markdown ---
    markdown = _render_markdown_report(report, evidences, opinions)
    report = report.model_copy(update={"markdown": markdown})

    # --- Write report to file ---
    report_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "audit_report.md")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        logger.info("Audit report written to %s", report_path)
    except Exception as exc:
        logger.error("Failed to write audit report: %s", exc)

    return {"report": report}
