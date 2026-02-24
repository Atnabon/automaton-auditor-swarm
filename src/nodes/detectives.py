"""
Detective nodes for the Automaton Auditor Swarm.

Each detective is a LangGraph node function that receives `AgentState` and
returns a partial state update with new `Evidence` objects keyed by
`dimension_id`.  Detectives run in parallel (fan-out) and their results are
merged by the `operator.ior` reducer on the `evidences` dict.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama
from pydantic import BaseModel

from src.state import AgentState, Evidence
from src.tools.doc_tools import (
    PDFDocument,
    extract_mentioned_paths,
    get_full_text,
    ingest_pdf,
    keyword_search,
)
from src.tools.repo_tools import (
    check_file_exists,
    clone_repo_sandboxed,
    extract_git_log,
    find_class_definitions,
    find_function_calls,
    find_imports,
    find_stategraph_builder,
    list_repo_files,
    parse_python_ast,
    scan_for_security_issues,
    CloneResult,
)

logger = logging.getLogger(__name__)

# LLM used for evidence synthesis — local Ollama model
_llm = ChatOllama(model="minimax-m2.5:cloud", temperature=0)


# ---------------------------------------------------------------------------
# RepoInvestigator — The Code Detective
# ---------------------------------------------------------------------------


def repo_investigator(state: AgentState) -> Dict[str, Any]:
    """Clone the target repo and collect forensic evidence across rubric dimensions.

    Produces Evidence for:
      - git_forensic_analysis
      - state_management_rigor
      - graph_orchestration
      - safe_tool_engineering
      - structured_output_enforcement
    """
    repo_url: str = state["repo_url"]
    evidences: Dict[str, Evidence] = {}

    # --- Step 1: Sandboxed clone -----------------------------------------------
    github_token = os.getenv("GITHUB_TOKEN")
    clone: CloneResult = clone_repo_sandboxed(repo_url, github_token)

    if not clone.success or not clone.repo_path:
        # Fatal: cannot analyse without the repo
        return {
            "evidences": {
                "git_forensic_analysis": Evidence(
                    dimension_id="git_forensic_analysis",
                    detective="RepoInvestigator",
                    found=False,
                    content=f"Clone failed: {clone.error}",
                    location=repo_url,
                    confidence=1.0,
                ),
            },
            "error": f"Clone failed: {clone.error}",
        }

    repo_path = clone.repo_path
    files = list_repo_files(repo_path)
    logger.info("Cloned %s — %d files found", repo_url, len(files))

    # --- Step 2: Git Forensic Analysis ----------------------------------------
    commits = extract_git_log(repo_path)
    commit_msgs = [f"{c.sha} {c.timestamp} {c.message}" for c in commits]
    progression_ok = len(commits) > 3
    evidences["git_forensic_analysis"] = Evidence(
        dimension_id="git_forensic_analysis",
        detective="RepoInvestigator",
        found=progression_ok,
        content=json.dumps(commit_msgs[:50], indent=2),
        location="git log",
        confidence=0.95 if progression_ok else 0.5,
    )

    # --- Step 3: State Management Rigor ---------------------------------------
    state_file = _find_file(files, ["src/state.py", "state.py"])
    if state_file:
        full_path = os.path.join(repo_path, state_file)
        tree = parse_python_ast(full_path)
        if tree:
            classes = find_class_definitions(tree)
            pydantic_classes = [c for c in classes if "BaseModel" in c["bases"]]
            typeddict_classes = [c for c in classes if "TypedDict" in c["bases"]]
            has_evidence_model = any(c["name"] == "Evidence" for c in pydantic_classes)
            has_judicial_opinion = any(c["name"] == "JudicialOpinion" for c in pydantic_classes)
            has_agent_state = any(
                c["name"] == "AgentState" for c in typeddict_classes + pydantic_classes
            )

            # Check for reducer annotations (operator.add / operator.ior)
            imports = find_imports(tree)
            has_operator = "operator" in imports

            snippet = Path(full_path).read_text(encoding="utf-8")[:2000]

            evidences["state_management_rigor"] = Evidence(
                dimension_id="state_management_rigor",
                detective="RepoInvestigator",
                found=has_evidence_model and has_agent_state,
                content=(
                    f"Pydantic BaseModel classes: {[c['name'] for c in pydantic_classes]}\n"
                    f"TypedDict classes: {[c['name'] for c in typeddict_classes]}\n"
                    f"Has operator import (for reducers): {has_operator}\n"
                    f"Has Evidence model: {has_evidence_model}\n"
                    f"Has JudicialOpinion model: {has_judicial_opinion}\n"
                    f"Has AgentState: {has_agent_state}\n\n"
                    f"--- Code Snippet ---\n{snippet}"
                ),
                location=state_file,
                confidence=0.9 if (has_evidence_model and has_agent_state) else 0.4,
            )
    else:
        evidences["state_management_rigor"] = Evidence(
            dimension_id="state_management_rigor",
            detective="RepoInvestigator",
            found=False,
            content="No state.py file found in expected locations.",
            location="src/state.py",
            confidence=0.9,
        )

    # --- Step 4: Graph Orchestration ------------------------------------------
    graph_file = _find_file(files, ["src/graph.py", "graph.py"])
    if graph_file:
        full_path = os.path.join(repo_path, graph_file)
        tree = parse_python_ast(full_path)
        if tree:
            graph_info = find_stategraph_builder(tree)
            snippet = Path(full_path).read_text(encoding="utf-8")[:3000]
            if graph_info:
                has_parallel = len(graph_info["nodes"]) >= 3
                evidences["graph_orchestration"] = Evidence(
                    dimension_id="graph_orchestration",
                    detective="RepoInvestigator",
                    found=True,
                    content=(
                        f"Nodes: {graph_info['nodes']}\n"
                        f"Edges: {graph_info['edges']}\n"
                        f"Conditional edges at lines: {graph_info['conditional_edges']}\n"
                        f"Parallel fan-out detected: {has_parallel}\n\n"
                        f"--- Code Snippet ---\n{snippet}"
                    ),
                    location=graph_file,
                    confidence=0.85,
                )
            else:
                evidences["graph_orchestration"] = Evidence(
                    dimension_id="graph_orchestration",
                    detective="RepoInvestigator",
                    found=False,
                    content=f"No StateGraph instantiation found.\n\n--- Code ---\n{snippet}",
                    location=graph_file,
                    confidence=0.7,
                )
    else:
        evidences["graph_orchestration"] = Evidence(
            dimension_id="graph_orchestration",
            detective="RepoInvestigator",
            found=False,
            content="No graph.py found.",
            location="src/graph.py",
            confidence=0.9,
        )

    # --- Step 5: Safe Tool Engineering ----------------------------------------
    tool_files = [f for f in files if f.startswith("src/tools/")]
    security_issues: List[Dict] = []
    uses_tempfile = False
    uses_subprocess = False
    clone_snippet = ""

    for tf in tool_files:
        full_path = os.path.join(repo_path, tf)
        tree = parse_python_ast(full_path)
        if tree:
            security_issues.extend(scan_for_security_issues(tree))
            imp = find_imports(tree)
            if "tempfile" in imp:
                uses_tempfile = True
            if "subprocess" in imp:
                uses_subprocess = True
            # Look for clone-related function
            clone_calls = find_function_calls(tree, "clone")
            if clone_calls:
                src = Path(full_path).read_text(encoding="utf-8")
                clone_snippet = src[:2000]

    evidences["safe_tool_engineering"] = Evidence(
        dimension_id="safe_tool_engineering",
        detective="RepoInvestigator",
        found=uses_tempfile and uses_subprocess and len(security_issues) == 0,
        content=(
            f"Uses tempfile: {uses_tempfile}\n"
            f"Uses subprocess (not os.system): {uses_subprocess}\n"
            f"Security issues: {json.dumps(security_issues, indent=2)}\n"
            f"Tool files scanned: {tool_files}\n\n"
            f"--- Clone snippet ---\n{clone_snippet}"
        ),
        location="src/tools/",
        confidence=0.85,
    )

    # --- Step 6: Structured Output Enforcement --------------------------------
    judges_file = _find_file(files, ["src/nodes/judges.py", "nodes/judges.py"])
    if judges_file:
        full_path = os.path.join(repo_path, judges_file)
        tree = parse_python_ast(full_path)
        if tree:
            structured_calls = find_function_calls(tree, "with_structured_output")
            bind_tools_calls = find_function_calls(tree, "bind_tools")
            snippet = Path(full_path).read_text(encoding="utf-8")[:2000]
            evidences["structured_output_enforcement"] = Evidence(
                dimension_id="structured_output_enforcement",
                detective="RepoInvestigator",
                found=len(structured_calls) > 0 or len(bind_tools_calls) > 0,
                content=(
                    f".with_structured_output() calls at lines: {structured_calls}\n"
                    f".bind_tools() calls at lines: {bind_tools_calls}\n\n"
                    f"--- Code Snippet ---\n{snippet}"
                ),
                location=judges_file,
                confidence=0.85,
            )
    else:
        evidences["structured_output_enforcement"] = Evidence(
            dimension_id="structured_output_enforcement",
            detective="RepoInvestigator",
            found=False,
            content="src/nodes/judges.py not found — judges not yet implemented.",
            location="src/nodes/judges.py",
            confidence=0.9,
        )

    # Store file listing for cross-reference by DocAnalyst
    evidences["repo_file_listing"] = Evidence(
        dimension_id="repo_file_listing",
        detective="RepoInvestigator",
        found=True,
        content=json.dumps(files, indent=2),
        location=repo_url,
        confidence=1.0,
    )

    # Cleanup is handled by temp_dir GC, but be explicit
    if clone.temp_dir:
        try:
            clone.temp_dir.cleanup()
        except Exception:
            pass

    return {"evidences": evidences}


# ---------------------------------------------------------------------------
# DocAnalyst — The Paperwork Detective
# ---------------------------------------------------------------------------


def doc_analyst(state: AgentState) -> Dict[str, Any]:
    """Analyse the PDF report for theoretical depth and accuracy.

    Produces Evidence for:
      - theoretical_depth
      - report_accuracy
    """
    pdf_path: Optional[str] = state.get("pdf_path")
    evidences: Dict[str, Evidence] = {}

    if not pdf_path:
        evidences["theoretical_depth"] = Evidence(
            dimension_id="theoretical_depth",
            detective="DocAnalyst",
            found=False,
            content="No PDF path provided.",
            location="N/A",
            confidence=1.0,
        )
        evidences["report_accuracy"] = Evidence(
            dimension_id="report_accuracy",
            detective="DocAnalyst",
            found=False,
            content="No PDF path provided.",
            location="N/A",
            confidence=1.0,
        )
        return {"evidences": evidences}

    pdf_doc = ingest_pdf(pdf_path)
    if pdf_doc is None:
        return {
            "evidences": {
                "theoretical_depth": Evidence(
                    dimension_id="theoretical_depth",
                    detective="DocAnalyst",
                    found=False,
                    content="Failed to ingest PDF.",
                    location=pdf_path,
                    confidence=1.0,
                ),
            }
        }

    # --- Theoretical Depth ----------------------------------------------------
    theory_keywords = [
        "Dialectical Synthesis",
        "Fan-In",
        "Fan-Out",
        "Metacognition",
        "State Synchronization",
    ]
    keyword_hits = keyword_search(pdf_doc, theory_keywords)
    full_text = get_full_text(pdf_doc)

    # Determine if keywords are substantive or just buzzwords
    keyword_contexts = {}
    for hit in keyword_hits:
        kw = hit["keyword"]
        if kw not in keyword_contexts:
            keyword_contexts[kw] = []
        keyword_contexts[kw].append(
            {"page": hit["page"], "context": hit["context"][:300]}
        )

    found_keywords = list(keyword_contexts.keys())
    missing_keywords = [k for k in theory_keywords if k not in found_keywords]

    evidences["theoretical_depth"] = Evidence(
        dimension_id="theoretical_depth",
        detective="DocAnalyst",
        found=len(found_keywords) >= 3,
        content=(
            f"Keywords found: {found_keywords}\n"
            f"Keywords missing: {missing_keywords}\n\n"
            f"Contexts:\n{json.dumps(keyword_contexts, indent=2, default=str)}"
        ),
        location=pdf_path,
        confidence=0.8 if len(found_keywords) >= 3 else 0.5,
    )

    # --- Report Accuracy (cross-reference with repo file listing) -------------
    mentioned_paths = extract_mentioned_paths(pdf_doc)

    # Store raw mentioned paths now — cross-reference is deferred to
    # evidence_aggregator, which runs after repo_investigator completes.
    # Running cross-reference here would always miss repo_file_listing
    # because doc_analyst and repo_investigator execute in parallel.
    evidences["report_accuracy"] = Evidence(
        dimension_id="report_accuracy",
        detective="DocAnalyst",
        found=len(mentioned_paths) > 0,
        content=(
            f"File paths mentioned in report: {json.dumps(mentioned_paths)}\n"
            f"[Cross-reference against repo pending — resolved by aggregator]"
        ),
        location=pdf_path,
        confidence=0.3,  # low until aggregator performs the cross-reference
    )

    return {"evidences": evidences}


# ---------------------------------------------------------------------------
# Evidence Aggregator — Fan-In synchronisation node
# ---------------------------------------------------------------------------


def evidence_aggregator(state: AgentState) -> Dict[str, Any]:
    """Synchronisation node that runs after all detectives complete.

    This is the fan-in barrier: both repo_investigator and doc_analyst have
    finished by the time this node executes.  We use this to:
    1. Cross-reference PDF-mentioned paths against the actual repo file listing.
    2. Validate all expected dimensions are present.
    """
    evidences = state.get("evidences", {})
    updated: Dict[str, Any] = {}

    # --- Cross-reference report_accuracy now that repo_file_listing exists ----
    report_acc = evidences.get("report_accuracy")
    repo_listing = evidences.get("repo_file_listing")

    if report_acc and repo_listing and repo_listing.content:
        try:
            repo_files: List[str] = json.loads(repo_listing.content)
        except (json.JSONDecodeError, TypeError):
            repo_files = []

        # Re-parse the mentioned paths from the stored content
        raw_content = report_acc.content or ""
        import re as _re
        path_match = _re.search(r"File paths mentioned in report: (\[.*?\])", raw_content, _re.DOTALL)
        try:
            mentioned_paths: List[str] = json.loads(path_match.group(1)) if path_match else []
        except (json.JSONDecodeError, AttributeError):
            mentioned_paths = []

        verified_paths = [p for p in mentioned_paths if p in repo_files]
        hallucinated_paths = [p for p in mentioned_paths if p not in repo_files]

        updated["report_accuracy"] = Evidence(
            dimension_id="report_accuracy",
            detective="DocAnalyst",
            found=len(hallucinated_paths) == 0 and len(mentioned_paths) > 0,
            content=(
                f"File paths mentioned in report: {mentioned_paths}\n"
                f"Verified paths: {verified_paths}\n"
                f"Hallucinated paths: {hallucinated_paths}\n"
                f"Repo files available for cross-ref: {len(repo_files)}"
            ),
            location=report_acc.location,
            confidence=0.9 if repo_files else 0.4,
        )
        logger.info(
            "Cross-reference complete: %d verified, %d hallucinated paths.",
            len(verified_paths),
            len(hallucinated_paths),
        )

    # --- Validate all expected dimensions are present -------------------------
    expected_dimensions = [
        "git_forensic_analysis",
        "state_management_rigor",
        "graph_orchestration",
        "safe_tool_engineering",
        "structured_output_enforcement",
        "theoretical_depth",
        "report_accuracy",
    ]
    all_evidences = {**evidences, **updated}
    present = [d for d in expected_dimensions if d in all_evidences]
    missing = [d for d in expected_dimensions if d not in all_evidences]

    logger.info(
        "Evidence aggregation complete: %d/%d dimensions collected. Missing: %s",
        len(present),
        len(expected_dimensions),
        missing or "none",
    )

    # Return updated evidence (operator.ior will merge with existing state)
    return {"evidences": updated} if updated else {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_file(files: List[str], candidates: List[str]) -> Optional[str]:
    """Return the first matching file path from a list of candidates."""
    for candidate in candidates:
        if candidate in files:
            return candidate
    # Fuzzy fallback: look for the basename anywhere
    for candidate in candidates:
        basename = os.path.basename(candidate)
        for f in files:
            if f.endswith(basename):
                return f
    return None
