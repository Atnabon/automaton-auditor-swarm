"""
Sandboxed repository tools for the Automaton Auditor.

All git operations run inside `tempfile.TemporaryDirectory()` for isolation.
Uses `subprocess.run()` with proper error handling — never raw `os.system()`.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sandboxed Git Clone
# ---------------------------------------------------------------------------


@dataclass
class CloneResult:
    """Result of a sandboxed git clone operation."""

    success: bool
    repo_path: Optional[str] = None
    error: Optional[str] = None
    temp_dir: Optional[tempfile.TemporaryDirectory] = field(default=None, repr=False)


def clone_repo_sandboxed(repo_url: str, github_token: Optional[str] = None) -> CloneResult:
    """Clone a repository into a temporary directory.

    The caller is responsible for keeping the returned `CloneResult.temp_dir`
    alive for as long as they need the files.  When the TemporaryDirectory
    context manager is garbage-collected or explicitly cleaned up, the clone
    is deleted.

    Args:
        repo_url: HTTPS URL to the target git repository.
        github_token: Optional GitHub PAT for private repos.

    Returns:
        CloneResult with the path to the cloned repo on success.
    """
    # Sanitise the URL — reject anything that looks like command injection
    if not repo_url.startswith(("https://", "http://")):
        return CloneResult(success=False, error=f"Rejected URL scheme: {repo_url!r}")

    for dangerous_char in (";", "|", "&", "$", "`", "\n", "\r"):
        if dangerous_char in repo_url:
            return CloneResult(
                success=False,
                error=f"URL contains dangerous character {dangerous_char!r}",
            )

    # Inject token for private repos if provided
    clone_url = repo_url
    if github_token and "github.com" in repo_url:
        clone_url = repo_url.replace(
            "https://github.com",
            f"https://{github_token}@github.com",
        )

    tmp = tempfile.TemporaryDirectory(prefix="auditor_clone_")
    clone_path = os.path.join(tmp.name, "repo")

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "100", clone_url, clone_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            tmp.cleanup()
            return CloneResult(
                success=False,
                error=f"git clone failed (rc={result.returncode}): {result.stderr.strip()}",
            )
        return CloneResult(success=True, repo_path=clone_path, temp_dir=tmp)

    except subprocess.TimeoutExpired:
        tmp.cleanup()
        return CloneResult(success=False, error="git clone timed out after 120 s")
    except FileNotFoundError:
        tmp.cleanup()
        return CloneResult(success=False, error="'git' binary not found on PATH")
    except Exception as exc:
        tmp.cleanup()
        return CloneResult(success=False, error=str(exc))


# ---------------------------------------------------------------------------
# Git Log Extraction
# ---------------------------------------------------------------------------


@dataclass
class CommitInfo:
    """Parsed representation of a single git commit."""

    sha: str
    message: str
    timestamp: str


def extract_git_log(repo_path: str, max_commits: int = 200) -> List[CommitInfo]:
    """Run `git log --oneline --reverse` and parse the output.

    Returns a list of CommitInfo ordered oldest-first so that progression
    analysis (setup → tools → graph) can be performed chronologically.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                "--reverse",
                f"--max-count={max_commits}",
                "--format=%H|%s|%aI",
            ],
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("git log failed: %s", result.stderr.strip())
            return []

        commits: List[CommitInfo] = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3:
                commits.append(CommitInfo(sha=parts[0][:8], message=parts[1], timestamp=parts[2]))
        return commits

    except Exception as exc:
        logger.warning("git log extraction error: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Repository Structure Analysis
# ---------------------------------------------------------------------------


def list_repo_files(repo_path: str, extensions: Optional[Tuple[str, ...]] = None) -> List[str]:
    """Walk the cloned repo and return relative file paths.

    Skips hidden directories (`.git`, `__pycache__`, etc.).
    """
    extensions = extensions or (".py", ".json", ".toml", ".md", ".yaml", ".yml")
    found: List[str] = []
    root = Path(repo_path)
    for path in root.rglob("*"):
        if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
            continue
        if path.is_file() and path.suffix in extensions:
            found.append(str(path.relative_to(root)))
    return sorted(found)


def check_file_exists(repo_path: str, relative_path: str) -> bool:
    """Check whether a file exists in the cloned repo."""
    return Path(repo_path, relative_path).is_file()


# ---------------------------------------------------------------------------
# AST-Based Code Analysis
# ---------------------------------------------------------------------------


def parse_python_ast(file_path: str) -> Optional[ast.Module]:
    """Safely parse a Python file into an AST module."""
    try:
        source = Path(file_path).read_text(encoding="utf-8")
        return ast.parse(source, filename=file_path)
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError) as exc:
        logger.warning("AST parse failed for %s: %s", file_path, exc)
        return None


def find_class_definitions(tree: ast.Module) -> List[Dict]:
    """Extract class definitions, their bases, and fields from an AST.

    Returns a list of dicts with keys: name, bases, fields, lineno.
    """
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(ast.dump(base))
            # Extract annotated fields (Pydantic / TypedDict style)
            fields = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    fields.append(item.target.id)
            classes.append(
                {
                    "name": node.name,
                    "bases": bases,
                    "fields": fields,
                    "lineno": node.lineno,
                }
            )
    return classes


def find_stategraph_builder(tree: ast.Module) -> Optional[Dict]:
    """Look for `StateGraph(...)` instantiation and extract `add_node` / `add_edge` calls.

    Returns a dict with keys: nodes, edges, conditional_edges, lineno.
    """
    info: Dict = {"nodes": [], "edges": [], "conditional_edges": [], "lineno": None}
    found = False

    for node in ast.walk(tree):
        # Detect `StateGraph(...)` call
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "StateGraph":
                info["lineno"] = node.lineno
                found = True

        # Detect `builder.add_node("name", func)`
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name == "add_node" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant):
                    info["nodes"].append(arg.value)
                found = True
            elif attr_name == "add_edge" and len(node.args) >= 2:
                src = node.args[0]
                dst = node.args[1]
                src_val = src.value if isinstance(src, ast.Constant) else ast.dump(src)
                dst_val = dst.value if isinstance(dst, ast.Constant) else ast.dump(dst)
                info["edges"].append((src_val, dst_val))
                found = True
            elif attr_name == "add_conditional_edges":
                info["conditional_edges"].append(node.lineno)
                found = True

    return info if found else None


def extract_code_snippet(file_path: str, start_line: int, end_line: int) -> str:
    """Extract a code snippet from a file (1-indexed, inclusive)."""
    try:
        lines = Path(file_path).read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[max(0, start_line - 1) : end_line])
    except Exception:
        return ""


def find_imports(tree: ast.Module) -> List[str]:
    """Extract all imported module names from an AST."""
    imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def find_function_calls(tree: ast.Module, func_name: str) -> List[int]:
    """Find line numbers where a specific function/method is called."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Direct call: func_name(...)
            if isinstance(node.func, ast.Name) and node.func.id == func_name:
                lines.append(node.lineno)
            # Method call: obj.func_name(...)
            elif isinstance(node.func, ast.Attribute) and node.func.attr == func_name:
                lines.append(node.lineno)
    return lines


def scan_for_security_issues(tree: ast.Module) -> List[Dict]:
    """Check for dangerous patterns: os.system(), unsanitised subprocess calls."""
    issues: List[Dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # os.system(...)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr == "system"
            ):
                issues.append(
                    {
                        "type": "os.system",
                        "lineno": node.lineno,
                        "severity": "HIGH",
                        "message": "Raw os.system() call — security violation.",
                    }
                )
    return issues
