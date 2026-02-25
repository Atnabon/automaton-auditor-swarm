# Automaton Auditor Swarm

An agentic swarm built on **LangGraph** that audits GitHub repositories and PDF reports using a dialectical judicial framework. Three detective agents collect forensic evidence in parallel, three judge personas deliberate with conflicting philosophies, and a Chief Justice synthesises the final verdict using deterministic rules.

## Architecture

```
START
  │
  ├──► RepoInvestigator (code detective)  ──┐
  │                                          │
  └──► DocAnalyst (document detective)     ──┤
                                             │
                          EvidenceAggregator  (fan-in sync)
                                             │
                   ┌─────────────────────────┤
                   │            │             │
              Prosecutor    Defense      TechLead     ← fan-out (TODO)
                   │            │             │
                   └─────────────────────────┤
                                             │
                                    ChiefJustice (TODO)
                                             │
                                            END
```

### Key Design Decisions

- **Pydantic over dicts** — `Evidence` and `JudicialOpinion` are strict `BaseModel` classes with typed fields, ensuring validation at every boundary instead of brittle nested dicts.
- **Annotated reducers** — `AgentState` uses `Annotated[Dict, operator.ior]` and `Annotated[List, operator.add]` so parallel agents merge state safely without overwrites.
- **AST parsing over regex** — Code analysis uses Python's `ast` module to extract class definitions, imports, and graph structure with precision, not fragile regex patterns.
- **Sandboxed cloning** — All git operations run inside `tempfile.TemporaryDirectory()` using `subprocess.run()` with full error handling. No `os.system()` calls.

## Project Structure

```
automaton-auditor-swarm/
├── main.py                     # CLI entry point
├── pyproject.toml              # Dependencies (managed via uv)
├── uv.lock                     # Locked dependency versions for reproducible installs
├── .env.example                # Required environment variables (copy to .env)
├── rubric/
│   └── week2_rubric.json       # Machine-readable evaluation rubric
├── src/
│   ├── state.py                # Pydantic/TypedDict state definitions with Annotated reducers
│   ├── graph.py                # StateGraph with fan-out/fan-in, conditional edges, checkpointing
│   ├── nodes/
│   │   ├── detectives.py       # RepoInvestigator & DocAnalyst nodes
│   │   ├── judges.py           # Prosecutor, Defense, TechLead (stub — final submission)
│   │   └── justice.py          # ChiefJustice synthesis (stub — final submission)
│   └── tools/
│       ├── repo_tools.py       # Sandboxed git clone, AST analysis, security scanning
│       └── doc_tools.py        # PDF ingestion, paragraph chunking, RAG-lite query
└── reports/
    └── interim_report.pdf      # Interim architectural report
```

## Setup

### Prerequisites

- **Python 3.11+** (check with `python --version`)
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Git CLI (`git --version`)
- **[Ollama](https://ollama.ai/)** running locally
- MiniMax M2.5 model pulled: `ollama pull minimax-m2.5:cloud`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/atnabon/automaton-auditor-swarm.git
cd automaton-auditor-swarm

# 2. Install all dependencies from the lock file (exact versions, reproducible)
uv sync

# 3. Configure environment variables
cp .env.example .env
# Open .env in your editor and fill in GITHUB_TOKEN, LANGCHAIN_API_KEY, etc.
```

> **Tip:** `uv sync` reads `uv.lock` to install the exact pinned dependency versions,
> ensuring the same environment on every machine.  For a plain pip install (no lock):
> `pip install -e .`

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OLLAMA_BASE_URL` | No | Ollama API URL (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | No | Ollama model name (default: `minimax2.5`) |
| `LANGCHAIN_TRACING_V2` | No | Set to `true` for LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | LangSmith project name |
| `GITHUB_TOKEN` | No | GitHub PAT for private repositories |

## Usage

### Run the Detective Graph

```bash
# Audit a public repository (detective phase only)
python main.py https://github.com/user/target-repo

# With a PDF report
python main.py https://github.com/user/target-repo --pdf reports/their_report.pdf

# Verbose output
python main.py https://github.com/user/target-repo --pdf report.pdf -v
```

### Example Output

```
🔍 Automaton Auditor Swarm — Detective Phase
   Target repo : https://github.com/user/target-repo
   PDF report  : reports/their_report.pdf

📋 Evidence Summary (7 items):

  ✅ git_forensic_analysis
     Location   : git log
     Confidence : 95%
     Preview    : ["abc1234 2025-02-20T10:00:00Z Initial project setup", ...]

  ✅ state_management_rigor
     Location   : src/state.py
     Confidence : 90%
     Preview    : Pydantic BaseModel classes: ['Evidence', 'JudicialOpinion']...

  ✅ graph_orchestration
     Location   : src/graph.py
     Confidence : 85%
     Preview    : Nodes: ['repo_investigator', 'doc_analyst', ...]...

✅ Detective phase complete.
```

## Current Status (Interim)

### Implemented ✅
- `src/state.py` — Full Pydantic/TypedDict state definitions with Annotated reducers
- `src/tools/repo_tools.py` — Sandboxed git clone, git log extraction, AST-based analysis
- `src/tools/doc_tools.py` — PDF ingestion, keyword search, path extraction
- `src/nodes/detectives.py` — RepoInvestigator and DocAnalyst as LangGraph nodes
- `src/graph.py` — StateGraph with detective fan-out/fan-in and checkpointing
- `rubric/week2_rubric.json` — Full machine-readable rubric

### Planned for Final Submission 🔜
- `src/nodes/judges.py` — Three parallel judge personas (Prosecutor, Defense, TechLead)
- `src/nodes/justice.py` — ChiefJustice with deterministic synthesis rules
- VisionInspector detective for diagram analysis
- Conditional edges for error handling
- Full Markdown report rendering
- LangSmith trace integration

## License

MIT
