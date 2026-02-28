# Automaton Auditor Swarm

An agentic swarm built on **LangGraph** that audits GitHub repositories and PDF reports using a dialectical judicial framework. Three detective agents collect forensic evidence in parallel, three judge personas deliberate with conflicting philosophies, and a Chief Justice synthesises the final verdict using deterministic rules.

## Architecture

```
START
  │
  ├──► RepoInvestigator  ──┐
  │                         │
  ├──► DocAnalyst         ──┤
  │                         │
  └──► VisionInspector    ──┤
                            │
               EvidenceAggregator  (fan-in sync #1)
                            │
               [conditional: error? insufficient evidence?]
                     ├── yes ──► END
                     └── no  ──► Judicial Fan-Out
                                   │
                ┌──────────────────┼──────────────────┐
                │                  │                   │
           Prosecutor          Defense            TechLead
                │                  │                   │
                └──────────────────┼──────────────────┘
                                   │
                        JudicialSynchronizer  (fan-in sync #2)
                                   │
                        [conditional: no opinions?]
                              ├── yes ──► END
                              └── no  ──► ChiefJustice
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
│   ├── graph.py                # StateGraph with 2x fan-out/fan-in, conditional edges, checkpointing
│   ├── nodes/
│   │   ├── detectives.py       # RepoInvestigator, DocAnalyst, VisionInspector, EvidenceAggregator
│   │   ├── judges.py           # Prosecutor, Defense, TechLead with .with_structured_output()
│   │   └── justice.py          # ChiefJustice deterministic synthesis with 4 named rules
│   └── tools/
│       ├── repo_tools.py       # Sandboxed git clone, AST analysis, security scanning
│       └── doc_tools.py        # PDF ingestion, paragraph chunking, RAG-lite query
└── reports/
    ├── interim_report.md       # Final architectural report
    ├── self_audit_report.md    # Self-audit: agent auditing its own repo
    ├── peer_audit_report.md    # Peer-audit: our agent auditing peer's repo
    ├── peer_received_report.md # Peer-received: peer's agent auditing our repo
    ├── audit_report.md         # Runtime-generated report (produced by ChiefJustice)
    └── video_script.md         # Demo video script (~3 min)
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

### Run the Full Auditor Pipeline

```bash
# Audit a public repository (full pipeline: detectives → judges → synthesis)
python main.py https://github.com/user/target-repo

# With a PDF report for document analysis
python main.py https://github.com/user/target-repo --pdf reports/their_report.pdf

# Verbose output (debug logging)
python main.py https://github.com/user/target-repo --pdf report.pdf -v
```

### Example Output

```
======================================================================
  AUTOMATON AUDITOR SWARM — Full Pipeline Execution
======================================================================
  Target repo : https://github.com/user/target-repo
  PDF report  : reports/their_report.pdf
  LangSmith   : ENABLED
  Project     : automaton-auditor-swarm
======================================================================

  DETECTIVE LAYER — Evidence Summary (8 items)

  ✅ git_forensic_analysis [FOUND]
     Detective   : RepoInvestigator
     Confidence  : 95%

  ✅ state_management_rigor [FOUND]
     Detective   : RepoInvestigator
     Confidence  : 95%

  JUDICIAL LAYER — Judge Opinions (30 opinions)

  --- graph_orchestration ---
    Prosecutor   | Score: 4/5 | Two fan-out/fan-in patterns verified...
    Defense      | Score: 5/5 | Full parallel execution confirmed...
    TechLead     | Score: 5/5 | Graph topology is correct...

  CHIEF JUSTICE — Final Audit Report

  Overall Result: 43/50 (4.3/5.0 average)

  Per-Criterion Scores:
    git_forensic_analysis                    : 4/5
    state_management_rigor                   : 5/5
    graph_orchestration                      : 5/5 [DISSENT]
    ...

  📄 Full report written to: reports/audit_report.md
  🔗 LangSmith trace: https://smith.langchain.com/...

  ✅ Automaton Auditor Swarm — Pipeline Complete
```

### Generated Reports

The pipeline produces the following report artifacts in `reports/`:

| Report | Description |
|--------|-------------|
| `audit_report.md` | Runtime-generated report with executive summary, criterion breakdown, dissent summaries, remediation plan |
| `self_audit_report.md` | Self-audit — agent run against its own repository |
| `peer_audit_report.md` | Peer-audit — our agent's findings from auditing a peer's repo |
| `peer_received_report.md` | Peer-received — report from a peer's agent auditing our repo |

## LangSmith Tracing

Set `LANGCHAIN_API_KEY` in your `.env` file to enable automatic tracing. All pipeline runs are traced to the `automaton-auditor-swarm` project in LangSmith. The trace shows all nodes: detectives (fan-out), evidence aggregation (fan-in), judges (fan-out), judicial sync (fan-in), and chief justice synthesis.

## License

MIT
