# Agent Orchestration System

A production-grade multi-agent orchestration framework built on **LangGraph**, featuring human-in-the-loop review, long-term vector memory, full observability, and PostgreSQL-backed persistence.

---

## Overview

This system decomposes complex user tasks into parallel subtask pipelines executed by specialist AI agents. Each task travels through a stateful LangGraph workflow — from memory retrieval and planning through parallel execution, automated review, synthesis, and memory consolidation — with optional human approval at any stage.

```
User Request
     │
     ▼
retrieve_memory ──► plan ──► [await_human?] ──► execute_subtasks (parallel)
                                                        │
                                               ┌────── ▼ ──────┐
                                               │    review     │
                                               │  [retry loop] │
                                               └──────┬────────┘
                                                      │
                                               synthesize ──► write_memory ──► END
```

---

## Architecture

### Agent Hierarchy

| Agent | Model | Role |
|---|---|---|
| **Supervisor** | `claude-opus-4-6` | Decomposes tasks into subtask plans with confidence scoring |
| **Research** | `claude-sonnet-4-6` | Web search, API lookups, fact-finding |
| **Analysis** | `claude-sonnet-4-6` | Data processing, structured extraction |
| **Writing** | `claude-sonnet-4-6` | Document drafting, summaries, reports |
| **Code** | `claude-sonnet-4-6` | Code generation, debugging, execution |
| **Reviewer** | `claude-haiku-4-5` | Quality gate — approves or triggers retries |

### Key Components

- **LangGraph Workflow** — 7-node `StateGraph` with `MemorySaver` / `PostgresSaver` checkpointer; true graph suspension via `interrupt()` + `Command(resume=...)`
- **HITL (Human-in-the-Loop)** — Redis-backed review queue, 4 decision paths (`approved` / `modified` / `rejected` / `take_over`), auto-escalation on low confidence
- **Long-term Memory** — ChromaDB vector store with cosine similarity retrieval, relevance scoring, and LLM-driven consolidation
- **Observability** — In-process snapshot store, per-agent span tracking, cost tracker with alert thresholds, step-through replay system
- **Persistence** — SQLAlchemy + PostgreSQL task store with graceful in-memory fallback; Alembic migrations
- **API** — FastAPI with request-ID middleware, `/health/live` + `/health/ready` probes, rate limiting, structured logging

---

## Project Structure

```
agent-orchestration/
├── orchestrator/
│   ├── agents/
│   │   ├── base.py              # BaseAgent, AgentResult, timed_invoke(), run_tool_loop()
│   │   ├── supervisor.py        # Task decomposition → ExecutionPlan
│   │   ├── reviewer.py          # Quality scoring → ReviewDecision
│   │   └── specialists/
│   │       ├── research.py      # Web search + API tools
│   │       ├── analysis.py      # Data analysis
│   │       ├── writing.py       # Document generation
│   │       └── code.py          # Code execution
│   ├── graph/
│   │   ├── state.py             # OrchestratorState TypedDict
│   │   ├── nodes.py             # All 7 LangGraph node functions
│   │   ├── edges.py             # Conditional routing logic
│   │   └── workflow.py          # StateGraph assembly + compiled_graph
│   ├── memory/
│   │   ├── manager.py           # retrieve() + store() + consolidate()
│   │   ├── semantic.py          # ChromaDB client with relevance scoring
│   │   └── working.py           # In-flight context buffer
│   ├── hitl/
│   │   ├── queue.py             # Redis-backed HITL review queue
│   │   └── escalation.py       # Escalation rules engine
│   ├── observability/
│   │   ├── snapshot_store.py    # In-process snapshot + span store
│   │   ├── cost_tracker.py      # Token → USD, per-task cost alerts
│   │   ├── replay.py            # Step-through replay session
│   │   └── tracer.py            # OpenTelemetry setup
│   ├── db/
│   │   ├── engine.py            # SQLAlchemy engine, init_db(), graceful fallback
│   │   ├── models.py            # TaskRecord ORM model
│   │   └── task_store.py        # CRUD with in-memory fallback
│   ├── api/
│   │   └── routes/
│   │       ├── tasks.py         # POST /tasks, GET /tasks, GET /tasks/{id}
│   │       ├── review.py        # POST /review/{id}/decide — resume suspended graph
│   │       ├── traces.py        # GET /traces/{id}/summary|spans|snapshots|diff
│   │       └── memory.py        # GET /memory
│   ├── tools/
│   │   ├── registry.py          # Tool registry
│   │   ├── web_search.py        # DuckDuckGo search wrapper
│   │   ├── api_client.py        # Generic REST API client tool
│   │   ├── code_executor.py     # Sandboxed code execution
│   │   └── file_ops.py          # File read/write tools
│   ├── config.py                # Pydantic Settings (all env vars)
│   └── main.py                  # FastAPI app, lifespan, middleware, health probes
├── ui/
│   ├── review/app.py            # Streamlit HITL review UI (port 8501)
│   └── traces/app.py            # Streamlit trace explorer UI (port 8502)
├── workers/
│   └── celery_app.py            # Celery worker configuration
├── tests/
│   ├── test_agents.py           # Agent unit tests
│   ├── test_hitl.py             # HITL flow tests (20 tests)
│   ├── test_memory.py           # Memory system tests (8 tests)
│   ├── test_observability.py    # Observability tests (17 tests)
│   ├── test_integration.py      # End-to-end pipeline tests (10 tests)
│   └── test_persistence.py      # DB layer tests (15 tests)
├── alembic/                     # Database migrations
├── docker-compose.yml           # Full production stack
├── Dockerfile                   # Multi-stage build
├── requirements.txt
├── pytest.ini
└── demo.py                      # CLI demo (runs without API key)
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Anthropic API key

### Local Development (without Docker)

```bash
git clone https://github.com/chandewardnyanesh/agent-orchestration.git
cd agent-orchestration

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY

# Start infrastructure (Postgres + Redis + ChromaDB)
docker compose up postgres redis chromadb -d

# Run migrations
alembic upgrade head

# Start the API
uvicorn orchestrator.main:app --reload

# Optional: start the UIs in separate terminals
streamlit run ui/review/app.py   # port 8501
streamlit run ui/traces/app.py  # port 8502
```

### Full Stack with Docker

```bash
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY

docker compose up --build
```

Services:

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| HITL Review UI | http://localhost:8501 |
| Trace Explorer UI | http://localhost:8502 |
| ChromaDB | http://localhost:8003 |

### Run the Demo (no API key needed)

```bash
python demo.py
```

Runs two full pipeline executions with mocked LLM responses, shows the node-by-node snapshot timeline, final output, and memory context injection on the second run.

---

## API Reference

### Submit a Task

```bash
curl -X POST http://localhost:8000/tasks/ \
  -H "Content-Type: application/json" \
  -d '{"user_request": "Research the top 3 Python web frameworks and write a comparison"}'
```

Response:
```json
{"task_id": "abc-123", "status": "pending", "message": "Task queued."}
```

### Poll Task Status

```bash
curl http://localhost:8000/tasks/abc-123
```

### Submit with Human Review Required

```bash
curl -X POST http://localhost:8000/tasks/ \
  -H "Content-Type: application/json" \
  -d '{"user_request": "Analyse our Q3 sales data", "require_human_review": true}'
```

When the task reaches `awaiting_human_approval`, open the Review UI at `http://localhost:8501` or POST directly:

```bash
curl -X POST http://localhost:8000/review/abc-123/decide \
  -H "Content-Type: application/json" \
  -d '{"decision": "approved", "notes": ""}'
```

Valid decisions: `approved` | `modified` | `rejected` | `take_over`

### View Trace

```bash
# Cost + token summary
curl http://localhost:8000/traces/abc-123/summary

# Per-agent spans
curl http://localhost:8000/traces/abc-123/spans

# Node-by-node snapshots
curl http://localhost:8000/traces/abc-123/snapshots

# Diff between two execution steps
curl "http://localhost:8000/traces/abc-123/diff?step_a=1&step_b=3"
```

### List Tasks

```bash
curl "http://localhost:8000/tasks/?limit=20&offset=0"
```

### Health Probes

```bash
curl http://localhost:8000/health/live    # liveness
curl http://localhost:8000/health/ready  # readiness (checks Postgres + Redis + ChromaDB)
```

---

## HITL — Human-in-the-Loop

The system escalates to human review under three conditions:

1. **Manual flag** — caller sets `require_human_review: true`
2. **Low confidence** — Supervisor confidence < `SUPERVISOR_CONFIDENCE_THRESHOLD` (default 0.70)
3. **Retry exhausted** — Specialist fails `MAX_SPECIALIST_RETRIES` times on the same subtask

When escalated, the LangGraph graph **truly suspends** (via `interrupt()`) and stores its full checkpoint in the checkpointer. The `awaiting_human_approval` status is written to the task store. The reviewer posts a decision which resumes the graph from the exact suspension point via `Command(resume=...)`.

**Rejection re-plan loop**: if the decision is `rejected`, the graph routes back to the `plan` node, injecting the reviewer's notes into the Supervisor prompt so the revised plan addresses the concerns.

---

## Observability

Every graph node saves a state snapshot after executing. Every agent call is wrapped by `timed_invoke()` which records:

- Wall-clock latency (ms)
- Token usage
- USD cost (model-specific rate table)
- Confidence score

Access all of this via the Trace Explorer UI or the `/traces/{task_id}/*` API endpoints.

Cost alerts fire (log WARNING) when a task exceeds `COST_ALERT_THRESHOLD_USD` (default $1.00).

The **replay system** lets you step through any past execution node-by-node and diff state between steps — useful for debugging unexpected plan decisions or reviewing what the agents saw at each stage.

---

## Configuration

All settings are environment variables. Copy `.env.example` and fill in your values.

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** Anthropic API key |
| `SUPERVISOR_MODEL` | `claude-opus-4-6` | Model for task planning |
| `SPECIALIST_MODEL` | `claude-sonnet-4-6` | Model for specialist agents |
| `REVIEWER_MODEL` | `claude-haiku-4-5-20251001` | Model for quality review |
| `SUPERVISOR_CONFIDENCE_THRESHOLD` | `0.7` | Below this → auto-HITL escalation |
| `REVIEWER_QUALITY_THRESHOLD` | `0.75` | Below this → trigger specialist retry |
| `MAX_SPECIALIST_RETRIES` | `2` | Max retries before escalating to human |
| `DATABASE_URL` | `postgresql://...` | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `CHROMA_HOST` | `localhost` | ChromaDB host |
| `COST_ALERT_THRESHOLD_USD` | `1.0` | Alert threshold per task |
| `CODE_EXECUTOR_SANDBOX` | `true` | Enable sandboxed code execution |

---

## Running Tests

```bash
# All tests (no infrastructure required)
pytest tests/

# Specific phase
pytest tests/test_integration.py -v

# Include live ChromaDB test
pytest tests/ -m integration
```

74 tests pass, 1 skipped (ChromaDB live roundtrip — requires a running ChromaDB instance).

---

## Database Migrations

```bash
# Apply all migrations
alembic upgrade head

# Create a new migration after model changes
alembic revision --autogenerate -m "describe the change"

# Check current revision
alembic current
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph 0.6 |
| LLM | Anthropic (Claude) |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL + SQLAlchemy + Alembic |
| Queue | Redis + Celery |
| Memory | ChromaDB (vector store) |
| Observability | OpenTelemetry + in-process span store |
| UI | Streamlit |
| Containerisation | Docker + Docker Compose |
| Tests | pytest |

---

## Limitations & Roadmap

- **Authentication** — API endpoints are currently open. Bearer token validation is planned.
- **Analysis & Code agents** — Currently invoke the LLM without bound tools. Tool integration (code execution sandbox, data analysis libraries) is in progress.
- **Celery** — Infrastructure is wired but task execution uses FastAPI `BackgroundTasks`. Migrating to Celery for true horizontal scaling is the next step.
- **Memory consolidation throttle** — LLM-driven consolidation runs at a fixed cluster threshold; a time-based circuit-breaker is planned to avoid excessive calls on high-throughput deployments.

---

## License

MIT
