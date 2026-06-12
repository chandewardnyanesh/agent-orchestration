# Architecture — Agent Orchestration System

This document describes the internal design of the system in detail: how data flows, how components communicate, why specific design decisions were made, and where the extension points are.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [LangGraph Workflow](#2-langgraph-workflow)
3. [State Machine](#3-state-machine)
4. [Agent Design](#4-agent-design)
5. [Human-in-the-Loop (HITL)](#5-human-in-the-loop-hitl)
6. [Memory System](#6-memory-system)
7. [Observability Pipeline](#7-observability-pipeline)
8. [Persistence Layer](#8-persistence-layer)
9. [API Layer](#9-api-layer)
10. [Infrastructure](#10-infrastructure)
11. [Data Flow Walkthrough](#11-data-flow-walkthrough)
12. [Design Decisions](#12-design-decisions)
13. [Extension Points](#13-extension-points)

---

## 1. System Overview

The system is built around a single core idea: **structured task decomposition with quality gates**.

A user submits a natural-language request. A Supervisor agent decomposes it into an ordered list of subtasks, each routed to a domain specialist. Each specialist output passes through a Reviewer before contributing to the final synthesised response. Long-term memory means every completed task makes the system smarter for future related requests.

```
                         ┌─────────────────────────────┐
                         │         FastAPI              │
                         │  POST /tasks  GET /tasks     │
                         │  POST /review/{id}/decide    │
                         └──────────────┬──────────────┘
                                        │ BackgroundTask
                                        ▼
                         ┌─────────────────────────────┐
                         │       LangGraph Graph        │
                         │                             │
                         │  retrieve_memory            │
                         │       ↓                     │
                         │     plan ←──────────────────┼─── re-plan on rejection
                         │       ↓                     │
                         │  [await_human?]             │
                         │       ↓                     │
                         │  execute_subtasks           │
                         │   (parallel fan-out)        │
                         │       ↓                     │
                         │    review ──→ [retry?]      │
                         │       ↓                     │
                         │   synthesize                │
                         │       ↓                     │
                         │  write_memory               │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────┼──────────────┐
                         │             │               │
                    PostgreSQL       Redis          ChromaDB
                  (task store)   (HITL queue)    (vector memory)
```

---

## 2. LangGraph Workflow

The orchestration graph is a `StateGraph[OrchestratorState]` compiled with a checkpointer (PostgresSaver when available, MemorySaver otherwise).

### Node Topology

```
START
  │
  ▼
retrieve_memory
  │  (always)
  ▼
plan
  │
  ├─── [hitl_required OR confidence < threshold] ──► await_human
  │                                                       │
  │           ┌──────── rejected ─────────────────────────┘
  │           │         (re-plan loop with rejection notes)
  │           ▼
  │         plan (second attempt)
  │           │
  │     take_over ──► END
  │           │ approved / modified
  │           ▼
  └──────────────────────────────────────► execute_subtasks
                                               │  (parallel wave execution)
                                               ▼
                                            review
                                               │
                                   ┌─── status == "retrying" ──► execute_subtasks (retry loop)
                                   │
                                   ├─── retries exhausted ──► await_human (escalate)
                                   │
                                   └─── all approved ──► synthesize
                                                             │
                                                             ▼
                                                         write_memory
                                                             │
                                                             ▼
                                                            END
```

### Edges

All routing is handled by three pure functions in `orchestrator/graph/edges.py`:

| Function | Source Node | Routing Logic |
|---|---|---|
| `after_plan` | `plan` | `hitl_required OR confidence < threshold` → `await_human`, else → `execute_subtasks` |
| `after_await_human` | `await_human` | `rejected` → `plan`, `take_over` → `END`, else → `execute_subtasks` |
| `after_review` | `review` | `status == "retrying"` + pending subtasks → `execute_subtasks`; exhausted retries → `await_human`; else → `synthesize` |

### Checkpointer

The graph is compiled with `checkpointer=_make_checkpointer()`. This function attempts to instantiate `PostgresSaver` (from `langgraph-checkpoint-postgres`) and falls back to `MemorySaver` if the package is unavailable or the DB is unreachable.

The checkpointer is what makes true graph suspension possible. When `interrupt()` is called inside `await_human`, LangGraph serialises the entire graph state (all node inputs/outputs, edges visited, pending futures) into the checkpointer under `thread_id = task_id`. The graph can then be resumed by passing `Command(resume=payload)` with the same `thread_id`.

---

## 3. State Machine

`OrchestratorState` is a `TypedDict` — every node receives a copy and returns a partial dict that LangGraph merges in.

```python
class OrchestratorState(TypedDict):
    # Identification
    task_id:               str
    user_request:          str
    trace_id:              str

    # Pipeline status
    status:                TaskStatus   # see below
    error:                 Optional[str]

    # Planning
    execution_plan:        Optional[dict]
    supervisor_confidence: float

    # Execution
    subtasks:              list[SubTaskState]
    completed_outputs:     dict[str, str]   # subtask_id → output

    # Memory
    memory_context:        str
    memory_written:        bool

    # Human-in-the-loop
    hitl_required:         bool
    hitl_level:            Optional[Literal["notify","approve_action","approve_plan","take_over"]]
    hitl_decision:         Optional[Literal["approved","rejected","modified","take_over"]]
    hitl_notes:            Optional[str]

    # Output
    final_output:          Optional[str]

    # Observability
    total_tokens:          int
    total_cost_usd:        float
```

### Task Status Lifecycle

```
pending
  │
  ▼
planning  (retrieve_memory + plan nodes running)
  │
  ├──► awaiting_human_approval  (graph suspended at await_human)
  │         │
  │    [human decides]
  │         │
  ├──► executing   (execute_subtasks running)
  │
  ▼
reviewing  (review node running)
  │
  ├──► retrying   (one or more subtasks re-queued)
  │
  ▼
synthesizing
  │
  ▼
completed / failed
```

### SubTaskState

Each element in `state["subtasks"]` carries its own mini state machine:

```
pending → in_progress → completed
                    └──► failed (after max retries)
```

The `attempt` counter increments each time the specialist re-runs. When `attempt >= max_specialist_retries` and the reviewer still rejects, the subtask status becomes `failed` and the graph routes to `await_human` for human escalation.

---

## 4. Agent Design

All agents extend `BaseAgent` in `orchestrator/agents/base.py`.

### BaseAgent

```python
class BaseAgent:
    name:  str       # used as span label
    model: str       # Anthropic model ID
    llm:   ChatAnthropic

    def invoke(self, inputs: dict) -> AgentResult:
        ...  # implemented by each subclass

    def timed_invoke(self, inputs: dict) -> AgentResult:
        # 1. Record wall-clock start time
        # 2. Wrap invoke() in an OpenTelemetry span (no-op if OTLP unavailable)
        # 3. Store span in in-process snapshot_store (always works)
        # 4. Record cost via CostTracker (Redis-backed with dict fallback)
        # 5. Return AgentResult with latency_ms populated
```

`timed_invoke()` is the single instrument point — all observability happens here. Nodes always call `timed_invoke()` rather than `invoke()` directly.

### AgentResult

```python
@dataclass
class AgentResult:
    output:     Any
    confidence: float        # 0–1
    tokens_used: int
    tool_calls:  list[dict]  # populated by run_tool_loop()
    latency_ms:  float       # filled by timed_invoke()
    error:       Optional[str]
```

### Supervisor Agent

The Supervisor is the only agent that calls the largest model (`claude-opus-4-6` by default). Its responsibilities:

1. Parse the user request + memory context into a structured `ExecutionPlan` (Pydantic model)
2. Assign each subtask a `specialist` type
3. Express `depends_on` relationships to enforce ordering
4. Set `confidence` and flag `requires_human_approval` if confidence < threshold

On a **re-plan** (after rejection), the node injects the reviewer's `hitl_notes` into the prompt:

```
⚠️  PREVIOUS PLAN WAS REJECTED BY A HUMAN REVIEWER.
Rejection reason: <notes>
You MUST address these concerns in your revised plan.
```

### Specialist Agents

| Agent | Tools bound | Primary use |
|---|---|---|
| Research | `web_search`, `api_client` | Fact-finding, current data |
| Analysis | *(planned: data tools)* | Structured extraction, statistics |
| Writing | none (pure generation) | Documents, summaries |
| Code | *(planned: code_executor)* | Code generation and execution |

Specialists that use tools go through `run_tool_loop()` in `base.py`, which handles the multi-turn tool-call cycle (LLM calls → tool execution → result injection → next LLM call) until the model stops issuing tool calls or the loop limit is reached.

### Reviewer Agent

The Reviewer uses the cheapest model (`claude-haiku-4-5`) and evaluates every specialist output against three axes:

- **Completeness** — did it cover what was asked?
- **Accuracy** — is the content well-reasoned?
- **Format** — does it match the `expected_output_format`?

It returns a `ReviewDecision` with `approved: bool`, `quality_score: float`, and structured `feedback`. The Supervisor's threshold (`reviewer_quality_threshold`, default 0.75) determines whether a subtask is approved or retried.

---

## 5. Human-in-the-Loop (HITL)

### Suspension Mechanism

LangGraph's `interrupt()` function is called inside the `await_human` node. When hit:

1. The graph serialises its entire checkpoint to the checkpointer under `thread_id = task_id`
2. LangGraph returns the partial state (with `__interrupt__` key) to the caller instead of continuing
3. The API detects `__interrupt__` and marks the task `awaiting_human_approval` in the task store
4. The graph execution thread exits — no polling, no blocking

### Resume Mechanism

When a reviewer POSTs to `/review/{task_id}/decide`:

```
POST /review/{task_id}/decide
{"decision": "approved", "notes": ""}
```

The API runs `compiled_graph.invoke(Command(resume=payload), config=thread_config)` in a background thread. LangGraph:

1. Loads the checkpoint from the checkpointer
2. Resumes execution inside `await_human` with `decision_payload = payload`
3. The node returns `{"hitl_decision": "approved", "hitl_notes": ""}` and the graph continues

### Escalation Rules (`orchestrator/hitl/escalation.py`)

```
Priority  Condition                               Level
────────  ──────────────────────────────────────  ────────────────
1         User requested review                   approve_action
2         Sensitive operation (send email, etc.)  approve_action
3         Specialist retries exhausted            take_over
4         Reviewer quality score < threshold      approve_action
5         Supervisor confidence < threshold       approve_plan
```

Auto-escalation (rule 5) fires even when the user did not request review — the system protects itself from executing low-confidence plans silently.

### HITL Queue (Redis)

```
Key schema:
  hitl:pending          → Redis LIST  (LPUSH to enqueue, RPOP to dequeue)
  hitl:decision:{id}    → Redis STRING (expires 1h, set on POST /decide)
```

The Review UI polls `GET /review/pending` which calls `HITLQueue.list_pending()`. All Redis operations fall back silently when Redis is unavailable.

---

## 6. Memory System

### Architecture

```
MemoryManager
    │
    ├── WorkingMemory   (Redis, TTL 1h per task_id)
    │   Key: working:{task_id}:{key}
    │
    └── SemanticMemory  (ChromaDB)
        Collection: agent_memory
        Embeddings: chromadb default (sentence-transformers)
```

### Retrieval

`MemoryManager.retrieve(query, top_k=3)` runs a cosine similarity search over ChromaDB and returns a formatted context string for the Supervisor:

```
[Memory 1] relevance=0.82  score=0.741  specialists=[research, writing]
Task: Research the top 3 Python web frameworks...
Subtasks: ['Research frameworks', 'Write comparison']
Outcome: FastAPI, Django, Flask comparison completed successfully.

[Memory 2] ...
```

The `relevance_score` stored in metadata increases each time that memory is retrieved and the resulting task succeeds — memories that prove repeatedly useful are surfaced higher.

### Storage

`MemoryManager.store()` is called by `write_memory` after every completed task. It writes:

- The full task narrative (request, subtasks, specialists, outcome)
- `quality_score` derived from reviewer approval rates
- `success` flag
- Initial `relevance_score` seeded from quality (range 0.5–0.8)

### Consolidation

When a specialist-type cluster reaches 5+ memories, `_maybe_consolidate()` uses a cheap LLM call to distil all of them into a single high-level summary stored as a `consolidated:` entry with `relevance_score=0.85`. The originals are demoted to `relevance_score=0.2` so the consolidation wins future retrievals. This prevents context flooding in high-volume deployments.

---

## 7. Observability Pipeline

### Three-Layer Design

```
Layer 1: OpenTelemetry (optional — requires OTLP collector)
    Spans exported to Jaeger/Zipkin when OTLP endpoint is reachable.
    All OTel calls are wrapped in try/except — if no collector is running,
    the span is a no-op and execution continues normally.

Layer 2: In-process Snapshot Store (always on)
    _snapshots: dict[task_id, list[Snapshot]]
    _spans:     dict[task_id, list[Span]]
    Written to by every node (_snap helper) and every timed_invoke() call.
    Accessible instantly via /traces/* endpoints.

Layer 3: CostTracker (Redis-backed, dict fallback)
    Every timed_invoke() call records (agent, model, tokens, USD).
    Aggregates available via CostTracker.total_for_task().
    Fires a WARNING log if total_usd > cost_alert_threshold_usd.
```

### Snapshot

After every graph node returns, `_snap(node_name, state, result)` saves a lightweight state slice:

```json
{
    "step": 2,
    "node": "execute_subtasks",
    "recorded_at": 1718000000.0,
    "state": {
        "status": "reviewing",
        "supervisor_confidence": 0.91,
        "total_tokens": 680,
        "hitl_decision": null,
        "subtasks": [
            {"id": "st-1", "specialist": "research", "status": "completed", "review_approved": null, "attempt": 1}
        ]
    }
}
```

### Replay System

`ReplaySession(task_id)` loads all snapshots from the in-process store and exposes:

| Method | Behaviour |
|---|---|
| `step_forward()` | Return next snapshot, advance internal cursor |
| `jump_to(n)` | Return snapshot at step n, cursor unchanged |
| `compare(a, b)` | Diff two steps — returns `{field: {step_a: val, step_b: val}}` |
| `all_steps()` | Return every snapshot in order |

The diff output powers the "Step Diff Viewer" in the Trace Explorer UI.

---

## 8. Persistence Layer

### Task Store

```
orchestrator/db/
    engine.py     — SQLAlchemy engine, init_db(), graceful fallback flag
    models.py     — TaskRecord ORM model
    task_store.py — CRUD (create/get/update/finish/list)
```

Every public function in `task_store.py` checks `is_pg_available()` first. If False, it operates on `_mem: dict[str, dict]` — the in-memory fallback that mirrors the DB schema exactly. This means the API works identically whether Postgres is running or not; Postgres just adds durability and survives server restarts.

### TaskRecord Schema

```sql
CREATE TABLE tasks (
    task_id             VARCHAR(64)  PRIMARY KEY,
    status              VARCHAR(32)  NOT NULL DEFAULT 'pending',
    user_request        TEXT         NOT NULL,
    supervisor_confidence FLOAT      NOT NULL DEFAULT 0.0,
    total_tokens        INTEGER      NOT NULL DEFAULT 0,
    total_cost_usd      FLOAT        NOT NULL DEFAULT 0.0,
    final_output        TEXT,
    subtasks            JSONB        NOT NULL DEFAULT '[]',
    hitl_required       BOOLEAN      NOT NULL DEFAULT false,
    hitl_decision       VARCHAR(32),
    hitl_notes          TEXT,
    memory_written      BOOLEAN      NOT NULL DEFAULT false,
    error               TEXT,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
```

### LangGraph Checkpointer

The graph checkpointer stores the serialised graph state (all node results, edges, pending interrupts) separately from the task store:

```
PostgresSaver  →  langgraph_checkpoints table (managed by the library)
MemorySaver    →  in-process dict (lost on restart, fine for development)
```

The fallback chain at startup:
```python
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    checkpointer = PostgresSaver.from_conn_string(database_url)
    checkpointer.setup()   # creates langgraph_checkpoints table
except Exception:
    checkpointer = MemorySaver()
```

---

## 9. API Layer

### Route Map

```
POST   /tasks/                     Submit a new task
GET    /tasks/                     List tasks (paginated: ?limit=20&offset=0)
GET    /tasks/{task_id}            Get task status and result

GET    /review/pending             List tasks awaiting human review
POST   /review/{task_id}/decide    Submit human decision (resumes suspended graph)

GET    /traces/{task_id}/summary   Aggregated cost + latency
GET    /traces/{task_id}/spans     Per-agent timing records
GET    /traces/{task_id}/snapshots Node-by-node state timeline
GET    /traces/{task_id}/replay/{step}  Single step for incremental replay
POST   /traces/{task_id}/replay    Full step list for replay session
GET    /traces/{task_id}/diff      Diff two steps (?step_a=0&step_b=1)

GET    /memory                     Memory store contents

GET    /health/live                Liveness probe (always 200)
GET    /health/ready               Readiness probe (checks Postgres + Redis + ChromaDB)
GET    /health                     Legacy health endpoint
```

### Execution Model

Tasks run in FastAPI `BackgroundTasks` threads, not in the request/response cycle. `POST /tasks/` returns immediately with the task ID; the caller polls `GET /tasks/{id}` for status updates. This is intentional:

- LLM calls can take 10–30 seconds
- HTTP clients have timeouts
- The review/resume cycle requires the initial request to have already returned

### Middleware Stack

```
Request arrives
    │
    ▼
CORSMiddleware          (allow all origins for development; tighten for production)
    │
    ▼
request_id_middleware   (attaches X-Request-ID header, logs method/path/status/latency)
    │
    ▼
slowapi rate limiter    (100 req/min per IP, graceful no-op if not installed)
    │
    ▼
Route handler
    │
    ▼ (on unhandled exception)
global_exception_handler (returns {detail, request_id} — never leaks stack traces)
```

---

## 10. Infrastructure

### Docker Compose Services

```
┌─────────────────────────────────────────────────────────────┐
│                      orchestration network                   │
│                                                             │
│   api (FastAPI)          review-ui (Streamlit :8501)        │
│   :8000                  trace-ui  (Streamlit :8502)        │
│                                                             │
│   worker (Celery)        migrate (one-shot: alembic upgrade) │
│                                                             │
│   postgres :5432         redis :6379        chromadb :8003  │
└─────────────────────────────────────────────────────────────┘
```

All services share a named bridge network (`orchestration`). Application services use `depends_on: condition: service_healthy` — they will not start until the infrastructure services pass their health checks. This prevents race conditions on cold boot.

### Health Check Chain

```
postgres  → pg_isready
redis     → redis-cli ping
chromadb  → curl /api/v1/heartbeat
api       → curl /health/live
review-ui → (starts after api is healthy)
trace-ui  → (starts after api is healthy)
```

### Dockerfile — Multi-Stage Build

```
Stage 1 (builder):
  - Full build toolchain
  - pip install --prefix=/install all dependencies
  - Result: /install directory with compiled packages

Stage 2 (runtime):
  - Minimal python:3.11-slim (no build tools)
  - COPY --from=builder /install /usr/local
  - Non-root user (uid 1000)
  - PYTHONPATH=/app
  - HEALTHCHECK built into image
```

Final image is ~40% smaller than a single-stage build because the compiler toolchain is discarded.

---

## 11. Data Flow Walkthrough

This traces a complete request through the system — from API call to final response.

### Step 1 — Submit

```
Client → POST /tasks/ {"user_request": "Research top 3 LLM frameworks"}
API    → generates task_id (UUID4)
API    → db.task_store.create_task(task_id, user_request, "pending")
API    → schedules _run_task() as BackgroundTask
API    → returns {"task_id": "abc-123", "status": "pending"}
```

### Step 2 — retrieve_memory

```
Graph starts on thread_id = task_id
Node: retrieve_memory
  → MemoryManager.retrieve("Research top 3 LLM frameworks", top_k=3)
  → ChromaDB cosine similarity query
  → Returns formatted context string (or "No relevant prior context found.")
  → _snap("retrieve_memory", state, result)  saves snapshot step 0
State: status="planning", memory_context="..."
```

### Step 3 — plan

```
Node: plan
  → Supervisor.timed_invoke({task_id, user_request, memory_context})
      → LLM call (claude-opus) with memory context in system prompt
      → Parses ExecutionPlan JSON
      → confidence=0.91 → no HITL needed
  → timed_invoke records span: {agent="supervisor", tokens=420, latency_ms=3200, usd=0.018}
  → _snap("plan", state, result)  saves snapshot step 1
State: subtasks=[{st-1 research}, {st-2 writing}], confidence=0.91, status="executing"
Edge: after_plan → confidence 0.91 ≥ 0.70 → "execute_subtasks"
```

### Step 4 — execute_subtasks

```
Node: execute_subtasks
  Wave 1 (no deps): [st-1 research]
    → ResearchAgent.timed_invoke({task_id, "Research LLM frameworks", context=""})
        → llm.bind_tools([web_search, api_client]).invoke(...)
        → tool loop: web_search("top LLM frameworks 2025") → results
        → llm.invoke(results) → final output
    → span saved: {agent="research", tokens=680, latency_ms=5100}
    → st-1.status="completed", completed_outputs["st-1"] = "LangGraph, CrewAI..."

  Wave 2 (depends on st-1): [st-2 writing]
    → WritingAgent.timed_invoke({task_id, "Write 200-word comparison", context="[st-1]: ..."})
    → span saved: {agent="writing", tokens=290, latency_ms=2200}
    → st-2.status="completed"

  → _snap("execute_subtasks", state, result)  saves snapshot step 2
State: all subtasks "completed", status="reviewing"
```

### Step 5 — review

```
Node: review
  For each subtask:
    → Reviewer.timed_invoke({task_id, description, output, expected_format, attempt=1})
    → ReviewDecision: {approved=True, quality_score=0.92}
    → st-1.review_approved=True, st-2.review_approved=True
  → all_approved=True → next_status="synthesizing"
  → _snap("review", state, result)  saves snapshot step 3
State: all subtasks approved, status="synthesizing"
Edge: after_review → "synthesize"
```

### Step 6 — synthesize

```
Node: synthesize
  → Joins approved subtask outputs with section headers
  → final_output = "### Research...\n{st-1 output}\n\n### Writing...\n{st-2 output}"
  → status="completed"
  → _snap("synthesize", state, result)  saves snapshot step 4
```

### Step 7 — write_memory

```
Node: write_memory
  → quality_score = 2/2 approved = 1.0
  → MemoryManager.store(task_id, user_request, plan, final_output, quality=1.0, success=True)
      → ChromaDB upsert with metadata
      → _maybe_consolidate() — checks cluster sizes
  → CostTracker.check_alert(task_id) — total $0.024 < $1.00 → no alert
  → memory_written=True
  → _snap("write_memory", state, result)  saves snapshot step 5
```

### Step 8 — finish

```
Graph returns final state dict (no __interrupt__ key)
_run_task() detects completion → db.task_store.finish_task(task_id, final)
  → TaskRecord.status="completed", final_output stored, subtasks JSON written

Client polls GET /tasks/abc-123 → receives completed state
```

---

## 12. Design Decisions

### Why LangGraph instead of a custom state machine?

LangGraph provides checkpointing, true graph suspension, and typed state management out of the box. Building the same `interrupt()/Command(resume=...)` pattern from scratch would require a persistent queue, a graph serialiser, and careful thread synchronisation. LangGraph collapses all of that into a few lines.

### Why in-process snapshot store instead of Jaeger/Zipkin?

Requiring developers to run an OTLP collector just to see what their agent did creates unnecessary friction. The in-process dict store works immediately with zero infrastructure and is accessible via the REST API within the same process. OTel export still works when a collector is available — it's additive, not exclusive.

### Why graceful fallback everywhere?

The system has four optional infrastructure dependencies (Postgres, Redis, ChromaDB, OTLP). If any one of them is unavailable, the core task execution still works. This is essential for:

- Running tests without Docker (`pytest` passes with zero external services)
- Local development without a full stack
- Partial infrastructure failures in production degrading gracefully

### Why BackgroundTasks instead of Celery for task execution?

FastAPI `BackgroundTasks` is simpler and sufficient for single-process deployments. Celery is wired and configured — migrating is a one-node change (replace `background.add_task(...)` with `celery_task.delay(...)`). The complexity is deferred until horizontal scaling is needed.

### Why TypedDict for OrchestratorState instead of Pydantic?

LangGraph requires the state schema to be a `TypedDict` or `dataclass`. Pydantic models are not directly compatible with LangGraph's state merging semantics. Domain models within agents (`ExecutionPlan`, `ReviewDecision`) use Pydantic for full validation.

---

## 13. Extension Points

### Adding a new specialist agent

1. Create `orchestrator/agents/specialists/your_specialist.py` extending `BaseAgent`
2. Add an entry to `_specialists` dict in `orchestrator/graph/nodes.py`
3. Add `"your_specialist"` to the `SpecialistType` literal in `supervisor.py`
4. The Supervisor will route subtasks to it automatically

### Adding a new tool

1. Create a LangChain `@tool`-decorated function in `orchestrator/tools/`
2. Register it: `tool_registry.register("tool_name", your_tool)`
3. Bind it to whichever specialist needs it: `self._tools = [tool_registry.get("tool_name")]`

### Replacing the LLM provider

Each agent instantiates its own LLM via `_build_llm(model)` in `base.py`. To swap providers, replace `ChatAnthropic(...)` with any LangChain-compatible chat model (`ChatOpenAI`, `ChatGoogle`, etc.) — nothing else changes.

### Adding a new graph node

1. Write a node function `def my_node(state: OrchestratorState) -> dict:` in `nodes.py`, ending with `return _snap("my_node", state, result)`
2. Add it to `build_graph()` in `workflow.py`: `g.add_node("my_node", my_node)`
3. Wire edges: `g.add_edge("predecessor", "my_node")` and `g.add_edge("my_node", "successor")`

### Upgrading snapshot store to PostgreSQL

Replace the `_snapshots` and `_spans` dicts in `snapshot_store.py` with SQLAlchemy writes. The public API (`save_snapshot`, `get_snapshots`, `save_span`, `get_spans`, `task_summary`) is the only interface used by the rest of the system — callers do not need to change.
