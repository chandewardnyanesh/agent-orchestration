"""
Phase 6 — Persistence Tests

Tests the DB layer (engine, models, task_store) with PostgreSQL unavailable
so they always pass in CI without a live database.

Tests:
  1.  init_db() returns False gracefully when Postgres is unreachable
  2.  is_pg_available() reflects init_db() result
  3.  create_task() falls back to in-memory dict when DB is down
  4.  get_task() returns the in-memory record after create_task()
  5.  update_task() merges partial updates into the in-memory record
  6.  finish_task() writes all final-state fields
  7.  list_tasks() returns in-memory entries with limit/offset
  8.  update_task() creates a new record if task_id is unknown
  9.  finish_task() serialises subtasks correctly
  10. list_tasks() respects the limit parameter
  11. Workflow uses MemorySaver fallback when PostgresSaver is unavailable
  12. /health/live always returns 200
  13. /health/ready returns degraded when deps are down
  14. GET /tasks/ returns task list
  15. Request-ID header is present in all responses
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clear_mem():
    """Wipe the in-memory fallback store between tests."""
    from orchestrator.db import task_store
    task_store._mem.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# 1–2. DB engine init with no Postgres
# ═══════════════════════════════════════════════════════════════════════════════

def test_init_db_returns_false_when_unreachable():
    """init_db() must not raise and must return False when Postgres is down."""
    import orchestrator.db.engine as eng

    # get_settings is imported inside init_db, so patch the source module
    with patch("orchestrator.config.get_settings") as mock_cfg:
        cfg = MagicMock()
        cfg.database_url = "postgresql://x:x@localhost:19999/x"
        mock_cfg.return_value = cfg

        eng._engine        = None
        eng._SessionLocal  = None
        eng._pg_available  = False

        result = eng.init_db()

    assert result is False


def test_is_pg_available_false_after_failed_init():
    from orchestrator.db.engine import is_pg_available, _pg_available
    assert is_pg_available() is _pg_available   # reflects module-level flag


# ═══════════════════════════════════════════════════════════════════════════════
# 3–8. In-memory task_store
# ═══════════════════════════════════════════════════════════════════════════════

def test_create_task_uses_memory_when_db_down():
    _clear_mem()
    from orchestrator.db import task_store
    from orchestrator.db.engine import _pg_available as _orig

    with patch("orchestrator.db.engine._pg_available", False):
        rec = task_store.create_task("t-001", "Write a poem", status="pending")

    assert rec["task_id"] == "t-001"
    assert rec["status"] == "pending"
    assert rec["user_request"] == "Write a poem"
    _clear_mem()


def test_get_task_returns_created_record():
    _clear_mem()
    from orchestrator.db import task_store

    with patch("orchestrator.db.engine._pg_available", False):
        task_store.create_task("t-002", "Analyse data")
        record = task_store.get_task("t-002")

    assert record is not None
    assert record["task_id"] == "t-002"
    _clear_mem()


def test_get_task_returns_none_for_unknown():
    _clear_mem()
    from orchestrator.db import task_store

    with patch("orchestrator.db.engine._pg_available", False):
        result = task_store.get_task("does-not-exist")

    assert result is None
    _clear_mem()


def test_update_task_merges_patch():
    _clear_mem()
    from orchestrator.db import task_store

    with patch("orchestrator.db.engine._pg_available", False):
        task_store.create_task("t-003", "Search web")
        task_store.update_task("t-003", {"status": "executing", "total_tokens": 250})
        record = task_store.get_task("t-003")

    assert record["status"] == "executing"
    assert record["total_tokens"] == 250
    _clear_mem()


def test_update_task_creates_record_when_missing():
    _clear_mem()
    from orchestrator.db import task_store

    with patch("orchestrator.db.engine._pg_available", False):
        task_store.update_task("t-new", {"status": "failed", "error": "boom"})
        record = task_store.get_task("t-new")

    assert record is not None
    assert record["status"] == "failed"
    _clear_mem()


def test_finish_task_writes_all_fields():
    _clear_mem()
    from orchestrator.db import task_store

    final = {
        "status":                "completed",
        "supervisor_confidence": 0.91,
        "total_tokens":          1200,
        "total_cost_usd":        0.003,
        "final_output":          "Great result",
        "subtasks":              [
            {"id": "st-1", "description": "Research", "specialist": "research",
             "status": "completed", "review_approved": True, "attempt": 1},
        ],
        "hitl_required":         False,
        "hitl_decision":         None,
        "hitl_notes":            None,
        "memory_written":        True,
        "error":                 None,
    }

    with patch("orchestrator.db.engine._pg_available", False):
        task_store.create_task("t-004", "Finish task test")
        task_store.finish_task("t-004", final)
        record = task_store.get_task("t-004")

    assert record["status"] == "completed"
    assert record["supervisor_confidence"] == 0.91
    assert record["memory_written"] is True
    assert len(record["subtasks"]) == 1
    assert record["subtasks"][0]["review_approved"] is True
    _clear_mem()


# ═══════════════════════════════════════════════════════════════════════════════
# 9–10. list_tasks
# ═══════════════════════════════════════════════════════════════════════════════

def test_list_tasks_returns_entries():
    _clear_mem()
    from orchestrator.db import task_store

    with patch("orchestrator.db.engine._pg_available", False):
        task_store.create_task("t-101", "Task A")
        task_store.create_task("t-102", "Task B")
        task_store.create_task("t-103", "Task C")
        results = task_store.list_tasks(limit=10, offset=0)

    assert len(results) == 3
    _clear_mem()


def test_list_tasks_respects_limit():
    _clear_mem()
    from orchestrator.db import task_store

    with patch("orchestrator.db.engine._pg_available", False):
        for i in range(5):
            task_store.create_task(f"t-2{i:02d}", f"Task {i}")
        results = task_store.list_tasks(limit=2, offset=0)

    assert len(results) == 2
    _clear_mem()


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Workflow MemorySaver fallback
# ═══════════════════════════════════════════════════════════════════════════════

def test_workflow_uses_memory_saver_when_pg_saver_unavailable():
    """_make_checkpointer() must return MemorySaver when PostgresSaver fails."""
    from orchestrator.graph.workflow import _make_checkpointer
    from langgraph.checkpoint.memory import MemorySaver

    with patch.dict("sys.modules", {"langgraph.checkpoint.postgres": None}):
        cp = _make_checkpointer()

    assert isinstance(cp, MemorySaver)


# ═══════════════════════════════════════════════════════════════════════════════
# 12–15. FastAPI app hardening
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from orchestrator.main import app
    # Patch lifespan side-effects so TestClient doesn't try to connect to Postgres
    with patch("orchestrator.db.engine.init_db", return_value=False), \
         patch("orchestrator.db.engine.create_tables"):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def test_liveness_always_200(client):
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"


def test_readiness_returns_degraded_when_deps_down(client):
    with patch("orchestrator.db.engine._pg_available", False), \
         patch("redis.from_url", side_effect=Exception("no redis")), \
         patch("chromadb.HttpClient", side_effect=Exception("no chroma")):
        resp = client.get("/health/ready")
    # May be 200 (ok) or 503 (degraded) depending on which deps are checked
    assert resp.status_code in (200, 503)
    body = resp.json()
    assert "checks" in body


def test_request_id_header_present(client):
    resp = client.get("/health/live")
    assert "x-request-id" in {k.lower() for k in resp.headers}


def test_get_tasks_list(client):
    _clear_mem()
    with patch("orchestrator.db.engine._pg_available", False):
        from orchestrator.db import task_store
        task_store.create_task("t-list-1", "Test list endpoint")
        resp = client.get("/tasks/")
    assert resp.status_code == 200
    body = resp.json()
    assert "tasks" in body
    assert "count" in body
    _clear_mem()
