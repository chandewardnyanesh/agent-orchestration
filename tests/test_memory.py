"""
Phase 2 — Memory System Tests

Tests:
  1. SemanticMemory upsert + query (live ChromaDB required)
  2. relevance_score increments on retrieval
  3. MemoryManager.retrieve() returns rich context string
  4. MemoryManager.store() with quality_score
  5. Consolidation triggers after _CONSOLIDATION_THRESHOLD tasks
  6. Graceful degradation when ChromaDB is unavailable
"""
from __future__ import annotations

import json
import time
import pytest
from unittest.mock import MagicMock, patch


# ── Helper: mock SemanticMemory for unit tests ───────────────────────────────

def _make_mock_semantic(stored: list = None):
    """Return a SemanticMemory mock pre-loaded with `stored` memories."""
    mem = MagicMock()
    store = list(stored or [])

    def fake_upsert(doc_id, document, metadata=None, relevance_score=0.5):
        store.append({"id": doc_id, "document": document,
                      "metadata": {**(metadata or {}), "relevance_score": relevance_score}})

    def fake_query(query_text, top_k=3):
        return [{"document": s["document"], "metadata": s["metadata"],
                 "weighted_score": s["metadata"].get("relevance_score", 0.5)}
                for s in store[:top_k]]

    def fake_get_all(limit=100):
        return store[:limit]

    def fake_count():
        return len(store)

    mem.upsert.side_effect = fake_upsert
    mem.query.side_effect  = fake_query
    mem.get_all.side_effect = fake_get_all
    mem.count.side_effect  = fake_count
    mem._collection = MagicMock()
    return mem


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SemanticMemory upsert + query (unit — mocked ChromaDB)
# ═══════════════════════════════════════════════════════════════════════════════

def test_semantic_memory_upsert_and_query():
    from orchestrator.memory.semantic import SemanticMemory

    sem = SemanticMemory.__new__(SemanticMemory)
    sem._collection = None   # start disabled
    # Upsert should be a no-op when disabled
    sem.upsert("id-1", "test doc")
    assert sem.query("anything") == []


def test_semantic_memory_graceful_degradation():
    from orchestrator.memory.semantic import SemanticMemory
    sem = SemanticMemory.__new__(SemanticMemory)
    sem._collection = None
    # All methods should return safe defaults
    assert sem.count() == 0
    assert sem.get_all() == []
    sem.delete("no-crash")   # should not raise


# ═══════════════════════════════════════════════════════════════════════════════
# 2. relevance_score seeds from quality_score in store()
# ═══════════════════════════════════════════════════════════════════════════════

def test_store_seeds_relevance_from_quality():
    from orchestrator.memory.manager import MemoryManager

    mock_sem = _make_mock_semantic()
    mock_wm  = MagicMock()

    mgr = MemoryManager.__new__(MemoryManager)
    mgr._semantic = mock_sem
    mgr._working  = mock_wm

    with patch.object(mgr, "_maybe_consolidate"):   # skip consolidation here
        mgr.store(
            task_id="t-1",
            user_request="Research AI frameworks",
            execution_plan={"subtasks": [{"specialist": "research", "description": "Find frameworks"}]},
            final_output="Found 3 frameworks",
            quality_score=1.0,
            success=True,
        )

    call_kwargs = mock_sem.upsert.call_args.kwargs
    # high quality → high initial relevance (0.5 + 1.0*0.3 = 0.8)
    assert abs(call_kwargs["relevance_score"] - 0.8) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# 3. retrieve() returns rich context string with relevance headers
# ═══════════════════════════════════════════════════════════════════════════════

def test_retrieve_returns_rich_context():
    from orchestrator.memory.manager import MemoryManager

    stored = [
        {
            "document": "Task: Compare Python frameworks\nOutcome: FastAPI is fastest",
            "metadata": {
                "task_id": "t-1",
                "specialist_types": json.dumps(["research"]),
                "relevance_score": 0.75,
                "quality_score": 0.9,
            },
            "weighted_score": 0.75,
        }
    ]
    mock_sem = _make_mock_semantic(stored)
    mock_wm  = MagicMock()

    mgr = MemoryManager.__new__(MemoryManager)
    mgr._semantic = mock_sem
    mgr._working  = mock_wm

    context = mgr.retrieve("Python web frameworks", top_k=1)

    assert "[Memory 1]" in context
    assert "relevance=" in context
    assert "research" in context
    assert "FastAPI" in context


def test_retrieve_empty_memory_returns_placeholder():
    from orchestrator.memory.manager import MemoryManager

    mock_sem = _make_mock_semantic([])
    mock_wm  = MagicMock()
    mgr = MemoryManager.__new__(MemoryManager)
    mgr._semantic = mock_sem
    mgr._working  = mock_wm

    result = mgr.retrieve("anything")
    assert "No relevant prior context" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Consolidation triggers at threshold
# ═══════════════════════════════════════════════════════════════════════════════

def test_consolidation_triggers_at_threshold():
    from orchestrator.memory.manager import MemoryManager, _CONSOLIDATION_THRESHOLD

    # Build a cluster of exactly THRESHOLD research memories
    stored = [
        {
            "id": f"t-{i}",
            "document": f"Task {i}: research task {i}",
            "metadata": {
                "task_id": f"t-{i}",
                "specialist_types": json.dumps(["research"]),
                "relevance_score": 0.5,
            }
        }
        for i in range(_CONSOLIDATION_THRESHOLD)
    ]

    mock_sem = _make_mock_semantic(stored)
    mock_wm  = MagicMock()

    mgr = MemoryManager.__new__(MemoryManager)
    mgr._semantic = mock_sem
    mgr._working  = mock_wm

    with patch.object(mgr, "_consolidate_cluster") as mock_consolidate:
        mgr._maybe_consolidate(["research"])
        mock_consolidate.assert_called_once()
        call_args = mock_consolidate.call_args
        assert call_args[0][0] == "research"
        assert len(call_args[0][1]) == _CONSOLIDATION_THRESHOLD


def test_consolidation_does_not_trigger_below_threshold():
    from orchestrator.memory.manager import MemoryManager, _CONSOLIDATION_THRESHOLD

    stored = [
        {
            "id": f"t-{i}",
            "document": f"Task {i}",
            "metadata": {
                "task_id": f"t-{i}",
                "specialist_types": json.dumps(["research"]),
                "relevance_score": 0.5,
            }
        }
        for i in range(_CONSOLIDATION_THRESHOLD - 1)  # one short
    ]

    mock_sem = _make_mock_semantic(stored)
    mock_wm  = MagicMock()
    mgr = MemoryManager.__new__(MemoryManager)
    mgr._semantic = mock_sem
    mgr._working  = mock_wm

    with patch.object(mgr, "_consolidate_cluster") as mock_consolidate:
        mgr._maybe_consolidate(["research"])
        mock_consolidate.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Live ChromaDB round-trip (skipped if ChromaDB not available)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
def test_chromadb_live_roundtrip():
    """Requires docker-compose up chromadb."""
    from orchestrator.memory.semantic import SemanticMemory

    sem = SemanticMemory()
    if not sem._collection:
        pytest.skip("ChromaDB not available")

    doc_id = f"test-{int(time.time())}"
    sem.upsert(doc_id, "Test: Python web frameworks comparison", {"task_id": doc_id})

    results = sem.query("Python frameworks", top_k=1)
    assert any(doc_id in r["metadata"].get("task_id", "") for r in results), \
        "Stored document not found in query results"

    sem.delete(doc_id)   # cleanup
