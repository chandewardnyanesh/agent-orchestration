"""
Assembles the LangGraph StateGraph for the full orchestration workflow.

Graph topology:
  retrieve_memory → plan → [await_human?] → execute_subtasks
                                              → review
                                              → [retry loop?]
                                              → synthesize
                                              → write_memory → END

Phase 3: compiled with MemorySaver checkpointer so the graph can be
suspended at await_human (via interrupt()) and resumed later via
Command(resume=...) from the review API endpoint.

Phase 6: try PostgresSaver first (persists checkpoints across restarts),
fall back to MemorySaver when the DB is unavailable.
"""
from __future__ import annotations

import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.graph.state import OrchestratorState
from orchestrator.graph.nodes import (
    retrieve_memory,
    plan,
    await_human,
    execute_subtasks,
    review,
    synthesize,
    write_memory,
)
from orchestrator.graph.edges import after_plan, after_await_human, after_review

logger = logging.getLogger(__name__)


def _make_checkpointer():
    """
    Try PostgresSaver (requires langgraph-checkpoint-postgres + live DB).
    Fall back to MemorySaver so the app always starts.
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from orchestrator.config import get_settings
        conn_string = get_settings().database_url
        checkpointer = PostgresSaver.from_conn_string(conn_string)
        checkpointer.setup()
        logger.info("LangGraph checkpointer: PostgresSaver")
        return checkpointer
    except Exception as exc:
        logger.info("PostgresSaver unavailable (%s) — using MemorySaver", exc)
        return MemorySaver()


# Global checkpointer — single instance shared across all graph invocations.
_checkpointer = _make_checkpointer()


def build_graph() -> StateGraph:
    g = StateGraph(OrchestratorState)

    # ── Add nodes ─────────────────────────────────────────────
    g.add_node("retrieve_memory", retrieve_memory)
    g.add_node("plan", plan)
    g.add_node("await_human", await_human)
    g.add_node("execute_subtasks", execute_subtasks)
    g.add_node("review", review)
    g.add_node("synthesize", synthesize)
    g.add_node("write_memory", write_memory)

    # ── Entry point ───────────────────────────────────────────
    g.set_entry_point("retrieve_memory")

    # ── Linear edges ──────────────────────────────────────────
    g.add_edge("retrieve_memory", "plan")
    g.add_edge("execute_subtasks", "review")
    g.add_edge("synthesize", "write_memory")
    g.add_edge("write_memory", END)

    # ── Conditional edges ─────────────────────────────────────
    g.add_conditional_edges("plan", after_plan, {
        "await_human": "await_human",
        "execute_subtasks": "execute_subtasks",
    })
    g.add_conditional_edges("await_human", after_await_human, {
        "plan": "plan",
        "execute_subtasks": "execute_subtasks",
        "END": END,
    })
    g.add_conditional_edges("review", after_review, {
        "execute_subtasks": "execute_subtasks",
        "await_human": "await_human",
        "synthesize": "synthesize",
    })

    return g


# Compiled graph — import this in the API and workers.
compiled_graph = build_graph().compile(checkpointer=_checkpointer)
