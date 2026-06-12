#!/usr/bin/env python3
"""
Phase 5 — Integration Demo
===========================
Demonstrates the full multi-agent orchestration pipeline without requiring
a live Anthropic API key.  Every agent call is intercepted by a lightweight
mock layer that returns structured responses that satisfy each node's
expectations.

Usage:
    python demo.py            # full demo (mocked LLM)
    python demo.py --live     # live run (needs ANTHROPIC_API_KEY)

What it shows:
  Step 1  Submit a task → polling until completed
  Step 2  Fetch and display the trace (cost, tokens, per-agent latency)
  Step 3  Show node-by-node snapshot timeline
  Step 4  Submit the *same request* again → memory is retrieved from prior run
  Step 5  Show the memory context that was injected into the second run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

# ── Project imports ───────────────────────────────────────────────────────────
from orchestrator.graph.workflow import build_graph
from orchestrator.observability.snapshot_store import (
    clear_snapshots, clear_spans, get_snapshots, task_summary,
)
from orchestrator.memory.manager import MemoryManager
from orchestrator.agents.base import AgentResult
from langgraph.checkpoint.memory import MemorySaver


# ═══════════════════════════════════════════════════════════════════════════════
# ANSI helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text

def bold(t: str)  -> str: return _c("1",      t)
def green(t: str) -> str: return _c("1;32",   t)
def blue(t: str)  -> str: return _c("1;34",   t)
def cyan(t: str)  -> str: return _c("36",     t)
def grey(t: str)  -> str: return _c("2",      t)
def yellow(t: str)-> str: return _c("1;33",   t)
def red(t: str)   -> str: return _c("1;31",   t)

def header(msg: str) -> None:
    width = 72
    print()
    print("─" * width)
    print(f"  {bold(msg)}")
    print("─" * width)

def step(n: int, msg: str) -> None:
    print(f"\n{bold(blue(f'Step {n}'))} {msg}")


# ═══════════════════════════════════════════════════════════════════════════════
# Mock LLM responses (used when --live is not passed)
# ═══════════════════════════════════════════════════════════════════════════════

USER_REQUEST = "Research the top 3 AI orchestration frameworks (LangGraph, CrewAI, AutoGen) and write a 200-word comparison summary."

TASK_ID_1 = "demo-run-001"
TASK_ID_2 = "demo-run-002"


def _plan_for(task_id: str) -> dict:
    return {
        "task_id":               task_id,
        "original_request":      USER_REQUEST,
        "confidence":            0.92,
        "reasoning":             "Two-specialist approach: research then writing.",
        "requires_human_approval": False,
        "subtasks": [
            {
                "id":                    "st-research",
                "description":           "Research LangGraph, CrewAI, and AutoGen features",
                "specialist":            "research",
                "depends_on":            [],
                "expected_output_format": "bullet points",
                "estimated_complexity":  "medium",
            },
            {
                "id":                    "st-writing",
                "description":           "Write 200-word comparison summary",
                "specialist":            "writing",
                "depends_on":            ["st-research"],
                "expected_output_format": "prose",
                "estimated_complexity":  "low",
            },
        ],
    }


RESEARCH_OUTPUT = """\
• LangGraph: Stateful graph execution, native HITL via interrupt(), best for complex workflows.
• CrewAI: Role-based multi-agent collaboration, easy to configure, good for team workflows.
• AutoGen: Microsoft-backed, conversation-centric, strong code-execution agent support."""

WRITING_OUTPUT = """\
Three frameworks dominate AI orchestration in 2025:

**LangGraph** (by LangChain) offers stateful graph execution with first-class support \
for human-in-the-loop via interrupt/resume.  It is the most powerful option for \
deterministic, auditable pipelines.

**CrewAI** takes a role-based approach where autonomous agents collaborate in crews. \
Its declarative API makes it the easiest framework to adopt for team-style multi-agent tasks.

**AutoGen** (Microsoft) centres on conversable agents that can execute code, \
browse the web, and call tools in round-table sessions.  It shines when dynamic \
code generation is core to the workflow.

All three frameworks are MIT-licensed and integrate with OpenAI, Anthropic, and \
open-source models.  For production systems requiring auditability, choose LangGraph. \
For rapid prototyping of collaborative AI teams, start with CrewAI."""

REVIEW_OK = {"approved": True, "quality_score": 0.93,
             "feedback": "Accurate and well-structured.", "issues": []}


# ── Mock agent singletons ─────────────────────────────────────────────────────

def _make_mocks(task_id: str):
    from orchestrator.graph.nodes import _supervisor, _reviewer, _specialists

    def _sup(*a, **kw):
        return AgentResult(output=_plan_for(task_id), confidence=0.92, tokens_used=420)

    def _res(*a, **kw):
        return AgentResult(output=RESEARCH_OUTPUT, confidence=0.95, tokens_used=310)

    def _wri(*a, **kw):
        return AgentResult(output=WRITING_OUTPUT, confidence=0.91, tokens_used=260)

    def _rev(*a, **kw):
        return AgentResult(output=REVIEW_OK, confidence=0.93, tokens_used=90)

    return (
        patch.object(_supervisor,              "timed_invoke", side_effect=_sup),
        patch.object(_specialists["research"], "timed_invoke", side_effect=_res),
        patch.object(_specialists["writing"],  "timed_invoke", side_effect=_wri),
        patch.object(_reviewer,                "timed_invoke", side_effect=_rev),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Graph runner
# ═══════════════════════════════════════════════════════════════════════════════

def _initial_state(task_id: str) -> dict:
    return {
        "task_id":               task_id,
        "user_request":          USER_REQUEST,
        "status":                "pending",
        "error":                 None,
        "execution_plan":        None,
        "supervisor_confidence": 0.0,
        "subtasks":              [],
        "completed_outputs":     {},
        "memory_context":        "",
        "memory_written":        False,
        "hitl_required":         False,
        "hitl_level":            None,
        "hitl_decision":         None,
        "hitl_notes":            None,
        "final_output":          None,
        "trace_id":              task_id,
        "total_tokens":          0,
        "total_cost_usd":        0.0,
    }


def run_task(task_id: str, live: bool) -> dict:
    """Run the graph, optionally with mocked LLM agents."""
    g = build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": task_id}}

    if live:
        return g.invoke(_initial_state(task_id), config=config)

    patches = _make_mocks(task_id)
    with patches[0], patches[1], patches[2], patches[3]:
        return g.invoke(_initial_state(task_id), config=config)


# ═══════════════════════════════════════════════════════════════════════════════
# Display helpers
# ═══════════════════════════════════════════════════════════════════════════════

STATUS_EMOJI = {
    "planning":               "🔍",
    "executing":              "⚙️ ",
    "reviewing":              "🔎",
    "synthesizing":           "✍️ ",
    "completed":              "✅",
    "failed":                 "❌",
    "awaiting_human_approval": "⏳",
    "retrying":               "🔁",
}


def show_status(final: dict) -> None:
    st = final.get("status", "—")
    emoji = STATUS_EMOJI.get(st, "•")
    print(f"  Status          : {emoji} {bold(st)}")
    print(f"  Confidence      : {final.get('supervisor_confidence', 0.0):.2f}")
    print(f"  Total tokens    : {final.get('total_tokens', 0):,}")
    print(f"  Memory written  : {green('yes') if final.get('memory_written') else grey('no')}")
    print()
    print(f"  {bold('Subtasks:')}")
    for sub in final.get("subtasks", []):
        approved = sub.get("review_approved")
        tick = green("✓") if approved else (red("✗") if approved is False else grey("?"))
        print(f"    {tick}  [{sub['specialist']:10}]  {sub['description'][:60]}")


def show_trace(task_id: str) -> None:
    summary = task_summary(task_id)
    print(f"  Total cost      : ${summary.get('total_usd', 0.0):.6f}")
    print(f"  Total tokens    : {summary.get('total_tokens', 0):,}")
    print(f"  Agent calls     : {summary.get('agent_calls', 0)}")
    print(f"  Total latency   : {summary.get('total_latency_ms', 0.0):.1f} ms")
    print()
    breakdown = summary.get("breakdown", [])
    if breakdown:
        print(f"  {'Agent':<14} {'Tokens':>8} {'Latency ms':>12} {'USD':>10}")
        print(f"  {'─'*14} {'─'*8} {'─'*12} {'─'*10}")
        for b in breakdown:
            print(f"  {b['agent']:<14} {b['tokens']:>8,} {b['latency_ms']:>12.1f} ${b['usd']:>9.6f}")


def show_snapshots(task_id: str) -> None:
    snaps = get_snapshots(task_id)
    if not snaps:
        print(grey("  (no snapshots recorded)"))
        return
    for snap in snaps:
        st = snap.get("state", {})
        status  = st.get("status", "—")
        emoji   = STATUS_EMOJI.get(status, "•")
        tokens  = st.get("total_tokens", 0)
        conf    = st.get("supervisor_confidence", 0.0)
        node_label = cyan(f"[{snap['node']}]")
        print(f"  Step {snap['step']}  {node_label:30}  {emoji} {status:<20}  "
              f"tokens={tokens:>5,}  conf={conf:.2f}")


def show_final_output(final: dict) -> None:
    out = final.get("final_output") or grey("(none)")
    print()
    for line in out.splitlines():
        print(f"  {line}")
    print()


def show_memory_context(task_id: str) -> None:
    """Retrieve memory as the supervisor would see it for a repeat request."""
    mgr = MemoryManager()
    ctx = mgr.retrieve(USER_REQUEST, top_k=3)
    if ctx and ctx.strip():
        print(cyan("  Memory retrieved for second run:"))
        for line in ctx.splitlines()[:10]:
            print(f"    {line}")
        if len(ctx.splitlines()) > 10:
            print(grey(f"    … ({len(ctx.splitlines())} lines total)"))
    else:
        print(grey("  (No memory context retrieved — ChromaDB may not be running)"))


# ═══════════════════════════════════════════════════════════════════════════════
# Main demo
# ═══════════════════════════════════════════════════════════════════════════════

def main(live: bool = False) -> None:
    mode_tag = "LIVE (real LLM)" if live else "MOCK (no API key required)"
    header(f"Multi-Agent Orchestration — Phase 5 Demo   [{mode_tag}]")

    print(f"\n  Request: {bold(USER_REQUEST[:80])}…")

    # ── Step 1: First run ─────────────────────────────────────────────────────
    step(1, "Submitting task and running full pipeline…")
    clear_snapshots(TASK_ID_1)
    clear_spans(TASK_ID_1)

    t0 = time.perf_counter()
    final1 = run_task(TASK_ID_1, live)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  {green('Done')}  ({elapsed:.0f} ms)")
    show_status(final1)

    # ── Step 2: Trace summary ─────────────────────────────────────────────────
    step(2, "Trace summary (cost + per-agent telemetry)")
    show_trace(TASK_ID_1)

    # ── Step 3: Snapshot timeline ─────────────────────────────────────────────
    step(3, "Node-by-node snapshot timeline")
    show_snapshots(TASK_ID_1)

    # ── Step 4: Final output ──────────────────────────────────────────────────
    step(4, "Final synthesised output")
    show_final_output(final1)

    # ── Step 5: Second run (memory retrieval) ─────────────────────────────────
    step(5, "Running same task again — memory should surface prior context")
    clear_snapshots(TASK_ID_2)
    clear_spans(TASK_ID_2)

    # Show what memory the supervisor would receive
    show_memory_context(TASK_ID_1)

    # Run the second task (mocked agents still return consistent output)
    if not live:
        # Update mock to use task_id_2 plan
        from orchestrator.graph.nodes import _supervisor, _reviewer, _specialists

        def _sup2(*a, **kw):
            # Capture memory_context from inputs to demonstrate retrieval
            inputs = a[0] if a else kw.get("inputs", {})
            ctx = inputs.get("memory_context", "")
            if ctx:
                print(f"\n  {cyan('[supervisor]')} Memory context received "
                      f"({len(ctx)} chars) — plan incorporates prior learnings")
            return AgentResult(output=_plan_for(TASK_ID_2), confidence=0.95, tokens_used=380)

        p2 = (
            patch.object(_supervisor,              "timed_invoke", side_effect=_sup2),
            patch.object(_specialists["research"], "timed_invoke",
                         return_value=AgentResult(output=RESEARCH_OUTPUT, confidence=0.95, tokens_used=300)),
            patch.object(_specialists["writing"],  "timed_invoke",
                         return_value=AgentResult(output=WRITING_OUTPUT, confidence=0.92, tokens_used=250)),
            patch.object(_reviewer,                "timed_invoke",
                         return_value=AgentResult(output=REVIEW_OK, confidence=0.93, tokens_used=85)),
        )
        with p2[0], p2[1], p2[2], p2[3]:
            final2 = build_graph().compile(checkpointer=MemorySaver()).invoke(
                _initial_state(TASK_ID_2),
                config={"configurable": {"thread_id": TASK_ID_2}},
            )
    else:
        final2 = run_task(TASK_ID_2, live=True)

    print()
    show_status(final2)

    # ── Summary ───────────────────────────────────────────────────────────────
    header("Demo Complete")
    run1_ok = final1.get("status") == "completed"
    run2_ok = final2.get("status") == "completed"
    mem_written = final1.get("memory_written", False)

    print(f"  Run 1 completed   : {green('yes') if run1_ok else red('no')}")
    print(f"  Run 2 completed   : {green('yes') if run2_ok else red('no')}")
    print(f"  Memory persisted  : {green('yes') if mem_written else yellow('no (ChromaDB unavailable)')}")
    print()
    print(f"  {grey('Trace Explorer UI:  streamlit run ui/traces/app.py')}")
    print(f"  {grey('Review UI:          streamlit run ui/review/app.py')}")
    print(f"  {grey('API server:         uvicorn orchestrator.main:app --reload')}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 orchestration demo")
    parser.add_argument("--live", action="store_true",
                        help="Use real Anthropic API (requires ANTHROPIC_API_KEY)")
    args = parser.parse_args()
    main(live=args.live)
