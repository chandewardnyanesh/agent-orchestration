"""
LangGraph node functions.
Each function receives the full OrchestratorState and returns a partial
state dict (LangGraph merges it in).
"""
from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from orchestrator.graph.state import OrchestratorState, SubTaskState
from orchestrator.agents.supervisor import SupervisorAgent
from orchestrator.agents.reviewer import ReviewerAgent
from orchestrator.agents.specialists.research import ResearchAgent
from orchestrator.agents.specialists.analysis import AnalysisAgent
from orchestrator.agents.specialists.writing import WritingAgent
from orchestrator.agents.specialists.code import CodeAgent

logger = logging.getLogger(__name__)


# ── Phase 4: snapshot helper ─────────────────────────────────────────────────

def _snap(node_name: str, state: OrchestratorState, result: dict) -> dict:
    """
    Save an execution snapshot then return result unchanged.
    Never raises — observability must not break the workflow.
    """
    try:
        from orchestrator.observability.snapshot_store import save_snapshot
        merged = {**state, **result}
        save_snapshot(state.get("task_id", ""), node_name, merged)
    except Exception as exc:
        logger.debug("snapshot save failed node=%s: %s", node_name, exc)
    return result


# Agent singletons (one instance per process)
_supervisor = SupervisorAgent()
_reviewer = ReviewerAgent()
_specialists = {
    "research": ResearchAgent(),
    "analysis": AnalysisAgent(),
    "writing": WritingAgent(),
    "code": CodeAgent(),
}


# ── Node: retrieve_memory ─────────────────────────────────────────────────────

def retrieve_memory(state: OrchestratorState) -> dict:
    """Query ChromaDB for memories related to the user request."""
    from orchestrator.memory.manager import MemoryManager
    mgr = MemoryManager()
    context = mgr.retrieve(state["user_request"], top_k=3)
    result = {"memory_context": context, "status": "planning"}
    return _snap("retrieve_memory", state, result)


# ── Node: plan ────────────────────────────────────────────────────────────────

def plan(state: OrchestratorState) -> dict:
    """
    Supervisor decomposes the task into an execution plan.

    Phase 3: passes hitl_notes as rejection_notes on re-plan so the
    Supervisor can address the human reviewer's concerns.
    """
    result = _supervisor.timed_invoke({
        "task_id":         state["task_id"],
        "user_request":    state["user_request"],
        "memory_context":  state.get("memory_context", ""),
        "rejection_notes": state.get("hitl_notes", ""),   # non-empty only on re-plan
    })

    plan_data = result.output  # dict from ExecutionPlan.model_dump()

    subtasks: list[SubTaskState] = [
        {
            "id": st["id"],
            "description": st["description"],
            "specialist": st["specialist"],
            "depends_on": st.get("depends_on", []),
            "expected_output_format": st.get("expected_output_format", "text"),
            "status": "pending",
            "output": None,
            "review_approved": None,
            "review_feedback": None,
            "attempt": 0,
        }
        for st in plan_data.get("subtasks", [])
    ]

    hitl_required = plan_data.get("requires_human_approval", False)

    out = {
        "execution_plan":        plan_data,
        "supervisor_confidence": result.confidence,
        "subtasks":              subtasks,
        "hitl_required":         hitl_required,
        "hitl_level":            "approve_plan" if hitl_required else None,
        "total_tokens":          state.get("total_tokens", 0) + result.tokens_used,
        "status":                "awaiting_human_approval" if hitl_required else "executing",
    }
    return _snap("plan", state, out)


# ── Node: await_human ─────────────────────────────────────────────────────────

def await_human(state: OrchestratorState) -> dict:
    """
    Phase 3 — True graph suspension via LangGraph interrupt().

    Flow:
      1. Push task to Redis HITL queue (powers the review UI).
      2. Call interrupt() — LangGraph serialises the graph checkpoint and
         raises GraphInterrupt, pausing execution here.
      3. The API catches GraphInterrupt and marks the task
         'awaiting_human_approval'.
      4. When a reviewer POSTs /review/{task_id}/decide, the API calls
         compiled_graph.invoke(Command(resume={...}), config=...) which
         resumes exactly from this point with the human's decision payload.
      5. The returned dict is merged into state as normal and routing
         continues via after_await_human().
    """
    from langgraph.types import interrupt
    from orchestrator.hitl.queue import HITLQueue

    HITLQueue().push({
        "task_id":              state["task_id"],
        "user_request":         state["user_request"],
        "execution_plan":       state.get("execution_plan"),
        "hitl_level":           state.get("hitl_level"),
        "supervisor_confidence": state.get("supervisor_confidence", 0.0),
    })
    logger.info(
        "task=%s pausing graph — pushed to HITL queue level=%s",
        state["task_id"], state.get("hitl_level"),
    )

    # ← Graph suspends here; resumes when Command(resume=payload) is sent.
    decision_payload = interrupt({
        "task_id":   state["task_id"],
        "hitl_level": state.get("hitl_level"),
    })

    decision = decision_payload.get("decision", "approved")
    notes    = decision_payload.get("notes", "")
    logger.info("task=%s resumed with decision=%s", state["task_id"], decision)

    next_status = {
        "approved":  "executing",
        "modified":  "executing",
        "rejected":  "failed",
        "take_over": "failed",
    }.get(decision, "executing")

    return {
        "hitl_decision": decision,
        "hitl_notes":    notes,
        "status":        next_status,
    }


# ── Node: execute_subtasks ────────────────────────────────────────────────────

def execute_subtasks(state: OrchestratorState) -> dict:
    """
    Run pending subtasks in dependency-ordered waves.  Subtasks within the
    same wave (no unresolved deps) execute in parallel via a thread pool.
    """
    completed_outputs = dict(state.get("completed_outputs", {}))
    subtasks = [dict(st) for st in state["subtasks"]]

    def _run_one(st: dict, snapshot: dict) -> dict:
        """Execute a single subtask; snapshot is a read-only copy of completed_outputs."""
        specialist = _specialists.get(st["specialist"])
        if not specialist:
            st["status"] = "failed"
            return st

        st["status"] = "in_progress"
        st["attempt"] += 1

        context = "\n\n".join(
            f"[{dep}]: {snapshot[dep]}"
            for dep in st["depends_on"]
            if dep in snapshot
        )
        result = specialist.timed_invoke({
            "task_id":             state["task_id"],   # links span to task in trace store
            "subtask_description": st["description"],
            "context":             context,
        })
        st["output"] = str(result.output)
        st["status"] = "completed"
        return st

    # Process dependency waves: each wave contains all subtasks whose deps are met
    max_waves = len(subtasks) + 1
    for _ in range(max_waves):
        ready = [
            st for st in subtasks
            if st["status"] == "pending"
            and all(completed_outputs.get(dep) for dep in st["depends_on"])
        ]
        if not ready:
            break

        # Capture deps snapshot before launching threads (read-only inside threads)
        deps_snapshot = dict(completed_outputs)

        with ThreadPoolExecutor(max_workers=len(ready)) as pool:
            futures = {pool.submit(_run_one, st, deps_snapshot): st["id"] for st in ready}
            for future in as_completed(futures):
                updated = future.result()
                # Write back to subtasks list and completed_outputs serially
                idx = next(i for i, s in enumerate(subtasks) if s["id"] == updated["id"])
                subtasks[idx] = updated
                if updated["status"] == "completed":
                    completed_outputs[updated["id"]] = updated["output"]

    out = {
        "subtasks":          subtasks,
        "completed_outputs": completed_outputs,
        "status":            "reviewing",
        "total_tokens":      state.get("total_tokens", 0),
    }
    return _snap("execute_subtasks", state, out)


# ── Node: review ──────────────────────────────────────────────────────────────

def review(state: OrchestratorState) -> dict:
    """Reviewer agent checks each completed subtask output."""
    subtasks = [dict(st) for st in state["subtasks"]]
    settings_obj = __import__("orchestrator.config", fromlist=["get_settings"]).get_settings()

    all_approved = True
    for st in subtasks:
        if st["status"] != "completed" or st["review_approved"]:
            continue

        result = _reviewer.timed_invoke({
            "task_id":             state["task_id"],
            "subtask_description": st["description"],
            "specialist_output":   st["output"] or "",
            "expected_format":     st["expected_output_format"],
            "attempt_number":      st["attempt"],
        })

        decision = result.output
        st["review_approved"] = decision["approved"]
        st["review_feedback"] = decision.get("feedback", "")

        if not decision["approved"]:
            all_approved = False
            if st["attempt"] < settings_obj.max_specialist_retries:
                st["status"] = "pending"  # allow retry
            else:
                # Exhausted retries — escalate to human
                st["status"] = "failed"

    next_status = "synthesizing" if all_approved else "retrying"
    out = {"subtasks": subtasks, "status": next_status}
    return _snap("review", state, out)


# ── Node: synthesize ──────────────────────────────────────────────────────────

def synthesize(state: OrchestratorState) -> dict:
    """Merge all approved subtask outputs into a final response."""
    parts = []
    for st in state["subtasks"]:
        if st.get("review_approved") and st.get("output"):
            parts.append(f"### {st['description']}\n{st['output']}")

    final_output = "\n\n".join(parts) if parts else "No approved outputs available."
    out = {"final_output": final_output, "status": "completed"}
    return _snap("synthesize", state, out)


# ── Node: write_memory ────────────────────────────────────────────────────────

def write_memory(state: OrchestratorState) -> dict:
    """
    Persist lessons learned from this task to ChromaDB.

    Phase 2: passes quality_score (mean reviewer score across subtasks)
    and success flag so the memory system can rank high-quality experiences.
    """
    from orchestrator.memory.manager import MemoryManager

    # Derive an aggregate quality score from the subtask reviewer scores
    subtasks = state.get("subtasks", [])
    approved = [st for st in subtasks if st.get("review_approved")]
    quality_score = (
        sum(1.0 for st in approved) / len(subtasks)
        if subtasks else 0.5
    )
    success = state.get("status") == "completed" and bool(approved)

    mgr = MemoryManager()
    mgr.store(
        task_id=state["task_id"],
        user_request=state["user_request"],
        execution_plan=state.get("execution_plan", {}),
        final_output=state.get("final_output", ""),
        quality_score=quality_score,
        success=success,
    )

    # Phase 4 — cost alert check
    try:
        from orchestrator.observability.cost_tracker import CostTracker
        CostTracker().check_alert(state["task_id"])
    except Exception:
        pass

    out = {"memory_written": True}
    return _snap("write_memory", state, out)
