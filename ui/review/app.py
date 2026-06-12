"""
HITL Review UI — Streamlit (Phase 3)
--------------------------------------
Run:  streamlit run ui/review/app.py

Phase 3 additions:
- Auto-refresh every 10 seconds
- Confidence badge coloured by level (red / amber / green)
- "Modify" button: lets reviewer edit plan subtasks before approving
- Task status panel showing all tasks (not just pending)
"""
from __future__ import annotations

import json
import time
import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="HITL Review Queue", layout="wide", page_icon="🧑‍💼")

# ── Sidebar — controls ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    auto_refresh = st.toggle("Auto-refresh (10 s)", value=True)
    if st.button("🔄 Refresh now"):
        st.rerun()
    st.divider()
    st.caption("Agent Orchestration — Phase 3 HITL")

# ── Auto-refresh counter ─────────────────────────────────────────────────────
if auto_refresh:
    placeholder = st.empty()
    for remaining in range(10, 0, -1):
        placeholder.caption(f"Auto-refresh in {remaining}s…")
        time.sleep(1)
    placeholder.empty()
    st.rerun()

# ── Main area ────────────────────────────────────────────────────────────────
st.title("🧑‍💼 Human-in-the-Loop Review Queue")

# ── Fetch pending items ───────────────────────────────────────────────────────
try:
    resp = httpx.get(f"{API_BASE}/review/pending", timeout=5)
    items = resp.json()
except Exception as e:
    st.error(f"❌ Cannot reach API at {API_BASE}: {e}")
    items = []


def _confidence_badge(conf: float) -> str:
    if conf >= 0.8:
        return f"🟢 {conf:.2f}"
    if conf >= 0.5:
        return f"🟡 {conf:.2f}"
    return f"🔴 {conf:.2f}"


def _hitl_label(level: str) -> str:
    labels = {
        "notify":         "📢 Notify",
        "approve_action": "⚠️ Approve Action",
        "approve_plan":   "📋 Approve Plan",
        "take_over":      "🚨 Take Over",
    }
    return labels.get(level, level)


# ── Pending review cards ──────────────────────────────────────────────────────
if not items:
    st.success("✅  No tasks awaiting review. All clear!")
else:
    st.subheader(f"📥 {len(items)} task(s) awaiting review")

    for item in items:
        task_id = item.get("task_id", "unknown")
        conf    = item.get("supervisor_confidence", 0.0)
        level   = item.get("hitl_level", "approve_plan")

        with st.expander(
            f"{_hitl_label(level)}  —  Task `{task_id[:8]}…`  —  Confidence {_confidence_badge(conf)}",
            expanded=True,
        ):
            col_plan, col_actions = st.columns([3, 1])

            # ── Plan details ──────────────────────────────────────────────────
            with col_plan:
                st.markdown("#### 📝 User Request")
                st.info(item.get("user_request", ""))

                st.markdown("#### 🗂️ Proposed Execution Plan")
                plan = item.get("execution_plan") or {}
                for st_item in plan.get("subtasks", []):
                    deps = st_item.get("depends_on", [])
                    dep_str = f"  _(depends on: {', '.join(deps)})_" if deps else ""
                    st.markdown(
                        f"- **`{st_item['specialist']}`** — {st_item['description']}{dep_str}"
                    )

                if plan.get("reasoning"):
                    st.caption(f"Supervisor reasoning: {plan['reasoning']}")

            # ── Action buttons ────────────────────────────────────────────────
            with col_actions:
                st.markdown("#### ✅ Decision")
                notes = st.text_area(
                    "Notes for Supervisor (rejection reason, etc.)",
                    key=f"notes_{task_id}",
                    height=100,
                    placeholder="Optional — required when rejecting",
                )

                if st.button("✅  Approve", key=f"approve_{task_id}", type="primary"):
                    r = httpx.post(
                        f"{API_BASE}/review/{task_id}/decide",
                        json={"decision": "approved", "notes": notes},
                    )
                    st.success(f"Approved! Graph resuming…  ({r.status_code})")

                if st.button("🔁  Reject & Re-plan", key=f"reject_{task_id}"):
                    if not notes:
                        st.warning("Please add rejection notes so the Supervisor knows what to fix.")
                    else:
                        r = httpx.post(
                            f"{API_BASE}/review/{task_id}/decide",
                            json={"decision": "rejected", "notes": notes},
                        )
                        st.warning(f"Rejected. Supervisor will re-plan.  ({r.status_code})")

                if st.button("🙋  Take Over", key=f"takeover_{task_id}"):
                    r = httpx.post(
                        f"{API_BASE}/review/{task_id}/decide",
                        json={"decision": "take_over", "notes": notes},
                    )
                    st.info(f"Marked for human takeover.  ({r.status_code})")

            # ── Modify plan ───────────────────────────────────────────────────
            with st.expander("✏️  Modify plan JSON before approving", expanded=False):
                st.caption(
                    "Edit the execution plan below, then click 'Approve Modified Plan'. "
                    "The edited JSON will be sent with decision=modified."
                )
                edited_plan = st.text_area(
                    "Plan JSON",
                    value=json.dumps(plan, indent=2),
                    height=250,
                    key=f"editplan_{task_id}",
                )
                if st.button("✅  Approve Modified Plan", key=f"modified_{task_id}"):
                    try:
                        parsed = json.loads(edited_plan)
                        r = httpx.post(
                            f"{API_BASE}/review/{task_id}/decide",
                            json={
                                "decision": "modified",
                                "notes":    json.dumps({"modified_plan": parsed}),
                            },
                        )
                        st.success(f"Modified plan approved. Graph resuming…  ({r.status_code})")
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON: {e}")

st.divider()

# ── Task status panel ─────────────────────────────────────────────────────────
st.subheader("📊 Recent Tasks (manual lookup)")
lookup_id = st.text_input("Paste a task_id to check status:", placeholder="xxxxxxxx-xxxx-…")
if lookup_id:
    try:
        r = httpx.get(f"{API_BASE}/tasks/{lookup_id.strip()}", timeout=5)
        data = r.json()
        status = data.get("status", "unknown")
        colour = {"completed": "🟢", "failed": "🔴", "awaiting_human_approval": "🟡"}.get(status, "⚪")
        st.markdown(f"**Status:** {colour} `{status}`")
        if data.get("final_output"):
            st.markdown("**Final Output:**")
            st.write(data["final_output"])
        if data.get("error"):
            st.error(data["error"])
        with st.expander("Full state JSON"):
            st.json(data)
    except Exception as e:
        st.error(str(e))
