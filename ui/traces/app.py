"""
Trace Explorer UI — Streamlit (Phase 4)
-----------------------------------------
Run:  streamlit run ui/traces/app.py

Features:
  - Cost & token metrics dashboard
  - Per-agent latency bar chart
  - Subtask execution tree (graphviz)
  - Node-by-node snapshot timeline with status badges
  - Step-through replay diff viewer
  - Cost alert warning banner
"""
from __future__ import annotations

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Trace Explorer",
    layout="wide",
    page_icon="🔭",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔭 Trace Explorer")
    st.caption("Phase 4 — Observability")
    st.divider()
    task_id = st.text_input("Task ID", placeholder="Paste a task UUID…")
    if st.button("🔄 Load Trace"):
        st.session_state["loaded_task"] = task_id
    st.divider()
    compare_a = st.number_input("Diff: step A", min_value=0, value=0)
    compare_b = st.number_input("Diff: step B", min_value=1, value=1)

loaded = st.session_state.get("loaded_task") or task_id

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🔭 Agent Execution Trace Explorer")

if not loaded:
    st.info("Enter a Task ID in the sidebar to explore its execution trace.")
    st.stop()

# ── Fetch all data in parallel ────────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch(endpoint: str):
    try:
        r = httpx.get(f"{API_BASE}{endpoint}", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

summary   = fetch(f"/traces/{loaded}/summary")
snapshots = fetch(f"/traces/{loaded}/snapshots")
spans     = fetch(f"/traces/{loaded}/spans")
task      = fetch(f"/tasks/{loaded}")

# ── Cost alert banner ─────────────────────────────────────────────────────────
total_usd = summary.get("total_usd", 0.0)
if total_usd > 1.0:
    st.error(f"🚨 COST ALERT — this task spent ${total_usd:.4f} (threshold $1.00)")

# ── Top metrics ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Status",          task.get("status", "—"))
col2.metric("Total Tokens",    f"{summary.get('total_tokens', 0):,}")
col3.metric("Total Cost",      f"${total_usd:.4f}")
col4.metric("Agent Calls",     summary.get("agent_calls", 0))
col5.metric("Total Latency",   f"{summary.get('total_latency_ms', 0):.0f} ms")

st.divider()

# ── Agent latency chart ───────────────────────────────────────────────────────
st.subheader("⏱️ Per-Agent Latency & Tokens")
breakdown = summary.get("breakdown", [])
if breakdown:
    import pandas as pd
    df = pd.DataFrame(breakdown)
    df["label"] = df["agent"] + "\n" + df["model"].str.split("-").str[0]
    st.bar_chart(df.set_index("label")[["latency_ms", "tokens"]])
    with st.expander("Full breakdown table"):
        st.dataframe(df[["agent", "model", "tokens", "latency_ms", "confidence", "usd"]])
else:
    st.info("No agent spans recorded yet.")

st.divider()

# ── Subtask execution tree (Graphviz) ─────────────────────────────────────────
st.subheader("🗂️ Subtask Execution Tree")
subtasks = task.get("subtasks", [])
if subtasks:
    dot_lines = ["digraph G {", "  rankdir=LR;", '  node [shape=box, style=filled];']
    colour_map = {
        "completed": "#90EE90",   # green
        "failed":    "#FFB6C1",   # red
        "pending":   "#FFFACD",   # yellow
        "in_progress": "#ADD8E6", # blue
    }
    for st_item in subtasks:
        sid   = st_item["id"].replace("-", "_")
        label = f"{st_item['specialist']}\\n{st_item['id'][:6]}…"
        col   = colour_map.get(st_item["status"], "#D3D3D3")
        dot_lines.append(f'  {sid} [label="{label}", fillcolor="{col}"];')
    # START node
    dot_lines.append('  START [shape=circle, fillcolor="#C0C0C0", style=filled];')
    first_ids = [
        st_item["id"].replace("-", "_")
        for st_item in subtasks
        if not st_item.get("depends_on")
    ]
    for fid in first_ids:
        dot_lines.append(f"  START -> {fid};")
    # Dependency edges
    for st_item in subtasks:
        for dep in (st_item.get("depends_on") or []):
            dot_lines.append(
                f'  {dep.replace("-","_")} -> {st_item["id"].replace("-","_")};'
            )
    dot_lines.append("}")
    st.graphviz_chart("\n".join(dot_lines))
else:
    st.info("No subtask data available.")

st.divider()

# ── Node snapshot timeline ────────────────────────────────────────────────────
st.subheader("📸 Node-by-Node Snapshot Timeline")
snaps = snapshots.get("snapshots", [])
if snaps:
    STATUS_EMOJI = {
        "planning":               "🔍",
        "executing":              "⚙️",
        "reviewing":              "🔎",
        "retrying":               "🔁",
        "synthesizing":           "✍️",
        "completed":              "✅",
        "failed":                 "❌",
        "awaiting_human_approval": "⏳",
    }
    for snap in snaps:
        st_state  = snap.get("state", {})
        status    = st_state.get("status", "—")
        emoji     = STATUS_EMOJI.get(status, "•")
        node_name = snap["node"]
        tokens    = st_state.get("total_tokens", 0)
        conf      = st_state.get("supervisor_confidence", 0.0)

        with st.expander(
            f"Step {snap['step']}  [{node_name}]  {emoji} {status}  — {tokens:,} tokens",
            expanded=(snap["step"] == len(snaps) - 1),
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence",   f"{conf:.2f}")
            c2.metric("Tokens so far", f"{tokens:,}")
            c3.metric("HITL decision", st_state.get("hitl_decision") or "—")

            sub_list = st_state.get("subtasks", [])
            if sub_list:
                import pandas as pd
                st.dataframe(pd.DataFrame(sub_list)[
                    ["id", "specialist", "status", "review_approved", "attempt"]
                ])
            if st_state.get("final_output_preview"):
                st.markdown("**Output preview:**")
                st.write(st_state["final_output_preview"])
else:
    st.info("No snapshots recorded yet — run a task first.")

st.divider()

# ── Step diff viewer ─────────────────────────────────────────────────────────
st.subheader("🔀 Step Diff Viewer")
if st.button("Compare steps"):
    try:
        diff = httpx.get(
            f"{API_BASE}/traces/{loaded}/diff",
            params={"step_a": int(compare_a), "step_b": int(compare_b)},
            timeout=5,
        ).json()
        d = diff.get("diff", {})
        if d:
            for field, vals in d.items():
                st.markdown(f"**`{field}`**")
                c1, c2 = st.columns(2)
                c1.write(f"Step {int(compare_a)}: `{vals['step_a']}`")
                c2.write(f"Step {int(compare_b)}: `{vals['step_b']}`")
        else:
            st.success("No differences between these two steps.")
    except Exception as e:
        st.error(str(e))

st.divider()

# ── Raw task state ────────────────────────────────────────────────────────────
with st.expander("📄 Raw task state JSON"):
    st.json(task)
