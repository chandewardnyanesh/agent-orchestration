"""
Memory Manager
--------------
Unified interface over short-term (Redis) and long-term (ChromaDB) memory.

Phase 2 additions:
- retrieve() now includes relevance_score in context surface to Supervisor
- store() adds quality_score, specialist_types, success flag to metadata
- consolidate() LLM-summarises clusters of 5+ similar tasks into a single
  higher-level memory — called automatically after every store()
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

from orchestrator.memory.working import WorkingMemory
from orchestrator.memory.semantic import SemanticMemory

logger = logging.getLogger(__name__)

# Trigger consolidation when a specialist_type cluster exceeds this count
_CONSOLIDATION_THRESHOLD = 5
_CONSOLIDATION_ID_PREFIX  = "consolidated:"


class MemoryManager:

    def __init__(self):
        self._working  = WorkingMemory()
        self._semantic = SemanticMemory()

    # ── Short-term (scoped to task_id, TTL 1 h) ──────────────────────────────

    def set_working(self, task_id: str, key: str, value) -> None:
        self._working.set(task_id, key, value)

    def get_working(self, task_id: str, key: str):
        return self._working.get(task_id, key)

    def clear_working(self, task_id: str) -> None:
        self._working.clear(task_id)

    # ── Long-term retrieval (surfaced to Supervisor) ──────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """
        Return a formatted context string for the Supervisor's planning prompt.

        Each memory entry shows:
          - What the task was
          - Which specialist types were used
          - Relevance score (how often this memory has proven useful)
          - Outcome snippet
        """
        results = self._semantic.query(query, top_k=top_k)
        if not results:
            return "No relevant prior context found."

        lines = []
        for i, item in enumerate(results, 1):
            doc   = item["document"]
            meta  = item.get("metadata", {})
            score = item.get("weighted_score", 0.0)
            rel   = float(meta.get("relevance_score", 0.5))
            specs = meta.get("specialist_types", "[]")
            try:
                specs = ", ".join(json.loads(specs))
            except Exception:
                pass

            header = (
                f"[Memory {i}] "
                f"relevance={rel:.2f}  score={score:.3f}  specialists=[{specs}]"
            )
            lines.append(f"{header}\n{doc[:500]}")

        return "\n\n".join(lines)

    # ── Long-term storage (called after every completed task) ─────────────────

    def store(
        self,
        task_id: str,
        user_request: str,
        execution_plan: dict,
        final_output: str,
        quality_score: float = 1.0,
        success: bool = True,
    ) -> None:
        """
        Persist lessons from a completed task.

        Includes quality_score and success flag so the Supervisor can
        prefer high-quality past approaches.
        """
        subtasks = execution_plan.get("subtasks", [])
        specialist_types = list({st["specialist"] for st in subtasks})

        document = (
            f"Task: {user_request}\n"
            f"Subtasks: {[st['description'] for st in subtasks]}\n"
            f"Specialists used: {specialist_types}\n"
            f"Success: {success}  Quality: {quality_score:.2f}\n"
            f"Outcome: {final_output[:300]}"
        )
        metadata = {
            "task_id":          task_id,
            "specialist_types": json.dumps(specialist_types),
            "quality_score":    quality_score,
            "success":          success,
            "stored_at":        time.time(),
        }

        # Seed relevance from quality (good outcomes start higher)
        initial_relevance = 0.5 + (quality_score * 0.3)  # range 0.5–0.8

        self._semantic.upsert(
            doc_id=task_id,
            document=document,
            metadata=metadata,
            relevance_score=initial_relevance,
        )
        logger.info("memory stored task=%s quality=%.2f specialists=%s",
                    task_id, quality_score, specialist_types)

        # Trigger consolidation check after every store
        self._maybe_consolidate(specialist_types)

    # ── Memory consolidation (Phase 2) ────────────────────────────────────────

    def _maybe_consolidate(self, specialist_types: list[str]) -> None:
        """
        If a specialist-type cluster has grown past the threshold,
        summarise all its memories into a single consolidated entry via LLM.
        This prevents the context window from being flooded with near-duplicate
        task histories and makes the Supervisor's planning smarter over time.
        """
        all_memories = self._semantic.get_all()
        if not all_memories:
            return

        for spec_type in specialist_types:
            # Gather raw (non-consolidated) memories for this specialist type
            cluster = [
                m for m in all_memories
                if not m["id"].startswith(_CONSOLIDATION_ID_PREFIX)
                and spec_type in json.loads(
                    m["metadata"].get("specialist_types", "[]")
                )
            ]

            if len(cluster) >= _CONSOLIDATION_THRESHOLD:
                logger.info(
                    "consolidating %d memories for specialist=%s",
                    len(cluster), spec_type
                )
                self._consolidate_cluster(spec_type, cluster)

    def _consolidate_cluster(self, spec_type: str, cluster: list[dict]) -> None:
        """
        Use an LLM to distil a cluster of similar task memories into one
        higher-level summary and store it as a consolidated memory.
        The original memories are kept but their relevance_score is reduced
        so the consolidated entry is preferred in future retrievals.
        """
        try:
            from orchestrator.config import get_settings
            from orchestrator.agents.base import _build_llm
            from langchain_core.messages import HumanMessage, SystemMessage

            settings = get_settings()
            llm = _build_llm(settings.reviewer_model)   # fast/cheap model

            docs = "\n\n---\n\n".join(m["document"][:400] for m in cluster)
            system = (
                "You are a memory consolidation assistant. "
                "Summarise the following task memories into a single concise paragraph "
                f"describing what was learned about '{spec_type}' tasks. "
                "Focus on: which approaches worked, common subtask patterns, and pitfalls. "
                "Max 200 words."
            )
            response = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=docs),
            ])
            summary = response.content.strip()

            # Store the consolidated summary
            consolidated_id = f"{_CONSOLIDATION_ID_PREFIX}{spec_type}:{int(time.time())}"
            self._semantic.upsert(
                doc_id=consolidated_id,
                document=f"[CONSOLIDATED — {spec_type}]\n{summary}",
                metadata={
                    "task_id":          consolidated_id,
                    "specialist_types": json.dumps([spec_type]),
                    "quality_score":    0.9,
                    "success":          True,
                    "source_count":     len(cluster),
                },
                relevance_score=0.85,   # consolidated memories start with high relevance
            )
            logger.info("consolidation stored id=%s source_count=%d",
                        consolidated_id, len(cluster))

            # Demote original memories so consolidated entry wins in retrieval
            for mem in cluster:
                old_meta = dict(mem["metadata"])
                old_meta["relevance_score"] = 0.2   # push them to the back
                try:
                    self._semantic._collection.update(
                        ids=[mem["id"]], metadatas=[old_meta]
                    )
                except Exception:
                    pass

        except Exception as exc:
            # Consolidation is a nice-to-have — never break the main flow
            logger.warning("memory consolidation failed: %s", exc)
