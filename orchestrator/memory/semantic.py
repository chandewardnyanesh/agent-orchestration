"""
Long-term semantic memory backed by ChromaDB.

Phase 2 additions:
- relevance_score stored per memory (updated on each successful retrieval)
- Weighted retrieval: cosine similarity × relevance_score for ranking
- count() helper used by consolidation logic in MemoryManager
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import chromadb
from orchestrator.config import get_settings

logger = logging.getLogger(__name__)

# Floor score for brand-new memories (0-1 scale, incremented on each use)
_INITIAL_RELEVANCE = 0.5
_RELEVANCE_INCREMENT = 0.1   # each retrieval boosts the score
_RELEVANCE_MAX = 1.0


class SemanticMemory:

    def __init__(self):
        cfg = get_settings()
        self._collection = None
        try:
            client = chromadb.HttpClient(host=cfg.chroma_host, port=cfg.chroma_port)
            client.heartbeat()                           # raises if ChromaDB is unreachable
            self._collection = client.get_or_create_collection(
                name=cfg.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("SemanticMemory connected — %d memories", self._collection.count())
        except Exception:
            logger.warning("ChromaDB unavailable — semantic memory disabled")

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        doc_id: str,
        document: str,
        metadata: Optional[dict] = None,
        relevance_score: float = _INITIAL_RELEVANCE,
    ) -> None:
        """Store or update a memory document with an initial relevance score."""
        if not self._collection:
            return
        meta = dict(metadata or {})
        meta["relevance_score"] = relevance_score
        meta["stored_at"] = time.time()
        self._collection.upsert(
            ids=[doc_id],
            documents=[document],
            metadatas=[meta],
        )
        logger.debug("memory upserted id=%s relevance=%.2f", doc_id, relevance_score)

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(self, query_text: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve the most relevant memories.

        Ranking = cosine_similarity × relevance_score
        Also increments the relevance_score of each retrieved memory so
        frequently useful memories naturally rise over time.
        """
        if not self._collection:
            return []

        count = self._collection.count()
        if count == 0:
            return []

        # Fetch more candidates than needed so we can re-rank by weighted score
        fetch_k = min(max(top_k * 3, 10), count)
        raw = self._collection.query(
            query_texts=[query_text],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        docs      = raw.get("documents",  [[]])[0]
        metas     = raw.get("metadatas",  [[]])[0]
        distances = raw.get("distances",  [[]])[0]   # cosine distance (lower = closer)

        # Convert distance → similarity, then weight by stored relevance_score
        results = []
        for doc, meta, dist in zip(docs, metas, distances):
            cosine_sim = max(0.0, 1.0 - dist)        # ChromaDB returns L2 or cosine dist
            rel_score  = float(meta.get("relevance_score", _INITIAL_RELEVANCE))
            weighted   = cosine_sim * rel_score
            results.append({"document": doc, "metadata": meta, "weighted_score": weighted})

        # Sort descending by weighted score, take top_k
        results.sort(key=lambda x: x["weighted_score"], reverse=True)
        top = results[:top_k]

        # Boost relevance_score of returned memories (they were useful)
        for item in top:
            doc_id = item["metadata"].get("task_id")
            if doc_id:
                self._boost_relevance(doc_id, item["metadata"])

        return top

    def _boost_relevance(self, doc_id: str, meta: dict) -> None:
        """Increment the relevance_score for a retrieved document."""
        current = float(meta.get("relevance_score", _INITIAL_RELEVANCE))
        new_score = min(current + _RELEVANCE_INCREMENT, _RELEVANCE_MAX)
        updated_meta = dict(meta)
        updated_meta["relevance_score"] = new_score
        try:
            self._collection.update(ids=[doc_id], metadatas=[updated_meta])
        except Exception as exc:
            logger.debug("relevance boost failed for %s: %s", doc_id, exc)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return total number of stored memories."""
        if not self._collection:
            return 0
        return self._collection.count()

    def get_all(self, limit: int = 100) -> list[dict]:
        """Return all stored memories (used by consolidation logic)."""
        if not self._collection or self._collection.count() == 0:
            return []
        raw = self._collection.get(limit=limit, include=["documents", "metadatas"])
        docs  = raw.get("documents",  [])
        metas = raw.get("metadatas",  [])
        ids   = raw.get("ids", [])
        return [{"id": i, "document": d, "metadata": m}
                for i, d, m in zip(ids, docs, metas)]

    def delete(self, doc_id: str) -> None:
        if not self._collection:
            return
        self._collection.delete(ids=[doc_id])
