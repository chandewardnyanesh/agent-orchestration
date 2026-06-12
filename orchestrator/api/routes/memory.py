"""Memory management endpoints."""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class MemoryQuery(BaseModel):
    query: str
    top_k: int = 3


@router.post("/search")
def search_memory(body: MemoryQuery):
    from orchestrator.memory.manager import MemoryManager
    result = MemoryManager().retrieve(body.query, body.top_k)
    return {"results": result}


@router.delete("/{task_id}")
def delete_memory(task_id: str):
    """GDPR-compliant: delete all memory associated with a task."""
    from orchestrator.memory.semantic import SemanticMemory
    from orchestrator.memory.working import WorkingMemory
    SemanticMemory().delete(task_id)
    WorkingMemory().clear(task_id)
    return {"deleted": task_id}
