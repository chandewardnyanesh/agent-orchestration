"""
FastAPI application entry point — Phase 6 production hardening.

Additions over Phase 5:
  - lifespan context manager (replaces deprecated @app.on_event)
  - DB init + table creation at startup
  - Request-ID middleware (X-Request-ID header on every response)
  - Structured request/response logging middleware
  - Global exception handler with sanitised error response
  - /health/live  — liveness (always 200 if process is alive)
  - /health/ready — readiness (checks Postgres + Redis)
  - Rate limiting via slowapi (100 req/min per IP by default)
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    from orchestrator.observability.tracer import setup_tracing
    setup_tracing()

    from orchestrator.db.engine import init_db, create_tables
    if init_db():
        create_tables()

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    from orchestrator.db.engine import _engine
    if _engine is not None:
        _engine.dispose()
        logger.info("DB connection pool closed")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agent Orchestration API",
    description="Multi-agent orchestration with tool use, memory, and HITL.",
    version="0.6.0",
    lifespan=lifespan,
)

# ── Rate limiting ─────────────────────────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
except ImportError:
    logger.info("slowapi not installed — rate limiting disabled")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# ── Request-ID middleware ─────────────────────────────────────────────────────

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = req_id
    t0 = time.perf_counter()
    response: Response = await call_next(request)
    latency_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = req_id
    logger.info(
        "method=%s path=%s status=%s latency_ms=%.1f req_id=%s",
        request.method, request.url.path, response.status_code, latency_ms, req_id,
    )
    return response


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", "—")
    logger.exception("Unhandled exception req_id=%s: %s", req_id, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": req_id},
    )


# ── Routers ───────────────────────────────────────────────────────────────────
from orchestrator.api.routes import tasks, memory, review, traces  # noqa: E402

app.include_router(tasks.router,  prefix="/tasks",  tags=["Tasks"])
app.include_router(memory.router, prefix="/memory", tags=["Memory"])
app.include_router(review.router, prefix="/review", tags=["HITL Review"])
app.include_router(traces.router, prefix="/traces", tags=["Observability"])


# ── Health endpoints ──────────────────────────────────────────────────────────

@app.get("/health/live", tags=["Health"])
def liveness():
    """Kubernetes liveness probe — returns 200 as long as the process is alive."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["Health"])
def readiness():
    """
    Kubernetes readiness probe — checks Postgres and Redis.
    Returns 200 only when all required dependencies are reachable.
    """
    checks: dict[str, str] = {}
    ready = True

    # PostgreSQL
    from orchestrator.db.engine import is_pg_available
    checks["postgres"] = "ok" if is_pg_available() else "unavailable"

    # Redis
    try:
        import redis as redis_lib
        from orchestrator.config import get_settings
        r = redis_lib.from_url(get_settings().redis_url, socket_connect_timeout=1)
        r.ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "unavailable"
        ready = False

    # ChromaDB
    try:
        import chromadb
        from orchestrator.config import get_settings
        cfg = get_settings()
        c = chromadb.HttpClient(host=cfg.chroma_host, port=cfg.chroma_port)
        c.heartbeat()
        checks["chromadb"] = "ok"
    except Exception:
        checks["chromadb"] = "unavailable"

    status_code = 200 if ready else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if ready else "degraded", "checks": checks},
    )


# Legacy single health endpoint kept for backwards compatibility
@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
