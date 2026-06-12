"""
SQLAlchemy engine and session factory.

Graceful degradation: if PostgreSQL is unavailable at startup,
_pg_available is set to False and callers fall back to the in-process
dict store — the rest of the application continues normally.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None
_pg_available: bool = False


class Base(DeclarativeBase):
    pass


def init_db() -> bool:
    """
    Initialize the SQLAlchemy engine.  Returns True if the database is
    reachable, False otherwise.  Safe to call multiple times.
    """
    global _engine, _SessionLocal, _pg_available

    from orchestrator.config import get_settings
    cfg = get_settings()

    try:
        _engine = create_engine(
            cfg.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            connect_args={"connect_timeout": 3},
        )
        # Verify connectivity
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
        _pg_available = True
        logger.info("PostgreSQL connected: %s", cfg.database_url.split("@")[-1])
        return True

    except Exception as exc:
        logger.warning("PostgreSQL unavailable (%s) — using in-memory fallback", exc)
        _engine = None
        _SessionLocal = None
        _pg_available = False
        return False


def is_pg_available() -> bool:
    return _pg_available


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a DB session; raises RuntimeError if Postgres is unavailable."""
    if not _pg_available or _SessionLocal is None:
        raise RuntimeError("PostgreSQL not available")
    session: Session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_tables() -> None:
    """Create all tables that don't exist yet.  No-op if Postgres is down."""
    if _engine is None:
        return
    from orchestrator.db import models  # noqa: F401 — registers models with Base
    Base.metadata.create_all(bind=_engine)
    logger.info("DB tables ensured")
