"""Database query tool — read-only SQL via SQLAlchemy."""
from langchain_core.tools import tool


@tool
def database_query_tool(sql: str) -> str:
    """
    Run a read-only SQL query against the configured PostgreSQL database.

    Args:
        sql: A SELECT statement to execute.

    Returns:
        Query results as a formatted string table.
    """
    # Only allow SELECT for safety
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT"):
        return "Error: Only SELECT queries are permitted."
    try:
        from sqlalchemy import create_engine, text
        from orchestrator.config import get_settings
        engine = create_engine(get_settings().database_url)
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).fetchall()
        if not rows:
            return "Query returned no rows."
        header = " | ".join(str(k) for k in rows[0]._fields)
        body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
        return f"{header}\n{'-' * len(header)}\n{body}"
    except Exception as e:
        return f"Database error: {e}"
