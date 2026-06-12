"""
Sandboxed code executor — runs Python in a subprocess with timeout.
In production, replace with a proper sandbox (e.g., Firecracker, E2B.dev).
"""
import subprocess
import sys
import tempfile
from langchain_core.tools import tool


@tool
def code_executor_tool(code: str, timeout: int = 30) -> str:
    """
    Execute Python code in a sandboxed subprocess and return stdout/stderr.

    Args:
        code: Python code to execute.
        timeout: Max execution time in seconds (default 30).

    Returns:
        Combined stdout and stderr from the execution.
    """
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            # TODO: add seccomp/namespace isolation for production
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Execution timed out after {timeout}s"
    except Exception as e:
        return f"Execution error: {e}"
    finally:
        import os
        os.unlink(tmp_path)
