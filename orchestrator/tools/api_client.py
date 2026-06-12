"""Generic HTTP API client tool."""
import httpx
from langchain_core.tools import tool


@tool
def api_call_tool(url: str, method: str = "GET", headers: dict = None, body: dict = None) -> str:
    """
    Make an HTTP request to an external API.

    Args:
        url:     Full URL to call.
        method:  HTTP method: GET, POST, PUT (default GET).
        headers: Optional dict of request headers.
        body:    Optional JSON body (for POST/PUT).

    Returns:
        Response body as a string (truncated to 4000 chars).
    """
    try:
        with httpx.Client(timeout=20) as client:
            response = client.request(
                method=method.upper(),
                url=url,
                headers=headers or {},
                json=body,
            )
            text = response.text[:4000]
            return f"[{response.status_code}] {text}"
    except Exception as e:
        return f"API call failed: {e}"
