"""Web search tool — uses DuckDuckGo (free). Swap for Tavily/SerpAPI in prod."""
from langchain_core.tools import tool


@tool
def web_search_tool(query: str, max_results: int = 5) -> str:
    """
    Search the web for information about a query.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        A formatted string of search results with titles, URLs, and snippets.
    """
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"**{r['title']}**\n{r['href']}\n{r['body']}\n")
        return "\n---\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search failed: {e}"
