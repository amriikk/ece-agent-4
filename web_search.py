"""
web_search.py — DuckDuckGo search wrapper for generic-domain questions.

Uses LangChain's DuckDuckGoSearchRun to retrieve results, then calls
Claude to synthesise a grounded answer with inline citations.
"""

import anthropic
from dataclasses import dataclass
from typing import List


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def search(query: str, max_results: int = 5) -> List[SearchResult]:
    """
    Run a DuckDuckGo search and return structured results.

    Install dependency: pip install duckduckgo-search langchain-community
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchResults
        tool = DuckDuckGoSearchResults(num_results=max_results, output_format="list")
        raw = tool.invoke(query)

        results = []
        for item in raw:
            results.append(SearchResult(
                title   = item.get("title", ""),
                url     = item.get("link", ""),
                snippet = item.get("snippet", ""),
            ))
        return results

    except ImportError:
        raise ImportError(
            "Missing dependency. Run: pip install duckduckgo-search langchain-community"
        )
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        return []


def synthesize_answer(question: str, results: List[SearchResult]) -> dict:
    """
    Call Claude to produce a grounded answer from search results.

    Returns:
        {
            "answer":    str — natural language answer with inline citations [1], [2], …
            "citations": List[{title, url}]
        }
    """
    if not results:
        return {
            "answer": "No search results were found for this query.",
            "citations": [],
        }

    client = anthropic.Anthropic()

    # Build context block
    context_lines = []
    for i, r in enumerate(results, 1):
        context_lines.append(
            f"[{i}] Title: {r.title}\n"
            f"    URL: {r.url}\n"
            f"    Snippet: {r.snippet}"
        )
    context = "\n\n".join(context_lines)

    prompt = (
        f"Use the following search results to answer the question.\n\n"
        f"Question: {question}\n\n"
        f"Search results:\n{context}\n\n"
        f"Instructions:\n"
        f"- Write a clear, factual answer (3–6 sentences).\n"
        f"- Cite your sources inline using [1], [2], etc.\n"
        f"- Only use information from the search results — do not invent facts.\n"
        f"- If the results don't fully answer the question, say so.\n"
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.content[0].text.strip()
    citations = [{"title": r.title, "url": r.url} for r in results]

    return {"answer": answer, "citations": citations}


def run(question: str) -> dict:
    """
    Full pipeline: question → DuckDuckGo search → grounded answer.

    Returns:
        {
            "answer":    str,
            "citations": List[{title, url}],
            "raw_results": List[SearchResult]
        }
    """
    results = search(question)
    synthesis = synthesize_answer(question, results)
    return {**synthesis, "raw_results": results}
