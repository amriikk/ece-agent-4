"""
web_search.py — DuckDuckGo search with query cleaning, result filtering,
fallback retry, and grounded answer synthesis.
"""

import re
import anthropic
from dataclasses import dataclass
from typing import List


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


# ── Low-quality domains to filter out ────────────────────────────────────────
_JUNK_DOMAINS = {
    "dictionary.com", "merriam-webster.com", "thesaurus.com",
    "vocabulary.com", "thefreedictionary.com", "cambridge.org/dictionary",
    "collinsdictionary.com", "macmillandictionary.com", "yourdictionary.com",
    "definitions.net", "synonyms.com",
}

def _is_junk(url: str) -> bool:
    return any(d in url for d in _JUNK_DOMAINS)


# ── Query cleaner ─────────────────────────────────────────────────────────────
def _clean_query(question: str) -> str:
    """
    'What caused the 2016 oil price crash?' → '2016 oil price crash causes'
    """
    q = question.strip().rstrip("?").strip()
    q = re.sub(
        r"^(what|why|how|who|when|where|can you|could you|please|tell me|explain|describe)\s+",
        "", q, flags=re.IGNORECASE
    ).strip()
    q = re.sub(
        r"^(caused?|is|are|was|were|do|does|did|the|a|an)\s+",
        "", q, flags=re.IGNORECASE
    ).strip()
    return q or question


# ── Core search ───────────────────────────────────────────────────────────────
def _get_ddgs():
    """
    Import DDGS from whichever package name is installed.
    Supports both the new 'ddgs' package and the legacy 'duckduckgo_search'.
    """
    try:
        from ddgs import DDGS
        return DDGS
    except ImportError:
        pass
    try:
        from duckduckgo_search import DDGS
        return DDGS
    except ImportError:
        raise ImportError("Run: pip install ddgs")


def search(question: str, max_results: int = 6) -> List[SearchResult]:
    """
    Search DuckDuckGo with automatic query cleaning and junk filtering.
    Retries with the original question if cleaned query returns no good results.
    """
    DDGS = _get_ddgs()

    queries_to_try = [_clean_query(question)]
    if queries_to_try[0] != question.strip().rstrip("?"):
        queries_to_try.append(question)

    for query in queries_to_try:
        try:
            raw = []
            with DDGS() as ddgs:
                for r in ddgs.text(
                    query,                          # positional — works in both old and new ddgs
                    region      = "wt-wt",
                    safesearch  = "off",
                    max_results = max_results + 3,
                ):
                    raw.append(r)

            results = []
            for r in raw:
                url = r.get("href", "")
                if _is_junk(url):
                    continue
                results.append(SearchResult(
                    title   = r.get("title", ""),
                    url     = url,
                    snippet = r.get("body", ""),
                ))
                if len(results) >= max_results:
                    break

            if results:
                return results

        except Exception as e:
            print(f"DDG search error (query='{query}'): {e}")
            continue

    return []


# ── Answer synthesis ──────────────────────────────────────────────────────────
def synthesize_answer(question: str, results: List[SearchResult]) -> dict:
    if not results:
        return {
            "answer": (
                "I couldn't find relevant search results for this question. "
                "Try rephrasing or asking a more specific question."
            ),
            "citations": [],
        }

    client  = anthropic.Anthropic()
    context = "\n\n".join(
        f"[{i}] {r.title}\n{r.url}\n{r.snippet}"
        for i, r in enumerate(results, 1)
    )

    response = client.messages.create(
        model      = "claude-sonnet-4-6",
        max_tokens = 600,
        messages   = [{"role": "user", "content": (
            f"Answer the following question using ONLY the search results below.\n\n"
            f"Question: {question}\n\n"
            f"Search results:\n{context}\n\n"
            f"Instructions:\n"
            f"- Write 3-5 clear, factual sentences directly answering the question.\n"
            f"- Cite sources inline as [1], [2], etc. after each claim.\n"
            f"- If the results don't fully answer the question, say so briefly.\n"
            f"- Do NOT say 'based on the search results' — just answer directly.\n"
            f"- Do NOT use bullet points."
        )}],
    )

    return {
        "answer":    response.content[0].text.strip(),
        "citations": [{"title": r.title, "url": r.url} for r in results],
    }


# ── Public entry point ────────────────────────────────────────────────────────
def run(question: str) -> dict:
    """Full pipeline: question → search → grounded answer with citations."""
    results   = search(question)
    synthesis = synthesize_answer(question, results)
    return {**synthesis, "raw_results": results}