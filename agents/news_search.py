"""News/forum search for pre-match context (2026 predictions only).
Filters queries to avoid score reveals — only fetches pre-match news.
Uses ddgs library (pip install ddgs) which handles bot-detection reliably.
"""
import re

SCORE_PATTERNS = re.compile(
    r'\b\d+[–\-]\d+\b|\bfinal score\b|\bscored?\b.*\bgoal\b|\bwon\b|\blost\b|\bdefeated\b',
    re.IGNORECASE,
)
RELEVANT_TERMS = re.compile(
    r'injury|injured|suspend|lineup|squad|form|preview|prediction|odds|'
    r'tactics|key player|missing|doubt|fitness|transfer|morale|pressure|'
    r'reddit|analysis|record|history|tournament|round|knockout',
    re.IGNORECASE,
)


def _clean_snippet(text: str) -> str:
    """Remove likely score-reveal sentences from a snippet."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = [s for s in sentences if not SCORE_PATTERNS.search(s)]
    return " ".join(cleaned).strip()


def ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """Search using ddgs library (handles bot detection)."""
    try:
        from ddgs import DDGS
        with DDGS() as d:
            raw = list(d.text(query, max_results=max_results * 2))
        results = []
        for r in raw:
            snippet = r.get("body", "")
            title = r.get("title", "")
            cleaned = _clean_snippet(snippet)
            if len(cleaned) < 30:
                continue
            results.append({"title": title, "snippet": cleaned})
            if len(results) >= max_results:
                break
        return results
    except Exception as e:
        return [{"title": "Search error", "snippet": str(e)}]


def search_prematch_news(team_a: str, team_b: str, match_date: str) -> str:
    """
    Search for pre-match news about team_a vs team_b without revealing the result.
    Queries: injuries, lineups, form, previews. Strips score-reveal sentences.
    """
    queries = [
        f"{team_a} {team_b} World Cup 2026 preview injury lineup",
        f"{team_a} vs {team_b} 2026 World Cup prediction analysis",
        f"{team_a} 2026 World Cup squad injuries form",
        f"{team_b} 2026 World Cup squad injuries form",
    ]

    all_snippets = []
    for q in queries[:3]:  # limit to 3 queries to stay fast
        results = ddg_search(q, max_results=3)
        for r in results:
            if RELEVANT_TERMS.search(r["snippet"] + " " + r["title"]):
                all_snippets.append(f"[{r['title']}] {r['snippet']}")

    if not all_snippets:
        return f"No relevant pre-match news found for {team_a} vs {team_b}"

    return (
        f"Pre-match news for {team_a} vs {team_b} (score-filtered):\n" +
        "\n".join(f"  • {s}" for s in all_snippets[:6])
    )


SEARCH_TOOL_ANTHROPIC = {
    "name": "search_prematch_news",
    "description": (
        "Search for pre-match news, injury reports, lineup announcements, and forum analysis "
        "for the two teams. Score reveals are filtered out automatically."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query_hint": {
                "type": "string",
                "description": "Optional extra keywords to focus the search (e.g. 'injury Mbappe')",
            }
        },
        "required": [],
    },
}

SEARCH_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "search_prematch_news",
        "description": "Search for pre-match news, injury reports, lineup announcements for both teams. Score reveals filtered.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_hint": {"type": "string"},
            },
            "required": [],
        },
    },
}
