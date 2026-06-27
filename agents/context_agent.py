"""LLM agent that gathers match context via tool calls.
For live 2026 predictions: includes web search for news/injuries/lineups.
For historical backtesting: web search disabled to prevent data leakage.
"""
import json
import asyncio
import pandas as pd
from agents.tools import (
    get_team_recent_form, get_head_to_head_results,
    get_elo_ratings, TOOL_SCHEMAS_ANTHROPIC, TOOL_SCHEMAS_OPENAI,
)
from agents.news_search import (
    search_prematch_news, SEARCH_TOOL_ANTHROPIC, SEARCH_TOOL_OPENAI,
)

ANTHROPIC_MODELS = {"claude-sonnet-4-6", "claude-opus-4-8"}
MAX_TOOL_CALLS = 6


def _make_system_prompt(team_a: str, team_b: str, date: str, stage: str,
                        include_news: bool = False) -> str:
    news_line = (
        "\n4. Current news: injuries, suspensions, lineup hints, team morale "
        "(use search_prematch_news — scores are filtered automatically)"
        if include_news else ""
    )
    return (
        f"You are an expert football analyst. Your job is to gather relevant evidence "
        f"to predict the outcome of a FIFA World Cup match: {team_a} vs {team_b} "
        f"({stage}) on {date}.\n\n"
        f"Use the available tools to look up:\n"
        f"1. Recent form for both teams\n"
        f"2. Head-to-head record\n"
        f"3. ELO strength ratings{news_line}\n\n"
        f"IMPORTANT: Do NOT recall or speculate about the specific result of this match. "
        f"Analyze purely from the data the tools return.\n\n"
        f"After gathering data, provide a concise analytical summary covering team strength, "
        f"recent momentum, key advantages, and (if available) injury/lineup concerns."
    )


async def gather_context_anthropic(
    client, model: str,
    df: pd.DataFrame, elo_history: dict,
    team_a: str, team_b: str, match_date: str, stage: str,
    include_news: bool = False,
) -> str:
    system = _make_system_prompt(team_a, team_b, match_date, stage, include_news)
    messages = [{"role": "user", "content": f"Analyze: {team_a} vs {team_b} on {match_date}"}]

    tools = TOOL_SCHEMAS_ANTHROPIC + ([SEARCH_TOOL_ANTHROPIC] if include_news else [])
    kwargs = {"max_tokens": 1024, "tools": tools}

    for _ in range(MAX_TOOL_CALLS):
        resp = await asyncio.get_event_loop().run_in_executor(
            None, lambda: client.messages.create(
                model=model, system=system, messages=messages, **kwargs
            )
        )
        # Collect assistant message
        assistant_content = []
        tool_calls_made = False

        for block in resp.content:
            assistant_content.append(block)
            if block.type == "tool_use":
                tool_calls_made = True

        messages.append({"role": "assistant", "content": assistant_content})

        if not tool_calls_made or resp.stop_reason == "end_turn":
            break

        # Process tool calls
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            args = block.input
            name = block.name
            if name == "get_team_recent_form":
                result = get_team_recent_form(df, args.get("team", team_a), match_date, args.get("n", 8))
            elif name == "get_head_to_head":
                result = get_head_to_head_results(df, args.get("team_a", team_a), args.get("team_b", team_b), match_date, args.get("n", 8))
            elif name == "get_elo_ratings":
                result = get_elo_ratings(elo_history, team_a, team_b, match_date)
            elif name == "search_prematch_news":
                result = search_prematch_news(team_a, team_b, match_date)
            else:
                result = "Unknown tool"
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

        messages.append({"role": "user", "content": tool_results})

    # Extract final text
    for block in resp.content:
        if hasattr(block, "text"):
            return block.text
    return "No context gathered"


async def gather_context_openai(
    client, model: str,
    df: pd.DataFrame, elo_history: dict,
    team_a: str, team_b: str, match_date: str, stage: str,
) -> str:
    system = _make_system_prompt(team_a, team_b, match_date, stage)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Analyze: {team_a} vs {team_b} on {match_date}"},
    ]

    is_o_series = model.startswith("o3") or model.startswith("o1")
    kwargs = {"model": model}
    if is_o_series:
        kwargs["reasoning_effort"] = "high"
    else:
        kwargs["tools"] = TOOL_SCHEMAS_OPENAI
    token_kw = {"max_completion_tokens": 1024} if is_o_series else {"max_tokens": 1024}

    for _ in range(MAX_TOOL_CALLS):
        resp = await asyncio.get_event_loop().run_in_executor(
            None, lambda: client.chat.completions.create(
                messages=messages, **token_kw, **kwargs
            )
        )
        msg = resp.choices[0].message
        messages.append(msg)

        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls or resp.choices[0].finish_reason in ("stop", "end_turn"):
            break

        for tc in tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            if name == "get_team_recent_form":
                result = get_team_recent_form(df, args.get("team", team_a), match_date, args.get("n", 8))
            elif name == "get_head_to_head":
                result = get_head_to_head_results(df, args.get("team_a", team_a), args.get("team_b", team_b), match_date, args.get("n", 8))
            elif name == "get_elo_ratings":
                result = get_elo_ratings(elo_history, team_a, team_b, match_date)
            else:
                result = "Unknown tool"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return msg.content or "No context gathered"


async def gather_context(
    anthropic_client, openai_client,
    model: str,
    df: pd.DataFrame, elo_history: dict,
    team_a: str, team_b: str, match_date: str, stage: str,
    include_news: bool = False,
) -> str:
    if model in ANTHROPIC_MODELS:
        return await gather_context_anthropic(
            anthropic_client, model, df, elo_history, team_a, team_b, match_date, stage,
            include_news=include_news,
        )
    else:
        return await gather_context_openai(
            openai_client, model, df, elo_history, team_a, team_b, match_date, stage
        )
