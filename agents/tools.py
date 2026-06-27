"""Tool implementations for the LLM context-gathering agent."""
import pandas as pd
import json
from data.features import team_form, head_to_head
from data.elo import get_elo_at


def get_team_recent_form(
    df: pd.DataFrame, team: str, before_date: str, n: int = 8
) -> str:
    """Return last N results for team before before_date."""
    cutoff = pd.Timestamp(before_date)
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < cutoff)
    )
    recent = df[mask].sort_values("date", ascending=False).head(n)
    if len(recent) == 0:
        return f"No historical data found for {team} before {before_date}"

    lines = [f"Last {len(recent)} matches for {team}:"]
    for _, row in recent.iterrows():
        is_home = row["home_team"] == team
        opp = row["away_team"] if is_home else row["home_team"]
        gs = row["home_score"] if is_home else row["away_score"]
        gc = row["away_score"] if is_home else row["home_score"]
        venue = "vs" if is_home else "@"
        result = "W" if gs > gc else ("D" if gs == gc else "L")
        lines.append(f"  {row['date'].date()} {result} {venue} {opp}: {gs}-{gc}")
    return "\n".join(lines)


def get_head_to_head_results(
    df: pd.DataFrame, team_a: str, team_b: str, before_date: str, n: int = 8
) -> str:
    """Return H2H record between team_a and team_b."""
    cutoff = pd.Timestamp(before_date)
    mask = (
        ((df["home_team"] == team_a) & (df["away_team"] == team_b)) |
        ((df["home_team"] == team_b) & (df["away_team"] == team_a))
    ) & (df["date"] < cutoff)
    h2h = df[mask].sort_values("date", ascending=False).head(n)

    if len(h2h) == 0:
        return f"No head-to-head record found between {team_a} and {team_b}"

    wins_a = draws = wins_b = 0
    lines = [f"Head-to-head: {team_a} vs {team_b} (last {len(h2h)} meetings):"]
    for _, row in h2h.iterrows():
        if row["home_team"] == team_a:
            gs, gc = row["home_score"], row["away_score"]
        else:
            gs, gc = row["away_score"], row["home_score"]
        result = "W" if gs > gc else ("D" if gs == gc else "L")
        if gs > gc: wins_a += 1
        elif gs == gc: draws += 1
        else: wins_b += 1
        lines.append(f"  {row['date'].date()} {team_a} {gs}-{gc} {team_b} ({result})")
    lines.append(f"Record: {team_a} {wins_a}W {draws}D {wins_b}L")
    return "\n".join(lines)


def get_elo_ratings(
    elo_history: dict, team_a: str, team_b: str, before_date: str
) -> str:
    """Return ELO ratings for both teams."""
    elo_a = get_elo_at(elo_history, team_a, before_date)
    elo_b = get_elo_at(elo_history, team_b, before_date)
    diff = elo_a - elo_b
    return (
        f"ELO ratings as of {before_date}:\n"
        f"  {team_a}: {elo_a:.0f}\n"
        f"  {team_b}: {elo_b:.0f}\n"
        f"  Difference (favoring {team_a if diff>0 else team_b}): {abs(diff):.0f} pts"
    )


TOOL_SCHEMAS_ANTHROPIC = [
    {
        "name": "get_team_recent_form",
        "description": "Get the recent match results for a team (wins/losses/draws and scores)",
        "input_schema": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "description": "Team name"},
                "n": {"type": "integer", "description": "Number of recent matches", "default": 8},
            },
            "required": ["team"],
        },
    },
    {
        "name": "get_head_to_head",
        "description": "Get head-to-head results between two teams",
        "input_schema": {
            "type": "object",
            "properties": {
                "team_a": {"type": "string"},
                "team_b": {"type": "string"},
                "n": {"type": "integer", "default": 8},
            },
            "required": ["team_a", "team_b"],
        },
    },
    {
        "name": "get_elo_ratings",
        "description": "Get ELO strength ratings for both teams",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

TOOL_SCHEMAS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "get_team_recent_form",
            "description": "Get the recent match results for a team",
            "parameters": {
                "type": "object",
                "properties": {
                    "team": {"type": "string"},
                    "n": {"type": "integer", "default": 8},
                },
                "required": ["team"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_head_to_head",
            "description": "Get head-to-head results between two teams",
            "parameters": {
                "type": "object",
                "properties": {
                    "team_a": {"type": "string"},
                    "team_b": {"type": "string"},
                    "n": {"type": "integer", "default": 8},
                },
                "required": ["team_a", "team_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_elo_ratings",
            "description": "Get ELO strength ratings for both teams",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]
