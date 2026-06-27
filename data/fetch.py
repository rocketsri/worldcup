"""Download and cache WC match data from ESPN API.
Source: site.api.espn.com — 2006-2026 (recent tournaments only).
Intentional scope: old WC data (pre-2006) is not useful for predicting modern teams
whose players, coaches, and staff have completely changed.
2026 WC has 104 scheduled games (72 group stage + 32 knockout); only completed
games are returned, so partial-tournament evaluation is supported naturally.
"""
import json
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ESPN_WC_YEARS = [2006, 2010, 2014, 2018, 2022, 2026]
STAGE_MAP = {
    "Group Stage": "group", "Group A": "group", "Group B": "group",
    "Group C": "group", "Group D": "group", "Group E": "group",
    "Group F": "group", "Group G": "group", "Group H": "group",
    "Round of 16": "round of 16", "Round Of 16": "round of 16",
    "Quarterfinals": "quarter-final", "Quarter-finals": "quarter-final",
    "Semifinals": "semi-final", "Semi-finals": "semi-final",
    "Third Place": "third-place", "Third-place": "third-place",
    "Final": "final",
}


def _parse_espn_event(event: dict, year: int) -> dict | None:
    comp = event["competitions"][0]
    if comp["status"]["type"]["state"] != "post":
        return None
    competitors = comp["competitors"]
    home = next((c for c in competitors if c["homeAway"] == "home"), None)
    away = next((c for c in competitors if c["homeAway"] == "away"), None)
    if not home or not away:
        return None
    try:
        home_score = int(home["score"])
        away_score = int(away["score"])
    except (KeyError, ValueError):
        return None

    raw_stage = comp.get("altGameNote", "")
    for key, val in STAGE_MAP.items():
        if key.lower() in raw_stage.lower():
            stage = val
            break
    else:
        stage = "group"

    return {
        "date": event["date"][:10],
        "home_team": home["team"]["displayName"],
        "away_team": away["team"]["displayName"],
        "home_score": home_score,
        "away_score": away_score,
        "tournament": "FIFA World Cup",
        "stage": stage,
        "neutral": True,
        "year": year,
    }


def fetch_espn_wc(year: int) -> list[dict]:
    cache = DATA_DIR / f"espn_wc_{year}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/FIFA.WORLD/scoreboard?dates={year}&limit=200"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    events = r.json().get("events", [])
    matches = [m for e in events if (m := _parse_espn_event(e, year))]
    cache.write_text(json.dumps(matches, indent=2))
    return matches


def load_all_wc_matches() -> pd.DataFrame:
    """Return DataFrame of WC matches 2006-2026 from ESPN (320 historical + 2026 completed games)."""
    cache = DATA_DIR / "all_wc_matches.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["date"])
        return df

    all_matches = []
    for year in ESPN_WC_YEARS:
        matches = fetch_espn_wc(year)
        all_matches.extend(matches)
        print(f"  WC {year}: {len(matches)} games loaded")

    df = pd.DataFrame(all_matches)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


def load_extended_matches() -> pd.DataFrame:
    """Load WC + Confederations Cup 1994-2022 from jfjelstul dataset (758 games).
    Gives ~2.4x more ELO signal vs WC-only (320 games). Includes proper
    tournament weighting: WC/Confed Cup are both FIFA senior competitions."""
    cache = DATA_DIR / "all_wc_matches_1994.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])
    raise FileNotFoundError(
        "Extended dataset not found. Run scripts/download_extended.py first."
    )


def get_wc_matches(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[df["year"] == year].copy()


def get_matches_before(df: pd.DataFrame, date) -> pd.DataFrame:
    return df[df["date"] < pd.Timestamp(date)].copy()
