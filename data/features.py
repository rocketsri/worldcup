"""Feature engineering for match prediction."""
import pandas as pd
import numpy as np
from data.elo import get_elo_at, DEFAULT_ELO

STAGE_WEIGHTS = {
    "group": 1.0, "round of 16": 1.2, "quarter-final": 1.3,
    "semi-final": 1.5, "third-place": 1.3, "final": 2.0,
}

# Co-host advantage for 2026 WC (USA/Canada/Mexico play in home stadiums)
# International soccer home advantage ≈ 50-70 ELO pts (smaller at neutral WC venues)
# Co-host advantage is intermediate (~40 pts) — partial crowd support but no travel benefit
WC_2026_COHOSTS = {"United States", "Canada", "Mexico"}
COHOST_ELO_BONUS = 40  # pts added to co-host ELO when they play on home soil in 2026

# Confederation quality index (relative to world mean) — based on WC win rates 2006-2022
# UEFA and CONMEBOL teams win ~60%+ of WC matches; AFC/CONCACAF/CAF/OFC lower
CONFEDERATION_ELO_PRIOR = {
    "UEFA": 80,       # Europe: historically strongest WC confederation
    "CONMEBOL": 60,   # South America: strong second
    "CONCACAF": -20,  # North/Central America: below average
    "CAF": -10,       # Africa: below average but improving
    "AFC": -30,       # Asia: weakest WC record
    "OFC": -60,       # Oceania: very rare WC participants
}

TEAM_CONFEDERATION = {
    # UEFA
    "France": "UEFA", "Germany": "UEFA", "Spain": "UEFA", "England": "UEFA",
    "Netherlands": "UEFA", "Belgium": "UEFA", "Portugal": "UEFA", "Italy": "UEFA",
    "Croatia": "UEFA", "Denmark": "UEFA", "Switzerland": "UEFA", "Poland": "UEFA",
    "Sweden": "UEFA", "Serbia": "UEFA", "Czech Republic": "UEFA", "Slovakia": "UEFA",
    "Austria": "UEFA", "Hungary": "UEFA", "Scotland": "UEFA", "Wales": "UEFA",
    "Turkey": "UEFA", "Greece": "UEFA", "Romania": "UEFA", "Ukraine": "UEFA",
    "Russia": "UEFA", "Slovenia": "UEFA", "Bosnia and Herzegovina": "UEFA",
    "North Macedonia": "UEFA", "Finland": "UEFA", "Iceland": "UEFA", "Albania": "UEFA",
    "Morocco": "CAF", "Senegal": "CAF", "Nigeria": "CAF", "Ghana": "CAF",
    "Cameroon": "CAF", "Ivory Coast": "CAF", "Egypt": "CAF", "Tunisia": "CAF",
    "Algeria": "CAF", "South Africa": "CAF",
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Peru": "CONMEBOL", "Paraguay": "CONMEBOL", "Bolivia": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    "United States": "CONCACAF", "Mexico": "CONCACAF", "Costa Rica": "CONCACAF",
    "Panama": "CONCACAF", "Honduras": "CONCACAF", "Jamaica": "CONCACAF",
    "Canada": "CONCACAF", "Trinidad and Tobago": "CONCACAF",
    "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC", "Saudi Arabia": "AFC",
    "Australia": "AFC", "Qatar": "AFC", "South Korea": "AFC",
    "New Zealand": "OFC",
}

# Geopolitical/historical rivalry scores: (team_a, team_b) → float [0, 1]
# Based on documented political tensions, historical confrontations, and cultural rivalries
# that sports psychology research shows can affect performance variance (not just win rate).
# Sources: Kuper & Szymanski "Soccernomics", academic studies on rivalry effects.
# Weight is intentionally small — treat as a tiebreaker / variance signal, not primary driver.
RIVALRY_SCORES: dict[frozenset, float] = {
    frozenset(["Argentina", "England"]):        0.85,  # Falklands War; Hand of God; generational defining rivalry
    frozenset(["USA", "Iran"]):                 0.90,  # Direct political adversaries; 1998 + 2022 WC tension
    frozenset(["USA", "Mexico"]):               0.65,  # Regional rivalry; immigration politics; CONCACAF dominance contest
    frozenset(["Germany", "England"]):          0.70,  # WWII legacy; '66 final; Penalty trauma
    frozenset(["France", "Germany"]):           0.60,  # Historical; European power contest
    frozenset(["Netherlands", "Germany"]):      0.75,  # WWII occupation memory; intense domestic rivalry
    frozenset(["Brazil", "Argentina"]):         0.80,  # Superclasico de las Americas; generational stars
    frozenset(["Spain", "Portugal"]):           0.50,  # Iberian derby; low geopolitical tension but intense rivalry
    frozenset(["South Korea", "Japan"]):        0.75,  # Historical occupation; cultural tension
    frozenset(["Israel", "any Arab nation"]):   0.95,  # Not applicable in WC yet (Israel in UEFA)
    frozenset(["Morocco", "Algeria"]):          0.70,  # Maghreb political tension
    frozenset(["Serbia", "Croatia"]):           0.90,  # Yugoslav Wars aftermath
    frozenset(["Serbia", "Bosnia and Herzegovina"]): 0.85,  # Same
    frozenset(["Russia", "Ukraine"]):           0.95,  # Active war context (2022 WC played without Russia)
}


def get_rivalry_score(team_a: str, team_b: str) -> float:
    """Return rivalry intensity [0, 1] for a pair. 0 = no special dynamics."""
    key = frozenset([team_a, team_b])
    return RIVALRY_SCORES.get(key, 0.0)


STAT_FEATURE_COLS = [
    "elo_diff", "elo_a", "elo_b",
    "form_a", "form_b", "form_diff",
    "goals_scored_a", "goals_conceded_a",
    "goals_scored_b", "goals_conceded_b",
    "h2h_win_rate_a", "h2h_games",
    "clean_sheet_rate_a", "clean_sheet_rate_b",
    "win_streak_a", "win_streak_b",
    "confederation_diff",
    "rivalry_score",      # geopolitical/historical rivalry intensity [0,1] — low weight, high variance signal
    "is_neutral", "stage_weight",
]
LLM_FEATURE_COLS = [
    "injury_impact", "motivation_factor", "tactical_edge",
    "news_sentiment_a", "news_sentiment_b",
]


def clean_sheet_rate(df: pd.DataFrame, team: str, before_date, n: int = 10) -> float:
    """Fraction of last n games where team conceded 0 goals."""
    cutoff = pd.Timestamp(before_date)
    mask = (((df["home_team"] == team) | (df["away_team"] == team)) & (df["date"] < cutoff))
    recent = df[mask].sort_values("date", ascending=False).head(n)
    if len(recent) == 0:
        return 0.3
    cs = 0
    for _, row in recent.iterrows():
        gc = row["away_score"] if row["home_team"] == team else row["home_score"]
        if gc == 0:
            cs += 1
    return cs / len(recent)


def win_streak(df: pd.DataFrame, team: str, before_date) -> int:
    """Current consecutive wins streak (negative = losing streak)."""
    cutoff = pd.Timestamp(before_date)
    mask = (((df["home_team"] == team) | (df["away_team"] == team)) & (df["date"] < cutoff))
    recent = df[mask].sort_values("date", ascending=False).head(8)
    streak = 0
    for _, row in recent.iterrows():
        is_home = row["home_team"] == team
        gs = row["home_score"] if is_home else row["away_score"]
        gc = row["away_score"] if is_home else row["home_score"]
        if gs > gc:
            if streak >= 0:
                streak += 1
            else:
                break
        elif gs < gc:
            if streak <= 0:
                streak -= 1
            else:
                break
        else:
            break
    return streak


def team_form(df: pd.DataFrame, team: str, before_date, n: int = 10) -> dict:
    cutoff = pd.Timestamp(before_date)
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < cutoff)
    )
    recent = df[mask].sort_values("date", ascending=False).head(n)
    if len(recent) == 0:
        return {"form": 0.5, "goals_scored": 1.2, "goals_conceded": 1.2}

    pts = goals_scored = goals_conceded = 0
    for _, row in recent.iterrows():
        is_home = row["home_team"] == team
        gs = row["home_score"] if is_home else row["away_score"]
        gc = row["away_score"] if is_home else row["home_score"]
        goals_scored += gs
        goals_conceded += gc
        if gs > gc:
            pts += 3
        elif gs == gc:
            pts += 1

    n_g = len(recent)
    return {
        "form": pts / (n_g * 3),
        "goals_scored": goals_scored / n_g,
        "goals_conceded": goals_conceded / n_g,
    }


def head_to_head(df: pd.DataFrame, team_a: str, team_b: str, before_date, n: int = 10) -> dict:
    cutoff = pd.Timestamp(before_date)
    mask = (
        ((df["home_team"] == team_a) & (df["away_team"] == team_b)) |
        ((df["home_team"] == team_b) & (df["away_team"] == team_a))
    ) & (df["date"] < cutoff)
    h2h = df[mask].sort_values("date", ascending=False).head(n)
    if len(h2h) == 0:
        return {"h2h_win_rate_a": 0.5, "h2h_games": 0}

    wins = 0.0
    for _, row in h2h.iterrows():
        if row["home_team"] == team_a:
            gs, gc = row["home_score"], row["away_score"]
        else:
            gs, gc = row["away_score"], row["home_score"]
        if gs > gc:
            wins += 1
        elif gs == gc:
            wins += 0.5

    return {"h2h_win_rate_a": wins / len(h2h), "h2h_games": len(h2h)}


_INTL_RANK_CACHE: dict[str, int] | None = None


def _get_intl_rank_lookup() -> dict[str, int]:
    """Lazily build team rank lookup from all-international ELO at 2022-01-01."""
    global _INTL_RANK_CACHE
    if _INTL_RANK_CACHE is not None:
        return _INTL_RANK_CACHE
    from pathlib import Path
    intl_path = Path(__file__).parent / "raw" / "international_results.csv"
    if not intl_path.exists():
        _INTL_RANK_CACHE = {}
        return _INTL_RANK_CACHE
    from data.elo import compute_elo_history_international, get_elo_at
    df_intl = pd.read_csv(intl_path, parse_dates=["date"])
    hist, _ = compute_elo_history_international(df_intl)
    cutoff = pd.Timestamp("2022-01-01")
    teams = set(df_intl["home_team"].tolist() + df_intl["away_team"].tolist())
    elos = {t: get_elo_at(hist, t, cutoff) for t in teams}
    sorted_teams = sorted(elos.items(), key=lambda x: x[1], reverse=True)
    _INTL_RANK_CACHE = {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}
    return _INTL_RANK_CACHE


def build_features(
    df: pd.DataFrame,
    elo_history: dict,
    team_a: str,
    team_b: str,
    match_date,
    stage: str = "group",
    neutral: bool = True,
    llm_features: dict | None = None,
    include_fifa_rank: bool = False,
) -> dict:
    cutoff = pd.Timestamp(match_date) - pd.Timedelta(days=1)
    match_year = pd.Timestamp(match_date).year

    elo_a = get_elo_at(elo_history, team_a, cutoff)
    elo_b = get_elo_at(elo_history, team_b, cutoff)

    # Apply co-host home advantage for 2026 WC
    if match_year == 2026:
        if team_a in WC_2026_COHOSTS:
            elo_a += COHOST_ELO_BONUS
        if team_b in WC_2026_COHOSTS:
            elo_b += COHOST_ELO_BONUS

    fa = team_form(df, team_a, cutoff)
    fb = team_form(df, team_b, cutoff)
    h2h = head_to_head(df, team_a, team_b, cutoff)

    cs_a = clean_sheet_rate(df, team_a, cutoff)
    cs_b = clean_sheet_rate(df, team_b, cutoff)
    ws_a = win_streak(df, team_a, cutoff)
    ws_b = win_streak(df, team_b, cutoff)

    conf_a = CONFEDERATION_ELO_PRIOR.get(TEAM_CONFEDERATION.get(team_a, ""), 0)
    conf_b = CONFEDERATION_ELO_PRIOR.get(TEAM_CONFEDERATION.get(team_b, ""), 0)

    feats = {
        "elo_diff": elo_a - elo_b,
        "elo_a": elo_a,
        "elo_b": elo_b,
        "form_a": fa["form"],
        "form_b": fb["form"],
        "form_diff": fa["form"] - fb["form"],
        "goals_scored_a": fa["goals_scored"],
        "goals_conceded_a": fa["goals_conceded"],
        "goals_scored_b": fb["goals_scored"],
        "goals_conceded_b": fb["goals_conceded"],
        "h2h_win_rate_a": h2h["h2h_win_rate_a"],
        "h2h_games": h2h["h2h_games"],
        "clean_sheet_rate_a": cs_a,
        "clean_sheet_rate_b": cs_b,
        "win_streak_a": float(ws_a),
        "win_streak_b": float(ws_b),
        "confederation_diff": float(conf_a - conf_b),
        "rivalry_score": get_rivalry_score(team_a, team_b),
        "is_neutral": int(neutral),
        "stage_weight": STAGE_WEIGHTS.get(stage, 1.0),
    }
    if include_fifa_rank:
        rank_lookup = _get_intl_rank_lookup()
        rank_a = rank_lookup.get(team_a, 100)
        rank_b = rank_lookup.get(team_b, 100)
        feats["fifa_rank_diff"] = float(rank_b - rank_a)  # positive = team_a is ranked higher (lower number)
    if llm_features:
        feats.update(llm_features)
    return feats


def build_dataset(
    df: pd.DataFrame,
    elo_history: dict,
    matches: pd.DataFrame,
    llm_features_map: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Build X, y_home, y_away, y_outcome for a set of matches."""
    rows = []
    y_home, y_away, y_outcome = [], [], []

    for _, row in matches.iterrows():
        match_id = f"{row['date'].date()}_{row['home_team']}_{row['away_team']}"
        llm_feats = (llm_features_map or {}).get(match_id)
        feats = build_features(
            df, elo_history,
            row["home_team"], row["away_team"],
            row["date"], row.get("stage", "group"),
            bool(row.get("neutral", True)),
            llm_feats,
        )
        rows.append(feats)
        y_home.append(row["home_score"])
        y_away.append(row["away_score"])
        if row["home_score"] > row["away_score"]:
            y_outcome.append(0)  # home win
        elif row["home_score"] < row["away_score"]:
            y_outcome.append(2)  # away win
        else:
            y_outcome.append(1)  # draw

    X = pd.DataFrame(rows)
    return X, pd.Series(y_home), pd.Series(y_away), pd.Series(y_outcome)
