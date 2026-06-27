"""
Rolling ELO ratings computed from WC match history (2006-2022).
Design decisions:
- Only recent WC data (2006+): older tournaments reflect different eras of football
- K=40 (higher than typical 20-32): WC games are few but high-stakes, need faster updates
- Time-decay between tournaments: ratings drift 15% toward mean between WCs
  (captures: player retirements, new coach, squad overhaul between 4-year cycles)
- Goal-difference multiplier: big wins carry more signal
"""
import pandas as pd
import numpy as np

K = 40
HOME_ADVANTAGE = 75  # pts; WC is mostly neutral so small
DEFAULT_ELO = 1500
TOURNAMENT_DECAY = 0.85  # rating regresses 15% toward mean between WC cycles

# K-factor multipliers for international tournament importance
# Source: adapted from World Football Elo Ratings methodology
TOURNAMENT_K_WEIGHT = {
    "FIFA World Cup": 2.0,
    "Copa América": 1.5,
    "UEFA Euro": 1.5,
    "African Cup of Nations": 1.5,
    "AFC Asian Cup": 1.5,
    "Gold Cup": 1.2,
    "CONCACAF Nations League": 1.2,
    "UEFA Nations League": 1.2,
    "FIFA World Cup qualification": 1.0,
    "UEFA Euro qualification": 1.0,
    "Friendly": 0.5,  # friendlies carry less signal — rotated squads, low stakes
}


def compute_elo_history(df: pd.DataFrame) -> dict[str, list[tuple]]:
    """
    Returns {team: [(date, elo_before_match), ...]} computed chronologically.
    Applies between-tournament decay: ratings regress toward 1500 between WC cycles
    to capture squad/coaching changes that happen over 4 years.
    """
    ratings: dict[str, float] = {}
    history: dict[str, list] = {}
    prev_year = None

    for _, row in df.sort_values("date").iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs = row["home_score"]
        as_ = row["away_score"]
        date = row["date"]
        neutral = bool(row.get("neutral", True))
        cur_year = date.year

        # Apply between-tournament decay when we cross into a new WC year
        if prev_year is not None and cur_year != prev_year:
            for team in list(ratings.keys()):
                ratings[team] = DEFAULT_ELO + TOURNAMENT_DECAY * (ratings[team] - DEFAULT_ELO)
        prev_year = cur_year

        r_h = ratings.get(home, DEFAULT_ELO)
        r_a = ratings.get(away, DEFAULT_ELO)

        home_adv = 0 if neutral else HOME_ADVANTAGE
        dr = (r_h + home_adv - r_a) / 400.0
        e_h = 1.0 / (1.0 + 10.0 ** (-dr))

        if hs > as_:
            s_h = 1.0
        elif hs < as_:
            s_h = 0.0
        else:
            s_h = 0.5

        gd = abs(hs - as_)
        if gd <= 1:
            w = 1.0
        elif gd == 2:
            w = 1.5
        else:
            w = min(1.75 + (gd - 3) * 0.25, 3.0)

        delta = K * w * (s_h - e_h)

        history.setdefault(home, []).append((date, r_h))
        history.setdefault(away, []).append((date, r_a))

        ratings[home] = r_h + delta
        ratings[away] = r_a - delta

    return history, ratings


def compute_elo_history_international(df_intl: pd.DataFrame) -> tuple[dict, dict]:
    """
    Compute ELO from ALL international results (post-1990, ~32k games).
    Uses tournament-weighted K-factor: WC games count 4x more than friendlies.
    No between-tournament decay (games are frequent enough to self-regulate).
    Returns same format as compute_elo_history() for drop-in compatibility.
    """
    ratings: dict[str, float] = {}
    history: dict[str, list] = {}

    for _, row in df_intl.sort_values("date").iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs = row["home_score"]
        as_ = row["away_score"]
        date = row["date"]
        neutral = str(row.get("neutral", "TRUE")).upper() == "TRUE"
        tournament = str(row.get("tournament", "Friendly"))

        k_weight = TOURNAMENT_K_WEIGHT.get(tournament, 0.8)
        k_eff = K * k_weight

        r_h = ratings.get(home, DEFAULT_ELO)
        r_a = ratings.get(away, DEFAULT_ELO)

        home_adv = 0 if neutral else HOME_ADVANTAGE
        dr = (r_h + home_adv - r_a) / 400.0
        e_h = 1.0 / (1.0 + 10.0 ** (-dr))

        s_h = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)

        gd = abs(hs - as_)
        if gd <= 1:
            w = 1.0
        elif gd == 2:
            w = 1.5
        else:
            w = min(1.75 + (gd - 3) * 0.25, 3.0)

        delta = k_eff * w * (s_h - e_h)

        history.setdefault(home, []).append((date, r_h))
        history.setdefault(away, []).append((date, r_a))

        ratings[home] = r_h + delta
        ratings[away] = r_a - delta

    return history, ratings


def get_elo_at(history: dict, team: str, before_date) -> float:
    """ELO rating for team just before before_date."""
    cutoff = pd.Timestamp(before_date)
    entries = history.get(team, [])
    prior = [r for d, r in entries if d < cutoff]
    return prior[-1] if prior else DEFAULT_ELO
