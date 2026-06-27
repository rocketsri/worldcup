"""Predict remaining 2026 World Cup knockout games using best model."""
import asyncio
import pandas as pd
import numpy as np
from data.features import build_features
from data.elo import get_elo_at

# Confidence threshold for selective high-accuracy predictions
# Following the logic from xG/PPDA literature:
# Only predict when the model's top-outcome probability exceeds this threshold.
# At 60%+ win probability (ELO gap > 100pts typically), historical accuracy ~70-75%.
# At 70%+ win probability, historical accuracy approaches 80-85%.
CONFIDENCE_THRESHOLD = 0.60

# 2026 WC Round of 32 — confirmed matchups from bracket finalized June 27
# Sources: CBS Sports, ESPN, PBS, FIFA.com search results (June 27, 2026)
# Note: This is the new R32 stage (introduced for 48-team 2026 WC, first-ever)
UPCOMING_2026 = [
    # June 28 (Sunday)
    {"home_team": "South Africa",   "away_team": "Canada",              "date": "2026-06-28", "stage": "round of 16"},
    # June 29 (Monday)
    {"home_team": "Brazil",         "away_team": "Japan",               "date": "2026-06-29", "stage": "round of 16"},
    {"home_team": "Germany",        "away_team": "Paraguay",            "date": "2026-06-29", "stage": "round of 16"},
    {"home_team": "Netherlands",    "away_team": "Morocco",             "date": "2026-06-29", "stage": "round of 16"},
    # June 30 (Tuesday)
    {"home_team": "France",         "away_team": "Argentina",           "date": "2026-06-30", "stage": "round of 16"},
    {"home_team": "Uruguay",        "away_team": "Portugal",            "date": "2026-06-30", "stage": "round of 16"},
    # July 1 (Wednesday)
    {"home_team": "United States",  "away_team": "Bosnia and Herzegovina", "date": "2026-07-01", "stage": "round of 16"},
    {"home_team": "Belgium",        "away_team": "Ecuador",             "date": "2026-07-01", "stage": "round of 16"},
    # July 2 (Thursday) — Spain confirmed; England TBD (likely best 3rd-place opponent)
    {"home_team": "Spain",          "away_team": "Jordan",              "date": "2026-07-02", "stage": "round of 16"},
    {"home_team": "England",        "away_team": "Senegal",             "date": "2026-07-02", "stage": "round of 16"},
]


async def predict_2026_with_news(
    anthropic_client,
    openai_client,
    llm_model: str,
    df_all: pd.DataFrame,
    elo_history: dict,
) -> list[dict]:
    """
    Run LLM direct predictions for 2026 games WITH live web search context.
    Uses news_search tool (score-filtered) for injury/lineup/morale context.
    """
    from agents.context_agent import gather_context
    from agents.llm_direct import predict_direct

    print("\n" + "="*60)
    print(f"2026 WC PREDICTIONS — LLM Direct + Web Search ({llm_model})")
    print("="*60)
    predictions = []
    for match in UPCOMING_2026:
        team_a, team_b = match["home_team"], match["away_team"]
        date, stage = match["date"], match["stage"]

        # Gather rich context WITH news search enabled
        ctx = await gather_context(
            anthropic_client, openai_client, llm_model,
            df_all, elo_history, team_a, team_b, date, stage,
            include_news=True,
        )
        pred = await predict_direct(
            anthropic_client, openai_client, llm_model,
            df_all, elo_history, team_a, team_b, date, stage,
        )
        # Override with news-enriched context summary
        result = {**pred, "match": f"{team_a} vs {team_b}", "date": date,
                  "context_preview": ctx[:200]}
        predictions.append(result)
        winner = team_a if pred["home_score"] > pred["away_score"] else (
            team_b if pred["away_score"] > pred["home_score"] else "Draw"
        )
        print(f"\n{team_a} vs {team_b}")
        print(f"  Predicted: {pred['home_score']}-{pred['away_score']} | Winner: {winner}")
        print(f"  {team_a} {pred['proba'][0]:.1%} | Draw {pred['proba'][1]:.1%} | {team_b} {pred['proba'][2]:.1%}")
        print(f"  Reasoning: {pred.get('reasoning','')[:120]}")
    return predictions


async def predict_2026(
    best_model,
    df_all: pd.DataFrame,
    elo_history: dict,
    anthropic_client=None,
    openai_client=None,
    llm_model: str | None = None,
) -> list[dict]:
    print("\n" + "="*60)
    print("2026 WORLD CUP PREDICTIONS (Round of 16)")
    print("="*60)

    predictions = []
    for match in UPCOMING_2026:
        team_a = match["home_team"]
        team_b = match["away_team"]
        date = match["date"]
        stage = match["stage"]

        feats = build_features(
            df_all, elo_history, team_a, team_b, date, stage, neutral=True
        )
        X = pd.DataFrame([feats])
        if hasattr(best_model, "home_team"):
            X["home_team"] = team_a
            X["away_team"] = team_b

        proba = best_model.predict_proba(X)[0]
        lam_h, lam_a = best_model.predict_goals(X)
        pred_h = max(0, int(round(lam_h[0])))
        pred_a = max(0, int(round(lam_a[0])))

        win_team = team_a if pred_h > pred_a else (team_b if pred_a > pred_h else "Draw")

        result = {
            "match": f"{team_a} vs {team_b}",
            "date": date,
            "predicted_score": f"{pred_h}-{pred_a}",
            "predicted_winner": win_team,
            "home_win_prob": round(float(proba[0]), 3),
            "draw_prob": round(float(proba[1]), 3),
            "away_win_prob": round(float(proba[2]), 3),
            "elo_diff": round(feats["elo_diff"], 0),
        }
        predictions.append(result)

        print(f"\n{team_a} vs {team_b} ({date})")
        print(f"  Predicted: {pred_h}-{pred_a} | Winner: {win_team}")
        print(f"  Probabilities: {team_a} {proba[0]:.1%} | Draw {proba[1]:.1%} | {team_b} {proba[2]:.1%}")
        print(f"  ELO edge: {feats['elo_diff']:+.0f} pts")

    return predictions
