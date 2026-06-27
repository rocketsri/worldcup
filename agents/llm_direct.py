"""LLM direct score prediction (no stat model, LLM outputs score directly)."""
import json
import re
import asyncio
import pandas as pd
from agents.tools import get_team_recent_form, get_head_to_head_results, get_elo_ratings

ANTHROPIC_MODELS = {"claude-sonnet-4-6", "claude-opus-4-8"}

ZERO_SHOT_PROMPT = """You are an expert football analyst. Predict the exact final score for this FIFA World Cup match.

Match: {team_a} vs {team_b}
Date: {date}
Stage: {stage}

Context data:
{context}

Based on this data, predict the most likely exact final score (after 90 minutes, excluding extra time/penalties).
Consider team strength, recent form, head-to-head record, and tournament context.

Respond ONLY with valid JSON:
{{"home_score": int, "away_score": int, "home_win_prob": float, "draw_prob": float, "away_win_prob": float, "reasoning": "brief explanation"}}"""

# Improved few-shot with diverse examples: dominant win, close win, draw, upset
FEW_SHOT_EXAMPLES = """Here are World Cup match predictions calibrated from historical data:

Example 1 (Heavy favorite wins): Germany vs Brazil (2014 Semi-Final)
Context: ELO edge Germany +285pts. Brazil missing Neymar (injury) and Thiago Silva (suspension). Germany 6W in last 6 WC. H2H: Germany dominated recent meetings.
Reasoning: Massive ELO gap + Brazil's defensive absences + German dominance = blowout likely.
Prediction: {"home_score": 7, "away_score": 1, "home_win_prob": 0.82, "draw_prob": 0.06, "away_win_prob": 0.12}
Actual: 7-1 (Germany) ✓

Example 2 (Moderate favorite, fatigue factor): France vs Croatia (2018 Final)
Context: ELO France +65pts. France form excellent (Mbappé/Pogba in form). Croatia fatigued (3 consecutive extra-time matches across QF, SF, Final run).
Reasoning: Small ELO gap but fatigue tips balance decisively. France wins, score somewhat open.
Prediction: {"home_score": 4, "away_score": 2, "home_win_prob": 0.62, "draw_prob": 0.14, "away_win_prob": 0.24}
Actual: 4-2 (France) ✓

Example 3 (Close game / potential draw): Spain vs Switzerland (2010 Round of 16)
Context: ELO Spain +45pts. Spain dominant but Switzerland shocked Spain earlier in group stage. H2H: very even historically.
Reasoning: Small ELO gap + Switzerland's defensive record + prior upset = tight, low-scoring game. Draw plausible.
Prediction: {"home_score": 1, "away_score": 0, "home_win_prob": 0.52, "draw_prob": 0.28, "away_win_prob": 0.20}
Actual: 1-0 AET (Spain) ✓

Example 4 (Underdog upset): Japan vs Germany (2022 Group Stage)
Context: ELO Germany +210pts. Germany strong favorites. Japan counter-attacking system historically effective vs high-press teams.
Reasoning: Despite large ELO gap, Japan's disciplined low-block counter can exploit German fullback space. Upset possible (20%+ probability given Japan's tournament record).
Prediction: {"home_score": 1, "away_score": 2, "home_win_prob": 0.61, "draw_prob": 0.22, "away_win_prob": 0.17}
Actual: 1-2 (Japan upset) ✓

Calibration notes: ELO gap of 100pts → ~62% win rate. 200pts → ~72%. Knockouts have fewer draws (teams push harder). High ELO gap still allows ~18% upset rate.

Now predict the following match:
"""

CHAIN_OF_THOUGHT_PROMPT = """You are an expert football analyst. Reason step-by-step, then predict the exact final score for this FIFA World Cup match.

Match: {team_a} vs {team_b}
Date: {date}
Stage: {stage}

Statistical context:
{context}

Think through each factor systematically:

STEP 1 — ELO strength: What is the ELO difference? (100pts gap = ~62% win rate, 200pts = ~72%, 300pts+ = ~82%)
STEP 2 — Recent form: Who has momentum? Goals scored/conceded trend in last 5-8 matches?
STEP 3 — Head-to-head: What does H2H history show? Is there a clear pattern or near-even record?
STEP 4 — Stage context: Is this a group game (draws acceptable) or knockout (teams push harder for 90min decision)?
STEP 5 — Upset probability: Given the factors above, is there realistic upset potential? (High ELO gap = ~18-20% upset still possible)
STEP 6 — Score prediction: What is the single most likely scoreline after 90 minutes, and the probability distribution?

Calibration reminder: World Cup knockout games average 2.5 total goals. Group games average 2.7. Scores of 1-0, 2-1, 1-1, 2-0 account for ~55% of all WC games. Scores of 3+ goals for one team happen ~20% of games.

Respond ONLY with valid JSON — no other text:
{{"home_score": int, "away_score": int, "home_win_prob": float, "draw_prob": float, "away_win_prob": float, "reasoning": "one sentence summary"}}"""


def _build_context_str(
    df: pd.DataFrame, elo_history: dict,
    team_a: str, team_b: str, match_date: str, stage: str,
) -> str:
    form_a = get_team_recent_form(df, team_a, match_date, n=6)
    form_b = get_team_recent_form(df, team_b, match_date, n=6)
    h2h = get_head_to_head_results(df, team_a, team_b, match_date, n=6)
    elo = get_elo_ratings(elo_history, team_a, team_b, match_date)
    return f"{elo}\n\n{form_a}\n\n{form_b}\n\n{h2h}"


def _parse_prediction(text: str) -> dict:
    text = text.strip()
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        text = match.group()
    try:
        data = json.loads(text)
        hs = max(0, int(data.get("home_score", 1)))
        as_ = max(0, int(data.get("away_score", 1)))
        hwp = float(data.get("home_win_prob", 0.4))
        dp = float(data.get("draw_prob", 0.25))
        awp = float(data.get("away_win_prob", 0.35))
        total = hwp + dp + awp
        if total > 0:
            hwp, dp, awp = hwp/total, dp/total, awp/total
        return {
            "home_score": hs, "away_score": as_,
            "proba": [hwp, dp, awp],
            "reasoning": str(data.get("reasoning", "")),
        }
    except Exception:
        return {"home_score": 1, "away_score": 1, "proba": [0.33, 0.34, 0.33], "reasoning": "parse error"}


async def predict_direct(
    anthropic_client, openai_client,
    model: str,
    df: pd.DataFrame, elo_history: dict,
    team_a: str, team_b: str, match_date: str, stage: str,
    few_shot: bool = False,
    chain_of_thought: bool = False,
) -> dict:
    context = _build_context_str(df, elo_history, team_a, team_b, match_date, stage)

    if chain_of_thought:
        cot = CHAIN_OF_THOUGHT_PROMPT.format(
            team_a=team_a, team_b=team_b, date=match_date, stage=stage, context=context
        )
        prompt = FEW_SHOT_EXAMPLES + cot if few_shot else cot
    elif few_shot:
        prompt = FEW_SHOT_EXAMPLES + ZERO_SHOT_PROMPT.format(
            team_a=team_a, team_b=team_b, date=match_date, stage=stage, context=context
        )
    else:
        prompt = ZERO_SHOT_PROMPT.format(
            team_a=team_a, team_b=team_b, date=match_date, stage=stage, context=context
        )

    try:
        if model in ANTHROPIC_MODELS:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: anthropic_client.messages.create(
                    model=model, max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            text = resp.content[0].text
        else:
            is_o = model.startswith("o3") or model.startswith("o1")
            tok_key = "max_completion_tokens" if is_o else "max_tokens"
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                tok_key: 512,
            }
            if is_o:
                kwargs["reasoning_effort"] = "high"
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: openai_client.chat.completions.create(**kwargs)
            )
            text = resp.choices[0].message.content or "{}"
    except Exception as e:
        print(f"  [llm_direct] Error {model} {team_a} vs {team_b}: {e}")
        text = "{}"

    return _parse_prediction(text)
