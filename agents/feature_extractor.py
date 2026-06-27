"""Convert LLM context summary → structured numeric features."""
import json
import asyncio
import re

ANTHROPIC_MODELS = {"claude-sonnet-4-6", "claude-opus-4-8"}

EXTRACTION_PROMPT = """Given the following match analysis context, extract exactly these 6 numeric features.

Features 1-5: float in [-1.0, 1.0] where positive = favors team_a, negative = favors team_b, 0 = neutral.
Feature 6: float in [0.0, 1.0] where higher = more intense rivalry/political tension.

1. injury_impact: net impact of injuries/suspensions (positive = team_b more affected)
2. motivation_factor: psychological/motivational edge (must-win, momentum, pressure)
3. tactical_edge: tactical or stylistic matchup advantage
4. news_sentiment_a: media/fan sentiment around team_a (confidence, morale, cohesion)
5. news_sentiment_b: media/fan sentiment around team_b

6. rivalry_intensity: [0,1] — geopolitical, historical, or political tension between the nations
   (e.g. 0.9 for USA vs Iran in 2022, 0.85 for Argentina vs England, 0.0 for Switzerland vs Ghana)
   This captures variance-increasing effects: high-rivalry matches tend to be less predictable,
   with underdogs performing above ELO expectations due to emotional intensity.

Context:
{context}

Respond ONLY with valid JSON, no explanation:
{{"injury_impact": float, "motivation_factor": float, "tactical_edge": float, "news_sentiment_a": float, "news_sentiment_b": float, "rivalry_intensity": float}}"""


def _parse_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        text = match.group()
    try:
        data = json.loads(text)
        bipolar = ["injury_impact", "motivation_factor", "tactical_edge", "news_sentiment_a", "news_sentiment_b"]
        result = {k: float(max(-1.0, min(1.0, data.get(k, 0.0)))) for k in bipolar}
        result["rivalry_intensity"] = float(max(0.0, min(1.0, data.get("rivalry_intensity", 0.0))))
        return result
    except Exception:
        return {k: 0.0 for k in ["injury_impact", "motivation_factor", "tactical_edge",
                                  "news_sentiment_a", "news_sentiment_b", "rivalry_intensity"]}


async def extract_features(
    anthropic_client, openai_client,
    model: str, context: str, team_a: str, team_b: str,
) -> dict:
    prompt = EXTRACTION_PROMPT.format(context=context[:3000])

    try:
        if model in ANTHROPIC_MODELS:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: anthropic_client.messages.create(
                    model=model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            text = resp.content[0].text
        else:
            is_o = model.startswith("o3") or model.startswith("o1")
            tok_key = "max_completion_tokens" if is_o else "max_tokens"
            kwargs = {"model": model, tok_key: 256,
                      "messages": [{"role": "user", "content": prompt}]}
            if is_o:
                kwargs["reasoning_effort"] = "medium"
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: openai_client.chat.completions.create(**kwargs)
            )
            text = resp.choices[0].message.content or "{}"
    except Exception as e:
        print(f"  [feature_extractor] Error for {team_a} vs {team_b}: {e}")
        text = "{}"

    return _parse_json(text)
