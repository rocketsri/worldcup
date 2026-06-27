"""Autoresearch coordinator — reads leaderboard, proposes next experiments."""
import json
import asyncio
from experiments.config import ExperimentConfig
from experiments.leaderboard import Leaderboard

COORDINATOR_PROMPT = """You are an ML research coordinator optimizing a World Cup score prediction system.

Here is the current experiment leaderboard (higher winner_accuracy is better, lower RPS is better):

{leaderboard}

The experiment grid tests:
- feature_set: elo_only, stat (ELO+form+H2H), stat_llm (stat + LLM qualitative features), llm_direct
- model_type: poisson, dixon_coles, xgboost, llm_direct
- llm_model: claude-sonnet-4-6, claude-opus-4-8, gpt-4o, o3-mini
- few_shot: true/false (for llm_direct only)

Based on the results so far, propose 3-5 new experiments to run. Focus on:
1. Combinations that seem promising based on what worked
2. Ablations to understand which component drives performance
3. Novel combinations not yet tried

For each experiment, provide:
- id: E{N} where N is the next sequential number
- name: descriptive name
- feature_set: one of the options above
- model_type: one of the options above
- llm_model: model name or null
- reasoning_effort: "high" or null
- few_shot: true or false
- hypothesis: why you think this will work better

Respond with valid JSON array of experiment configs:
[{{"id": "E15", "name": "...", "feature_set": "...", "model_type": "...", "llm_model": "...", "reasoning_effort": null, "few_shot": false, "hypothesis": "..."}}]"""


def run_coordinator(
    leaderboard: Leaderboard,
    anthropic_client,
    start_id: int = 15,
) -> list[ExperimentConfig]:
    """Use Opus to propose next experiments based on leaderboard."""
    lb_str = leaderboard.to_summary_str()
    prompt = COORDINATOR_PROMPT.replace("{leaderboard}", lb_str)

    print("\n[Coordinator] Analyzing leaderboard and proposing next experiments...")
    try:
        resp = anthropic_client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        print(f"[Coordinator] Response:\n{text[:500]}...")

        # Parse JSON array
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            print("[Coordinator] Could not parse JSON response")
            return []
        proposals = json.loads(match.group())

    except Exception as e:
        print(f"[Coordinator] Error: {e}")
        return []

    configs = []
    for i, p in enumerate(proposals[:5]):
        try:
            eid = p.get("id", f"E{start_id + i}")
            config = ExperimentConfig(
                id=eid,
                name=p.get("name", f"Coordinator experiment {eid}"),
                feature_set=p.get("feature_set", "stat"),
                model_type=p.get("model_type", "poisson"),
                llm_model=p.get("llm_model"),
                reasoning_effort=p.get("reasoning_effort"),
                few_shot=bool(p.get("few_shot", False)),
                hypothesis=p.get("hypothesis", ""),
            )
            configs.append(config)
            print(f"  [Coordinator] Proposed: {config.id} - {config.name}: {config.hypothesis[:80]}")
        except Exception as e:
            print(f"  [Coordinator] Could not parse proposal {i}: {e}")

    # Save coordinator reasoning
    import json as _json
    from pathlib import Path
    (Path("results") / "coordinator_reasoning.json").write_text(
        _json.dumps({"leaderboard": lb_str, "proposals": proposals}, indent=2)
    )
    return configs
