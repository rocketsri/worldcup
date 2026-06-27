"""Experiment configuration and initial grid."""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ExperimentConfig:
    id: str
    name: str
    feature_set: Literal["elo_only", "stat", "stat_llm", "llm_direct"]
    model_type: Literal["poisson", "dixon_coles", "xgboost", "llm_direct"]
    llm_model: str | None = None
    reasoning_effort: str | None = None
    few_shot: bool = False
    chain_of_thought: bool = False
    hypothesis: str = ""

    @property
    def needs_llm(self) -> bool:
        return self.feature_set in ("stat_llm", "llm_direct")


INITIAL_GRID: list[ExperimentConfig] = [
    # --- Stat-only (no LLM) ---
    ExperimentConfig("E01", "ELO-only Poisson",    "elo_only",  "poisson"),
    ExperimentConfig("E02", "Stat Poisson",         "stat",      "poisson"),
    ExperimentConfig("E03", "Stat Dixon-Coles",     "stat",      "dixon_coles"),
    ExperimentConfig("E04", "Stat XGBoost",         "stat",      "xgboost"),
    # --- LLM-augmented (stat features + LLM qualitative features) ---
    ExperimentConfig("E05", "Stat+LLM Poisson (Sonnet)",   "stat_llm", "poisson",
                     llm_model="claude-sonnet-4-6"),
    ExperimentConfig("E06", "Stat+LLM Poisson (Opus)",     "stat_llm", "poisson",
                     llm_model="claude-opus-4-8"),
    ExperimentConfig("E07", "Stat+LLM Poisson (GPT-4o)",   "stat_llm", "poisson",
                     llm_model="gpt-4o"),
    ExperimentConfig("E08", "Stat+LLM Poisson (o3-mini)",  "stat_llm", "poisson",
                     llm_model="o3-mini", reasoning_effort="high"),
    ExperimentConfig("E09", "Stat+LLM XGBoost (Sonnet)",   "stat_llm", "xgboost",
                     llm_model="claude-sonnet-4-6"),
    # --- LLM direct prediction ---
    ExperimentConfig("E10", "LLM Direct (Sonnet 0-shot)",  "llm_direct", "llm_direct",
                     llm_model="claude-sonnet-4-6"),
    ExperimentConfig("E11", "LLM Direct (Opus 0-shot)",    "llm_direct", "llm_direct",
                     llm_model="claude-opus-4-8"),
    ExperimentConfig("E12", "LLM Direct (GPT-4o 0-shot)",  "llm_direct", "llm_direct",
                     llm_model="gpt-4o"),
    ExperimentConfig("E13", "LLM Direct (o3-mini high)",   "llm_direct", "llm_direct",
                     llm_model="o3-mini", reasoning_effort="high"),
    ExperimentConfig("E14", "LLM Direct (Sonnet 5-shot)",  "llm_direct", "llm_direct",
                     llm_model="claude-sonnet-4-6", few_shot=True),
]

# Coordinator-proposed experiments (from Opus analysis of initial leaderboard)
COORDINATOR_GRID: list[ExperimentConfig] = [
    ExperimentConfig("E15", "LLM Direct (GPT-4o 5-shot)", "llm_direct", "llm_direct",
                     llm_model="gpt-4o", few_shot=True,
                     hypothesis="GPT-4o is top performer. Test if few-shot examples improve its calibration "
                                "like they did for other models (few-shot effect isolation on best base model)"),
    ExperimentConfig("E18", "Stat+LLM Poisson (o3-mini fixed)", "stat_llm", "poisson",
                     llm_model="o3-mini", reasoning_effort="high",
                     hypothesis="E08 failed due to max_tokens API error, not a modeling issue. "
                                "Re-run with max_completion_tokens to complete the LLM sweep."),
    ExperimentConfig("E19", "LLM Direct (Opus 5-shot)", "llm_direct", "llm_direct",
                     llm_model="claude-opus-4-8", few_shot=True,
                     hypothesis="Opus has best robust accuracy (62.5% on 2022). Test if few-shot examples "
                                "allow Opus to close gap with GPT-4o on 2018 while maintaining 2022 robustness."),
    ExperimentConfig("E20", "LLM Direct (Opus CoT + 5-shot)", "llm_direct", "llm_direct",
                     llm_model="claude-opus-4-8", few_shot=True, chain_of_thought=True,
                     hypothesis="Chain-of-thought forces systematic ELO→form→H2H→stage reasoning before "
                                "committing to a score. Improved few-shot includes upset example and draw example "
                                "to prevent overconfidence on close games. Targets 70%+ on 2022 WC."),
    ExperimentConfig("E21", "LLM Direct (Sonnet CoT + 5-shot)", "llm_direct", "llm_direct",
                     llm_model="claude-sonnet-4-6", few_shot=True, chain_of_thought=True,
                     hypothesis="Chain-of-thought applied to Sonnet. Sonnet 0-shot hit 60.9% on 2022. "
                                "Structured reasoning may reduce errors on borderline games."),
    ExperimentConfig("E22", "Stat Poisson (5-feature minimal)", "stat", "poisson",
                     hypothesis="Drop-one-out ablation showed h2h_games, goals_conceded_a, goals_scored_b "
                                "all HURT the model. 5-feature minimal (elo_diff, confederation_diff, "
                                "win_streak_a/b, stage_weight) achieves 54.7% on 2022 WC vs 48.4% for "
                                "20-feature baseline — overfitting on sparse WC data."),
]

MINIMAL_STAT_COLS = ['elo_diff', 'confederation_diff', 'win_streak_a', 'win_streak_b', 'stage_weight']
