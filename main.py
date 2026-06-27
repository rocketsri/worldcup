"""
World Cup Score Prediction Harness — Master Orchestrator
Phases:
  1. Load data + compute ELO
  2. Run stat-only experiments (E01-E04), quick eval on 2018 WC
  3. Run LLM experiments (E05-E14), quick eval on 2018 WC
  4. Coordinator: propose E15+ based on leaderboard
  5. Full eval (top configs) on 2022 WC
  6. Predict 2026 WC Round of 16
"""
import asyncio
import sys
import time
import pandas as pd

from config import make_clients, QUICK_EVAL_YEAR, FULL_EVAL_YEAR
from data.fetch import load_all_wc_matches, get_wc_matches
from data.elo import compute_elo_history
from experiments.config import INITIAL_GRID, ExperimentConfig
from experiments.runner import run_all_experiments, run_stat_experiment, run_llm_experiment
from experiments.leaderboard import Leaderboard, ExperimentResult
from agents.coordinator import run_coordinator


def print_phase(n: int, title: str):
    print(f"\n{'='*60}")
    print(f"PHASE {n}: {title}")
    print(f"{'='*60}")


async def main():
    t_start = time.time()

    # ── Phase 1: Data & ELO ──────────────────────────────────────
    print_phase(1, "Loading data & computing ELO ratings")
    df_all = load_all_wc_matches()
    print(f"  Loaded {len(df_all)} WC matches (2006-2022)")

    elo_history, final_elos = compute_elo_history(df_all)
    print(f"  ELO computed for {len(elo_history)} teams")

    # Show top 10 ELO ratings after all history (for verification)
    top10 = sorted(final_elos.items(), key=lambda x: x[1], reverse=True)[:10]
    print("  Top 10 ELO ratings (current):")
    for team, elo in top10:
        print(f"    {team:<25} {elo:.0f}")

    quick_val = get_wc_matches(df_all, QUICK_EVAL_YEAR)
    full_val = get_wc_matches(df_all, FULL_EVAL_YEAR)
    print(f"\n  Quick eval set: {len(quick_val)} games (WC {QUICK_EVAL_YEAR})")
    print(f"  Full eval set:  {len(full_val)} games (WC {FULL_EVAL_YEAR})")

    # ── Phase 2: Stat-only experiments ───────────────────────────
    print_phase(2, "Running stat-only experiments (E01-E04)")
    anthropic_client, openai_client = make_clients()
    leaderboard = Leaderboard()

    stat_configs = [c for c in INITIAL_GRID if not c.needs_llm]
    await run_all_experiments(
        stat_configs, df_all, elo_history,
        quick_val, None, leaderboard,
        anthropic_client, openai_client,
        run_full=False,
    )
    leaderboard.display()

    # ── Phase 3: LLM experiments ─────────────────────────────────
    print_phase(3, "Running LLM experiments (E05-E14) — parallel API calls")
    llm_configs = [c for c in INITIAL_GRID if c.needs_llm]
    await run_all_experiments(
        llm_configs, df_all, elo_history,
        quick_val, None, leaderboard,
        anthropic_client, openai_client,
        run_full=False,
    )
    leaderboard.display()

    # ── Phase 4: Coordinator ─────────────────────────────────────
    print_phase(4, "Coordinator: proposing next experiments")
    coordinator_configs = run_coordinator(leaderboard, anthropic_client, start_id=15)
    if coordinator_configs:
        await run_all_experiments(
            coordinator_configs, df_all, elo_history,
            quick_val, None, leaderboard,
            anthropic_client, openai_client,
            run_full=False,
        )
        leaderboard.display()

    # ── Phase 5: Full eval on top configs ────────────────────────
    print_phase(5, "Full evaluation on 2022 WC (top 8 configs)")
    all_results = list(leaderboard.results.values())
    top8 = sorted(
        [r for r in all_results if r.quick_winner_acc is not None and not r.error],
        key=lambda r: r.quick_winner_acc,
        reverse=True,
    )[:8]

    print(f"  Running full eval for: {[r.exp_id for r in top8]}")
    all_configs = INITIAL_GRID + (coordinator_configs or [])
    top_configs = [c for c in all_configs if c.id in {r.exp_id for r in top8}]

    await run_all_experiments(
        top_configs, df_all, elo_history,
        quick_val, full_val, leaderboard,
        anthropic_client, openai_client,
        run_full=True,
    )
    leaderboard.display()

    # ── Phase 6: 2026 Predictions ─────────────────────────────────
    print_phase(6, "Predicting 2026 WC Round of 16")

    # Use best stat model for predictions (+ news search for 2026)
    # unused, replaced below
    _unused = None
    # Rebuild best stat model trained on all available data (2006-2021)
    from data.features import build_dataset
    from evaluation.backtest import fast_eval
    from experiments.runner import _make_model

    # Find best stat config
    stat_result_ids = {r.exp_id for r in all_results if not r.llm_model and not r.error}
    best_stat_config = next(
        (c for c in sorted(
            [c for c in all_configs if c.id in stat_result_ids],
            key=lambda c: leaderboard.results.get(c.id, ExperimentResult("", "", "", "", None)).full_winner_acc or 0,
            reverse=True,
        )), stat_configs[1]  # fallback to E02
    )

    print(f"  Using model: {best_stat_config.id} — {best_stat_config.name}")
    best_model = _make_model(best_stat_config)
    train_data = df_all[df_all["year"] < 2026].copy()
    X_train, yh, ya, _ = build_dataset(df_all, elo_history, train_data)
    if best_stat_config.model_type == "dixon_coles":
        best_model.fit(X_train, yh, ya, df_raw=train_data)
    else:
        best_model.fit(X_train, yh, ya)

    from predict.predict_2026 import predict_2026
    predictions = await predict_2026(best_model, df_all, elo_history)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE — total elapsed: {elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    leaderboard.display()


if __name__ == "__main__":
    asyncio.run(main())
