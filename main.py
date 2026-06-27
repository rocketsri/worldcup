"""
World Cup Score Prediction Harness — Master Orchestrator
Phases:
  1. Load data + compute ELO
  2. Run stat-only experiments (E01-E04), quick eval on 2018 WC
  3. Run LLM experiments (E05-E14), quick eval on 2018 WC
  4. Coordinator grid (E15-E22): deterministic replay of Opus coordinator proposals
  4b. International ELO experiments (E23-E24): run if data/raw/international_results.csv present
  5. Full eval (top 8 configs) on 2026 WC (true out-of-sample holdout)
  6. Predict 2026 WC Round of 32
"""
import asyncio
import sys
import time
import pandas as pd

from config import make_clients, QUICK_EVAL_YEAR, FULL_EVAL_YEAR
from data.fetch import load_all_wc_matches, get_wc_matches
from data.elo import compute_elo_history, get_elo_at
from experiments.config import INITIAL_GRID, COORDINATOR_GRID, ExperimentConfig
from experiments.runner import run_all_experiments, run_stat_experiment, _make_model
from experiments.leaderboard import Leaderboard, ExperimentResult
from evaluation.backtest import fast_eval
from data.features import build_features as bf


def print_phase(n, title: str):
    print(f"\n{'='*60}")
    print(f"PHASE {n}: {title}")
    print(f"{'='*60}")


def _build_intl_win_prob_map(df_all: pd.DataFrame, elo_history_intl: dict) -> dict:
    """Pre-compute intl_win_prob for every match in df_all (used by E24)."""
    result = {}
    for _, row in df_all.iterrows():
        match_id = f"{row['date'].date()}_{row['home_team']}_{row['away_team']}"
        cutoff = row["date"] - pd.Timedelta(days=1)
        ie_a = get_elo_at(elo_history_intl, row["home_team"], cutoff)
        ie_b = get_elo_at(elo_history_intl, row["away_team"], cutoff)
        dr = (ie_a - ie_b) / 400.0
        result[match_id] = {"intl_win_prob": 1.0 / (1.0 + 10.0 ** (-dr))}
    return result


async def _run_intl_experiment(
    config: ExperimentConfig,
    df_all: pd.DataFrame,
    elo_history_wc: dict,
    elo_history_intl: dict,
    quick_val: pd.DataFrame,
    full_val: pd.DataFrame | None,
    leaderboard: Leaderboard,
):
    """Run E23 (intl ELO as main source) or E24 (dual ELO via injected features)."""
    t0 = time.time()
    result = ExperimentResult(
        exp_id=config.id, name=config.name,
        feature_set=config.feature_set, model_type=config.model_type,
        llm_model=None, hypothesis=config.hypothesis,
    )
    try:
        model = _make_model(config)
        if config.use_intl_elo:
            # E23: international ELO replaces WC ELO as the main rating source
            quick = fast_eval(model, df_all, elo_history_intl, quick_val, bf)
        else:
            # E24: WC ELO + intl_win_prob injected as a per-match feature
            intl_map = _build_intl_win_prob_map(df_all, elo_history_intl)
            quick = fast_eval(model, df_all, elo_history_wc, quick_val, bf,
                              llm_features_map=intl_map)

        result.quick_winner_acc = quick.winner_acc
        result.n_games = quick.n_games
        leaderboard.save(result)
        print(f"  [{config.id}] quick_acc={quick.winner_acc:.3f} ({quick.n_games} games)")

        if full_val is not None:
            model2 = _make_model(config)
            if config.use_intl_elo:
                full = fast_eval(model2, df_all, elo_history_intl, full_val, bf)
            else:
                intl_map = _build_intl_win_prob_map(df_all, elo_history_intl)
                full = fast_eval(model2, df_all, elo_history_wc, full_val, bf,
                                 llm_features_map=intl_map)
            result.full_winner_acc = full.winner_acc
            result.full_exact_acc = full.exact_acc
            result.full_mae = full.mae_goals
            result.full_rps = full.rps
            print(f"  [{config.id}] full_acc={full.winner_acc:.3f} rps={full.rps:.3f}")

    except Exception as e:
        result.error = str(e)
        print(f"  [{config.id}] ERROR: {e}")

    result.elapsed_s = time.time() - t0
    leaderboard.save(result)
    return result


async def main():
    t_start = time.time()

    # ── Phase 1: Data & ELO ──────────────────────────────────────
    print_phase(1, "Loading data & computing ELO ratings")
    df_all = load_all_wc_matches()
    print(f"  Loaded {len(df_all)} WC matches (2006-2026, completed games)")

    elo_history, final_elos = compute_elo_history(df_all)
    print(f"  ELO computed for {len(elo_history)} teams")

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

    # ── Phase 4: Coordinator grid ─────────────────────────────────
    # Use the pre-recorded COORDINATOR_GRID (deterministic, avoids paying Opus on every run).
    # To re-run the coordinator dynamically, clear COORDINATOR_GRID in experiments/config.py
    # and the block below will fall back to calling run_coordinator().
    print_phase(4, "Running coordinator-proposed experiments (E15-E22)")
    non_intl_coordinator = [c for c in COORDINATOR_GRID if not c.use_intl_elo
                            and not (c.feature_cols and "intl_win_prob" in c.feature_cols)]
    if non_intl_coordinator:
        await run_all_experiments(
            non_intl_coordinator, df_all, elo_history,
            quick_val, None, leaderboard,
            anthropic_client, openai_client,
            run_full=False,
        )
    else:
        from agents.coordinator import run_coordinator
        print("  COORDINATOR_GRID empty — running live Opus coordinator")
        dynamic_configs = run_coordinator(leaderboard, anthropic_client, start_id=15)
        if dynamic_configs:
            await run_all_experiments(
                dynamic_configs, df_all, elo_history,
                quick_val, None, leaderboard,
                anthropic_client, openai_client,
                run_full=False,
            )
    leaderboard.display()

    # ── Phase 4b: International ELO experiments (E23/E24) ────────
    intl_configs = [c for c in COORDINATOR_GRID
                    if c.use_intl_elo or (c.feature_cols and "intl_win_prob" in c.feature_cols)]
    if intl_configs:
        print_phase("4b", "International ELO experiments (E23/E24)")
        from pathlib import Path
        intl_path = Path("data/raw/international_results.csv")
        if intl_path.exists():
            from data.elo import compute_elo_history_international
            df_intl = pd.read_csv(intl_path, parse_dates=["date"])
            df_intl_post90 = df_intl[df_intl["date"].dt.year >= 1990].copy()
            elo_history_intl, _ = compute_elo_history_international(df_intl_post90)
            print(f"  International ELO computed from {len(df_intl_post90)} games (post-1990)")

            for config in intl_configs:
                await _run_intl_experiment(
                    config, df_all, elo_history, elo_history_intl,
                    quick_val, None, leaderboard,
                )
        else:
            print("  Skipping E23/E24: data/raw/international_results.csv not found")
            print("  Download from https://github.com/martj42/international_results")
        leaderboard.display()

    # ── Phase 5: Full eval on top configs ────────────────────────
    print_phase(5, f"Full evaluation on {FULL_EVAL_YEAR} WC (top 8 configs)")
    all_results = list(leaderboard.results.values())
    top8 = sorted(
        [r for r in all_results if r.quick_winner_acc is not None and not r.error],
        key=lambda r: r.quick_winner_acc,
        reverse=True,
    )[:8]

    print(f"  Running full eval for: {[r.exp_id for r in top8]}")
    all_configs = INITIAL_GRID + COORDINATOR_GRID
    top_configs = [c for c in all_configs if c.id in {r.exp_id for r in top8}]

    # Separate intl-ELO experiments for special handling
    top_intl = [c for c in top_configs
                if c.use_intl_elo or (c.feature_cols and "intl_win_prob" in c.feature_cols)]
    top_normal = [c for c in top_configs if c not in top_intl]

    if top_normal:
        await run_all_experiments(
            top_normal, df_all, elo_history,
            quick_val, full_val, leaderboard,
            anthropic_client, openai_client,
            run_full=True,
        )

    if top_intl:
        from pathlib import Path
        if Path("data/raw/international_results.csv").exists():
            from data.elo import compute_elo_history_international
            if "elo_history_intl" not in dir():
                df_intl = pd.read_csv("data/raw/international_results.csv", parse_dates=["date"])
                df_intl_post90 = df_intl[df_intl["date"].dt.year >= 1990].copy()
                elo_history_intl, _ = compute_elo_history_international(df_intl_post90)
            for config in top_intl:
                await _run_intl_experiment(
                    config, df_all, elo_history, elo_history_intl,
                    quick_val, full_val, leaderboard,
                )

    leaderboard.display()

    # ── Phase 6: 2026 Predictions ─────────────────────────────────
    print_phase(6, "Predicting 2026 WC Round of 32")

    # Find best stat config (exclude intl-ELO experiments — they can't predict 2026 games
    # without international results up to 2026, and WC-only ELO is better calibrated anyway)
    stat_result_ids = {
        r.exp_id for r in all_results
        if not r.llm_model and not r.error
    }
    best_stat_config = next(
        (c for c in sorted(
            [c for c in all_configs
             if c.id in stat_result_ids and not c.use_intl_elo],
            key=lambda c: (
                leaderboard.results.get(c.id, ExperimentResult("", "", "", "", None)).full_winner_acc
                or leaderboard.results.get(c.id, ExperimentResult("", "", "", "", None)).quick_winner_acc
                or 0
            ),
            reverse=True,
        )),
        all_configs[1],  # fallback to E02
    )

    print(f"  Using model: {best_stat_config.id} — {best_stat_config.name}")
    best_model = _make_model(best_stat_config)
    from data.features import build_dataset
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
