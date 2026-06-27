"""Async parallel experiment runner."""
import asyncio
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from experiments.config import ExperimentConfig
from experiments.leaderboard import Leaderboard, ExperimentResult
from evaluation.metrics import evaluate_predictions, EvalResult
from data.features import build_features, build_dataset, STAT_FEATURE_COLS

LLM_SEMAPHORE = asyncio.Semaphore(6)  # max concurrent LLM API calls


async def run_stat_experiment(
    config: ExperimentConfig,
    df_all: pd.DataFrame,
    elo_history: dict,
    quick_val: pd.DataFrame,
    full_val: pd.DataFrame | None,
    leaderboard: Leaderboard,
):
    """Run a stat-only experiment (Poisson / Dixon-Coles / XGBoost)."""
    t0 = time.time()
    result = ExperimentResult(
        exp_id=config.id, name=config.name,
        feature_set=config.feature_set, model_type=config.model_type,
        llm_model=None, hypothesis=config.hypothesis,
    )

    try:
        from evaluation.backtest import fast_eval
        from data.features import build_features as bf

        # Instantiate model
        model = _make_model(config)
        dc_mode = config.model_type == "dixon_coles"

        # Quick eval on 2018 WC
        quick = fast_eval(model, df_all, elo_history, quick_val, bf, dc_mode=dc_mode)
        result.quick_winner_acc = quick.winner_acc
        result.n_games = quick.n_games
        leaderboard.save(result)
        print(f"  [{config.id}] quick_acc={quick.winner_acc:.3f} ({quick.n_games} games)")

        # Full eval on 2022 WC (if requested)
        if full_val is not None:
            model2 = _make_model(config)
            full = fast_eval(model2, df_all, elo_history, full_val, bf, dc_mode=dc_mode)
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


async def run_llm_experiment(
    config: ExperimentConfig,
    df_all: pd.DataFrame,
    elo_history: dict,
    quick_val: pd.DataFrame,
    full_val: pd.DataFrame | None,
    leaderboard: Leaderboard,
    anthropic_client, openai_client,
):
    """Run an LLM-augmented or LLM-direct experiment."""
    t0 = time.time()
    result = ExperimentResult(
        exp_id=config.id, name=config.name,
        feature_set=config.feature_set, model_type=config.model_type,
        llm_model=config.llm_model, hypothesis=config.hypothesis,
    )

    try:
        if config.model_type == "llm_direct":
            quick = await _eval_llm_direct(
                config, df_all, elo_history, quick_val,
                anthropic_client, openai_client,
            )
            result.quick_winner_acc = quick.winner_acc
            result.n_games = quick.n_games
            leaderboard.save(result)
            print(f"  [{config.id}] quick_acc={quick.winner_acc:.3f}")

            if full_val is not None:
                full = await _eval_llm_direct(
                    config, df_all, elo_history, full_val,
                    anthropic_client, openai_client,
                )
                result.full_winner_acc = full.winner_acc
                result.full_exact_acc = full.exact_acc
                result.full_mae = full.mae_goals
                result.full_rps = full.rps
                print(f"  [{config.id}] full_acc={full.winner_acc:.3f} rps={full.rps:.3f}")
        else:
            # stat_llm: gather LLM features then run stat model
            llm_map = await _gather_llm_features(
                config, df_all, elo_history, quick_val,
                anthropic_client, openai_client,
            )
            from evaluation.backtest import fast_eval
            from data.features import build_features as bf
            model = _make_model(config)
            quick = fast_eval(model, df_all, elo_history, quick_val, bf, llm_map)
            result.quick_winner_acc = quick.winner_acc
            result.n_games = quick.n_games
            leaderboard.save(result)
            print(f"  [{config.id}] quick_acc={quick.winner_acc:.3f}")

            if full_val is not None:
                llm_map_full = await _gather_llm_features(
                    config, df_all, elo_history, full_val,
                    anthropic_client, openai_client,
                )
                model2 = _make_model(config)
                full = fast_eval(model2, df_all, elo_history, full_val, bf, llm_map_full)
                result.full_winner_acc = full.winner_acc
                result.full_exact_acc = full.exact_acc
                result.full_mae = full.mae_goals
                result.full_rps = full.rps
                print(f"  [{config.id}] full_acc={full.winner_acc:.3f} rps={full.rps:.3f}")

    except Exception as e:
        import traceback
        result.error = str(e)
        print(f"  [{config.id}] ERROR: {e}")
        traceback.print_exc()

    result.elapsed_s = time.time() - t0
    leaderboard.save(result)
    return result


async def _eval_llm_direct(
    config, df_all, elo_history, val_matches,
    anthropic_client, openai_client,
) -> EvalResult:
    from agents.llm_direct import predict_direct

    async def predict_one(row):
        async with LLM_SEMAPHORE:
            match_date = str(row["date"].date())
            return await predict_direct(
                anthropic_client, openai_client,
                config.llm_model, df_all, elo_history,
                row["home_team"], row["away_team"],
                match_date, row.get("stage", "group"),
                few_shot=config.few_shot,
                chain_of_thought=config.chain_of_thought,
            )

    tasks = [predict_one(row) for _, row in val_matches.iterrows()]
    predictions = await asyncio.gather(*tasks)

    pred_proba = np.array([p["proba"] for p in predictions])
    pred_h = np.array([p["home_score"] for p in predictions])
    pred_a = np.array([p["away_score"] for p in predictions])
    true_h = val_matches["home_score"].values
    true_a = val_matches["away_score"].values

    return evaluate_predictions(pred_proba, pred_h, pred_a, true_h, true_a)


async def _gather_llm_features(
    config, df_all, elo_history, val_matches,
    anthropic_client, openai_client,
) -> dict:
    from agents.context_agent import gather_context
    from agents.feature_extractor import extract_features

    async def process_one(row):
        async with LLM_SEMAPHORE:
            match_date = str(row["date"].date())
            ctx = await gather_context(
                anthropic_client, openai_client, config.llm_model,
                df_all, elo_history,
                row["home_team"], row["away_team"], match_date, row.get("stage", "group"),
            )
            feats = await extract_features(
                anthropic_client, openai_client, config.llm_model,
                ctx, row["home_team"], row["away_team"],
            )
            match_id = f"{row['date'].date()}_{row['home_team']}_{row['away_team']}"
            return match_id, feats

    tasks = [process_one(row) for _, row in val_matches.iterrows()]
    results = await asyncio.gather(*tasks)
    return dict(results)


def _make_model(config: ExperimentConfig):
    use_llm = config.feature_set == "stat_llm"
    if config.model_type == "poisson":
        from models.poisson_model import PoissonModel
        if config.feature_cols:
            cols = config.feature_cols
        elif config.feature_set == "elo_only":
            cols = ["elo_diff", "stage_weight"]
        else:
            cols = STAT_FEATURE_COLS.copy()
        return PoissonModel(feature_cols=cols)
    elif config.model_type == "dixon_coles":
        from models.dixon_coles import DixonColesModel
        return DixonColesModel()
    elif config.model_type == "xgboost":
        from models.xgboost_model import XGBoostWDL
        return XGBoostWDL(use_llm_features=use_llm)
    raise ValueError(f"Unknown model type: {config.model_type}")


async def run_all_experiments(
    configs: list[ExperimentConfig],
    df_all: pd.DataFrame,
    elo_history: dict,
    quick_val: pd.DataFrame,
    full_val: pd.DataFrame | None,
    leaderboard: Leaderboard,
    anthropic_client, openai_client,
    run_full: bool = False,
):
    """Launch all experiments concurrently."""
    fv = full_val if run_full else None
    tasks = []
    for config in configs:
        if not config.needs_llm:
            tasks.append(run_stat_experiment(
                config, df_all, elo_history, quick_val, fv, leaderboard
            ))
        else:
            tasks.append(run_llm_experiment(
                config, df_all, elo_history, quick_val, fv, leaderboard,
                anthropic_client, openai_client,
            ))

    await asyncio.gather(*tasks)
