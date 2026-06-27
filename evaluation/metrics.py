"""Evaluation metrics for match prediction."""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    winner_acc: float = 0.0
    exact_acc: float = 0.0
    mae_goals: float = 0.0
    rps: float = 0.0
    n_games: int = 0
    details: list = field(default_factory=list)

    def __repr__(self):
        return (f"WinnerAcc={self.winner_acc:.3f} ExactAcc={self.exact_acc:.3f} "
                f"MAE={self.mae_goals:.3f} RPS={self.rps:.3f} n={self.n_games}")


def outcome_label(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return 0  # home win
    if home_score < away_score:
        return 2  # away win
    return 1  # draw


def winner_accuracy(pred_outcomes: np.ndarray, true_outcomes: np.ndarray) -> float:
    return float((pred_outcomes == true_outcomes).mean())


def exact_score_accuracy(pred_h, pred_a, true_h, true_a) -> float:
    return float(((pred_h == true_h) & (pred_a == true_a)).mean())


def mae_goals(pred_h, pred_a, true_h, true_a) -> float:
    return float((np.abs(pred_h - true_h) + np.abs(pred_a - true_a)).mean() / 2)


def ranked_probability_score(proba: np.ndarray, true_outcomes: np.ndarray) -> float:
    """
    RPS for 3-outcome predictions (home win=0, draw=1, away win=2).
    Lower is better. Proper scoring rule.
    """
    K = 3
    n = len(true_outcomes)
    rps_total = 0.0
    for i in range(n):
        true_one_hot = np.zeros(K)
        true_one_hot[int(true_outcomes[i])] = 1
        cum_pred = np.cumsum(proba[i])
        cum_true = np.cumsum(true_one_hot)
        rps_total += np.sum((cum_pred[:-1] - cum_true[:-1]) ** 2) / (K - 1)
    return rps_total / n


def evaluate_predictions(
    pred_proba: np.ndarray,
    pred_h: np.ndarray,
    pred_a: np.ndarray,
    true_h: np.ndarray,
    true_a: np.ndarray,
) -> EvalResult:
    true_outcomes = np.array([outcome_label(h, a) for h, a in zip(true_h, true_a)])
    pred_outcomes = np.argmax(pred_proba, axis=1)

    return EvalResult(
        winner_acc=winner_accuracy(pred_outcomes, true_outcomes),
        exact_acc=exact_score_accuracy(pred_h, pred_a, true_h, true_a),
        mae_goals=mae_goals(pred_h.astype(float), pred_a.astype(float),
                            true_h.astype(float), true_a.astype(float)),
        rps=ranked_probability_score(pred_proba, true_outcomes),
        n_games=len(true_h),
    )
