"""Independent Poisson regression fit via scipy MLE."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from models.base import PredictionModel

FEATURE_COLS = [
    "elo_diff", "form_diff", "goals_scored_a", "goals_conceded_a",
    "goals_scored_b", "goals_conceded_b", "h2h_win_rate_a", "stage_weight",
]


class PoissonModel(PredictionModel):
    def __init__(self, feature_cols: list[str] | None = None):
        self.feature_cols = feature_cols or FEATURE_COLS
        self.beta_home = None
        self.beta_away = None

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.feature_cols if c in X.columns]
        mat = X[cols].fillna(0).values
        return np.hstack([np.ones((len(mat), 1)), mat])

    def fit(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series):
        D = self._design(X)
        k = D.shape[1]
        y_h = y_home.values.astype(float)
        y_a = y_away.values.astype(float)

        def neg_ll(params):
            bh, ba = params[:k], params[k:]
            lam_h = np.exp(D @ bh)
            lam_a = np.exp(D @ ba)
            lam_h = np.clip(lam_h, 1e-6, 20)
            lam_a = np.clip(lam_a, 1e-6, 20)
            ll = (poisson.logpmf(y_h, lam_h) + poisson.logpmf(y_a, lam_a)).sum()
            return -ll

        x0 = np.zeros(2 * k)
        x0[0] = np.log(max(y_h.mean(), 0.1))
        x0[k] = np.log(max(y_a.mean(), 0.1))
        res = minimize(neg_ll, x0, method="L-BFGS-B", options={"maxiter": 500})
        self.beta_home = res.x[:k]
        self.beta_away = res.x[k:]
        self._fitted_cols = [c for c in self.feature_cols if c in X.columns]

    def predict_goals(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        D = self._design(X)
        lam_h = np.clip(np.exp(D @ self.beta_home), 0.1, 15)
        lam_a = np.clip(np.exp(D @ self.beta_away), 0.1, 15)
        return lam_h, lam_a
