"""Abstract base class for all prediction models."""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class PredictionModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series):
        ...

    @abstractmethod
    def predict_goals(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (lambda_home, lambda_away) expected goals arrays."""
        ...

    def predict_proba(self, X: pd.DataFrame, max_goals: int = 8) -> np.ndarray:
        """Return P(home_win), P(draw), P(away_win) for each row."""
        lam_h, lam_a = self.predict_goals(X)
        n = len(lam_h)
        proba = np.zeros((n, 3))
        g = np.arange(max_goals + 1)

        for i in range(n):
            from scipy.stats import poisson
            ph = poisson.pmf(g, lam_h[i])
            pa = poisson.pmf(g, lam_a[i])
            score_matrix = np.outer(ph, pa)
            proba[i, 0] = np.tril(score_matrix, -1).sum()  # home win
            proba[i, 1] = np.trace(score_matrix)            # draw
            proba[i, 2] = np.triu(score_matrix, 1).sum()   # away win
        return proba

    def predict_score(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Most likely score (mode of Poisson)."""
        lam_h, lam_a = self.predict_goals(X)
        return np.maximum(np.floor(lam_h), 0).astype(int), np.maximum(np.floor(lam_a), 0).astype(int)
