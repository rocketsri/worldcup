"""XGBoost wrapper for W/D/L classification and goals regression."""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from models.base import PredictionModel

FEATURE_COLS = [
    "elo_diff", "elo_a", "elo_b",
    "form_a", "form_b", "form_diff",
    "goals_scored_a", "goals_conceded_a",
    "goals_scored_b", "goals_conceded_b",
    "h2h_win_rate_a", "h2h_games",
    "is_neutral", "stage_weight",
]
LLM_COLS = ["injury_impact", "motivation_factor", "tactical_edge",
            "news_sentiment_a", "news_sentiment_b"]


class XGBoostWDL(PredictionModel):
    """Predict W/D/L probabilities + expected goals from those."""
    def __init__(self, use_llm_features: bool = False):
        self.use_llm = use_llm_features
        self.clf = XGBClassifier(
            objective="multi:softprob", num_class=3, n_estimators=200,
            max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="mlogloss",
            random_state=42, verbosity=0,
        )
        self.reg_home = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
        )
        self.reg_away = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
        )
        self._cols = None

    def _get_cols(self, X: pd.DataFrame) -> list[str]:
        cols = [c for c in FEATURE_COLS if c in X.columns]
        if self.use_llm:
            cols += [c for c in LLM_COLS if c in X.columns]
        return cols

    def fit(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series):
        self._cols = self._get_cols(X)
        Xf = X[self._cols].fillna(0)
        y_outcome = np.where(
            y_home > y_away, 0,
            np.where(y_home < y_away, 2, 1)
        )
        self.clf.fit(Xf, y_outcome)
        self.reg_home.fit(Xf, y_home)
        self.reg_away.fit(Xf, y_away)

    def predict_goals(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Xf = X[self._cols].fillna(0)
        lam_h = np.clip(self.reg_home.predict(Xf), 0.1, 15)
        lam_a = np.clip(self.reg_away.predict(Xf), 0.1, 15)
        return lam_h, lam_a

    def predict_proba(self, X: pd.DataFrame, max_goals: int = 8) -> np.ndarray:
        Xf = X[self._cols].fillna(0)
        return self.clf.predict_proba(Xf)  # shape (n, 3): home_win, draw, away_win
