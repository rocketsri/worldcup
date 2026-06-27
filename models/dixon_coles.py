"""Dixon-Coles bivariate Poisson model with time-decay weighting."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from models.base import PredictionModel

DECAY = 0.003  # days^-1


def _tau(x, y, lam, mu, rho):
    """Low-score correction factor."""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


class DixonColesModel(PredictionModel):
    def __init__(self, decay: float = DECAY):
        self.decay = decay
        self.attack_: dict[str, float] = {}
        self.defense_: dict[str, float] = {}
        self.home_adv_: float = 0.0
        self.rho_: float = 0.0
        self.teams_: list[str] = []
        self._ref_date = None

    def fit(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series,
            df_raw: pd.DataFrame | None = None):
        """
        X must contain home_team, away_team, date columns (passed in as extra cols).
        y_home / y_away are goal counts.
        df_raw: original matches DataFrame with those extra columns.
        """
        if df_raw is None:
            # Fall back to Poisson-only
            from models.poisson_model import PoissonModel
            self._fallback = PoissonModel()
            self._fallback.fit(X, y_home, y_away)
            return

        teams = sorted(set(df_raw["home_team"]) | set(df_raw["away_team"]))
        self.teams_ = teams
        t_idx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)

        ref_date = df_raw["date"].max()
        self._ref_date = ref_date

        weights = np.exp(-self.decay * (ref_date - df_raw["date"]).dt.days.values)
        home_teams = df_raw["home_team"].values
        away_teams = df_raw["away_team"].values
        y_h = y_home.values.astype(float)
        y_a = y_away.values.astype(float)

        def pack(attack, defense, home_adv, rho):
            return np.concatenate([attack, defense, [home_adv, rho]])

        def unpack(params):
            a = params[:n_teams]
            d = params[n_teams:2*n_teams]
            h = params[-2]
            r = params[-1]
            return a, d, h, r

        def neg_ll(params):
            a, d, h, r = unpack(params)
            ll = 0.0
            for i in range(len(home_teams)):
                hi = t_idx.get(home_teams[i], 0)
                ai = t_idx.get(away_teams[i], 0)
                lam = np.exp(a[hi] - d[ai] + h)
                mu = np.exp(a[ai] - d[hi])
                lam = max(lam, 1e-6)
                mu = max(mu, 1e-6)
                x, y = int(y_h[i]), int(y_a[i])
                tau = _tau(x, y, lam, mu, r)
                tau = max(tau, 1e-10)
                score_ll = (
                    x * np.log(lam) - lam - sum(np.log(range(1, x+1)) if x > 0 else [0]) +
                    y * np.log(mu) - mu - sum(np.log(range(1, y+1)) if y > 0 else [0]) +
                    np.log(tau)
                )
                ll += weights[i] * score_ll
            return -ll

        x0 = pack(
            np.zeros(n_teams),
            np.zeros(n_teams),
            0.1,
            -0.1,
        )
        bounds = (
            [(-3, 3)] * n_teams +
            [(-3, 3)] * n_teams +
            [(-1, 2), (-0.99, 0.99)]
        )
        try:
            res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 1000, "ftol": 1e-8})
            a, d, h, r = unpack(res.x)
        except Exception:
            a, d, h, r = unpack(x0)

        self.attack_ = {t: float(a[i]) for t, i in t_idx.items()}
        self.defense_ = {t: float(d[i]) for t, i in t_idx.items()}
        self.home_adv_ = float(h)
        self.rho_ = float(r)

    def _lambdas(self, home_team: str, away_team: str, neutral: bool = True) -> tuple[float, float]:
        a_h = self.attack_.get(home_team, 0.0)
        d_h = self.defense_.get(home_team, 0.0)
        a_a = self.attack_.get(away_team, 0.0)
        d_a = self.defense_.get(away_team, 0.0)
        h = 0.0 if neutral else self.home_adv_
        lam = max(np.exp(a_h - d_a + h), 0.1)
        mu = max(np.exp(a_a - d_h), 0.1)
        return lam, mu

    def predict_goals(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self, "_fallback"):
            return self._fallback.predict_goals(X)
        lam_h = np.zeros(len(X))
        lam_a = np.zeros(len(X))
        for i, (_, row) in enumerate(X.iterrows()):
            ht = row.get("home_team", "")
            at = row.get("away_team", "")
            neutral = bool(row.get("is_neutral", True))
            lam_h[i], lam_a[i] = self._lambdas(ht, at, neutral)
        return lam_h, lam_a
