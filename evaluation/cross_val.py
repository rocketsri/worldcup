"""
Rolling WC cross-validation: train on prior tournaments, test on next.
This is more rigorous than fast_eval (which trains on all pre-val data globally).

Folds (chronological):
  Fold 1: Train 2006      → Test 2010
  Fold 2: Train 2006+2010 → Test 2014
  Fold 3: Train 2006+2010+2014 → Test 2018
  Fold 4: Train 2006+2010+2014+2018 → Test 2022
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from evaluation.metrics import evaluate_predictions, EvalResult

WC_YEARS = [2006, 2010, 2014, 2018, 2022]
FOLD_SPLITS = [
    ([2006],                    2010),
    ([2006, 2010],              2014),
    ([2006, 2010, 2014],        2018),
    ([2006, 2010, 2014, 2018],  2022),
]


@dataclass
class CrossValResult:
    fold_results: list[tuple[int, EvalResult]] = field(default_factory=list)

    @property
    def mean_winner_acc(self) -> float:
        accs = [r.winner_acc for _, r in self.fold_results]
        return float(np.mean(accs)) if accs else 0.0

    @property
    def mean_rps(self) -> float:
        vals = [r.rps for _, r in self.fold_results]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def mean_mae(self) -> float:
        vals = [r.mae_goals for _, r in self.fold_results]
        return float(np.mean(vals)) if vals else 0.0

    def __repr__(self):
        lines = [f"Cross-WC Validation ({len(self.fold_results)} folds):"]
        for yr, r in self.fold_results:
            lines.append(f"  Test {yr}: winner_acc={r.winner_acc:.3f} rps={r.rps:.3f} mae={r.mae_goals:.3f} n={r.n_games}")
        lines.append(f"  MEAN:      winner_acc={self.mean_winner_acc:.3f} rps={self.mean_rps:.3f} mae={self.mean_mae:.3f}")
        return "\n".join(lines)


def rolling_wc_cv(
    model_factory,
    df_all: pd.DataFrame,
    elo_history: dict,
    feature_fn,
    llm_features_map: dict | None = None,
    dc_mode: bool = False,
) -> CrossValResult:
    """
    Run rolling cross-validation across WC folds.
    model_factory: callable() → fresh model instance
    """
    from data.features import build_dataset
    result = CrossValResult()

    for train_years, test_year in FOLD_SPLITS:
        train = df_all[df_all["year"].isin(train_years)].copy()
        test = df_all[df_all["year"] == test_year].copy()

        if len(train) < 10 or len(test) == 0:
            continue

        # Build training features using ELO up to start of test year
        X_train, yh_train, ya_train, _ = build_dataset(
            df_all, elo_history, train, llm_features_map
        )

        model = model_factory()
        if dc_mode:
            model.fit(X_train, yh_train, ya_train, df_raw=train)
        else:
            model.fit(X_train, yh_train, ya_train)

        # Predict test year
        rows, true_h, true_a = [], [], []
        for _, row in test.iterrows():
            match_id = f"{row['date'].date()}_{row['home_team']}_{row['away_team']}"
            llm_feats = (llm_features_map or {}).get(match_id)
            feats = feature_fn(
                df_all, elo_history,
                row["home_team"], row["away_team"],
                row["date"], row.get("stage", "group"),
                bool(row.get("neutral", True)),
                llm_feats,
            )
            if dc_mode:
                feats["home_team"] = row["home_team"]
                feats["away_team"] = row["away_team"]
            rows.append(feats)
            true_h.append(row["home_score"])
            true_a.append(row["away_score"])

        X_test = pd.DataFrame(rows)
        proba = model.predict_proba(X_test)
        lam_h, lam_a = model.predict_goals(X_test)

        fold_result = evaluate_predictions(
            proba,
            np.round(lam_h).astype(int),
            np.round(lam_a).astype(int),
            np.array(true_h),
            np.array(true_a),
        )
        result.fold_results.append((test_year, fold_result))
        print(f"    Fold {train_years[-1]}→{test_year}: winner_acc={fold_result.winner_acc:.3f} n={fold_result.n_games}")

    return result
