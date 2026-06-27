"""Temporal backtesting engine — enforces data cutoff per match."""
import numpy as np
import pandas as pd
from evaluation.metrics import evaluate_predictions, EvalResult


def backtest(
    model,
    df_all: pd.DataFrame,
    elo_history: dict,
    val_matches: pd.DataFrame,
    feature_fn,
    llm_features_map: dict | None = None,
    dc_mode: bool = False,
) -> EvalResult:
    """
    For each match in val_matches, train on all data before that match,
    then predict. Returns aggregate EvalResult.
    """
    pred_proba_list = []
    pred_h_list = []
    pred_a_list = []
    true_h_list = []
    true_a_list = []

    val_matches = val_matches.sort_values("date")

    for _, row in val_matches.iterrows():
        match_date = row["date"]
        cutoff = match_date - pd.Timedelta(days=1)

        train = df_all[df_all["date"] < match_date].copy()
        if len(train) < 10:
            continue

        from data.features import build_dataset, STAT_FEATURE_COLS
        X_train, yh_train, ya_train, _ = build_dataset(
            df_all, elo_history, train, llm_features_map
        )

        if dc_mode:
            model.fit(X_train, yh_train, ya_train, df_raw=train)
        else:
            model.fit(X_train, yh_train, ya_train)

        match_id = f"{row['date'].date()}_{row['home_team']}_{row['away_team']}"
        llm_feats = (llm_features_map or {}).get(match_id)
        feats = feature_fn(
            df_all, elo_history,
            row["home_team"], row["away_team"],
            match_date, row.get("stage", "group"),
            bool(row.get("neutral", True)),
            llm_feats,
        )

        X_pred = pd.DataFrame([feats])
        if dc_mode:
            X_pred["home_team"] = row["home_team"]
            X_pred["away_team"] = row["away_team"]
            X_pred["is_neutral"] = int(row.get("neutral", True))

        proba = model.predict_proba(X_pred)
        lam_h, lam_a = model.predict_goals(X_pred)

        pred_proba_list.append(proba[0])
        pred_h_list.append(int(round(lam_h[0])))
        pred_a_list.append(int(round(lam_a[0])))
        true_h_list.append(row["home_score"])
        true_a_list.append(row["away_score"])

    if not pred_proba_list:
        return EvalResult()

    return evaluate_predictions(
        np.array(pred_proba_list),
        np.array(pred_h_list),
        np.array(pred_a_list),
        np.array(true_h_list),
        np.array(true_a_list),
    )


def fast_eval(
    model,
    df_all: pd.DataFrame,
    elo_history: dict,
    val_matches: pd.DataFrame,
    feature_fn,
    llm_features_map: dict | None = None,
    dc_mode: bool = False,
) -> EvalResult:
    """
    Train once on all data before the earliest val match, evaluate on all val matches.
    Faster but slightly leaky for sequential tournaments.
    """
    earliest = val_matches["date"].min()
    train = df_all[df_all["date"] < earliest].copy()
    if len(train) < 10:
        return EvalResult()

    from data.features import build_dataset
    X_train, yh_train, ya_train, _ = build_dataset(
        df_all, elo_history, train, llm_features_map
    )

    if dc_mode:
        model.fit(X_train, yh_train, ya_train, df_raw=train)
    else:
        model.fit(X_train, yh_train, ya_train)

    rows, true_h, true_a = [], [], []
    for _, row in val_matches.iterrows():
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

    X_val = pd.DataFrame(rows)
    proba = model.predict_proba(X_val)
    lam_h, lam_a = model.predict_goals(X_val)

    return evaluate_predictions(
        proba,
        np.round(lam_h).astype(int),
        np.round(lam_a).astype(int),
        np.array(true_h),
        np.array(true_a),
    )
