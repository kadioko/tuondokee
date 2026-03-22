from __future__ import annotations

from typing import Any

import optuna
import pandas as pd

from valuebetting.modeling import evaluate_time_series



def tune_lightgbm(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    target_type: str,
    binary_market: str,
    n_splits: int,
    random_state: int,
    calibration: bool,
    n_trials: int,
    timeout_seconds: int | None,
) -> tuple[dict[str, Any], optuna.study.Study]:
    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        metrics, _ = evaluate_time_series(
            dataset=dataset,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            target_type=target_type,
            binary_market=binary_market,
            n_splits=n_splits,
            random_state=random_state,
            calibration=calibration,
            params=params,
        )
        return metrics["cv_log_loss_mean"]

    study = optuna.create_study(direction="minimize", study_name="valuebetting_lgbm")
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)
    return study.best_params, study
