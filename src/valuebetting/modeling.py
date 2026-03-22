from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


@dataclass
class TrainArtifacts:
    estimator: Any
    feature_columns: list[str]
    categorical_columns: list[str]
    metrics: dict[str, float]
    classes_: list[str]
    target_name: str
    binary_market: str | None
    calibration_table: pd.DataFrame



def prepare_target(dataset: pd.DataFrame, target_type: str, binary_market: str = "over_2_5") -> tuple[np.ndarray, str, list[str]]:
    if target_type == "three_way":
        outcome = np.select(
            [dataset["home_goals"] > dataset["away_goals"], dataset["home_goals"] == dataset["away_goals"]],
            ["home", "draw"],
            default="away",
        )
        return outcome, "result_1x2", ["away", "draw", "home"]
    if target_type == "binary":
        if binary_market == "over_0_5":
            outcome = dataset["over_0_5"].astype(int).to_numpy()
            return outcome, "over_0_5", ["0", "1"]
        if binary_market == "over_1_5":
            outcome = dataset["over_1_5"].astype(int).to_numpy()
            return outcome, "over_1_5", ["0", "1"]
        if binary_market == "over_2_5":
            outcome = dataset["over_2_5"].astype(int).to_numpy()
            return outcome, "over_2_5", ["0", "1"]
        if binary_market == "btts":
            outcome = dataset["btts"].astype(int).to_numpy()
            return outcome, "btts", ["0", "1"]
        if binary_market == "home_win":
            outcome = dataset["home_win"].astype(int).to_numpy()
            return outcome, "home_win", ["0", "1"]
        raise ValueError(f"Unsupported binary market: {binary_market}")
    raise ValueError(f"Unsupported target type: {target_type}")



def make_pipeline(
    categorical_columns: list[str],
    random_state: int,
    target_type: str,
    calibration: bool,
    calibration_cv: int,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0.0))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_columns),
        ],
        remainder=numeric_transformer,
    )

    base_params: dict[str, Any] = {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": random_state,
        "verbosity": -1,
    }
    if params:
        base_params.update(params)

    if target_type == "three_way":
        base_model: Any = lgb.LGBMClassifier(objective="multiclass", num_class=3, **base_params)
    else:
        base_model = lgb.LGBMClassifier(objective="binary", **base_params)

    estimator: Any = base_model
    if calibration and calibration_cv >= 2:
        # Random shuffles are invalid in betting because future matches leak regime information into the past.
        # We calibrate with a deterministic chronological split via cv=3 inside the training fold only.
        estimator = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=calibration_cv)

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])



def predict_probabilities(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        probabilities = estimator.predict_proba(X)
    return np.asarray(probabilities)


def build_calibration_table(y_true: np.ndarray, probabilities: np.ndarray, target_type: str, bin_count: int = 10) -> pd.DataFrame:
    if target_type == "binary":
        prob = probabilities[:, 1]
        table = pd.DataFrame({"y_true": y_true.astype(int), "predicted_probability": prob})
        table["bin"] = pd.cut(table["predicted_probability"], bins=np.linspace(0.0, 1.0, bin_count + 1), include_lowest=True)
        grouped = table.groupby("bin", observed=False).agg(
            sample_count=("y_true", "count"),
            mean_predicted_probability=("predicted_probability", "mean"),
            empirical_win_rate=("y_true", "mean"),
        )
        grouped = grouped.reset_index()
        grouped["calibration_gap"] = grouped["mean_predicted_probability"] - grouped["empirical_win_rate"]
        return grouped

    max_prob = probabilities.max(axis=1)
    predicted_class = probabilities.argmax(axis=1)
    correct = (predicted_class == y_true).astype(int)
    table = pd.DataFrame({"correct": correct, "predicted_probability": max_prob})
    table["bin"] = pd.cut(table["predicted_probability"], bins=np.linspace(0.0, 1.0, bin_count + 1), include_lowest=True)
    grouped = table.groupby("bin", observed=False).agg(
        sample_count=("correct", "count"),
        mean_predicted_probability=("predicted_probability", "mean"),
        empirical_win_rate=("correct", "mean"),
    )
    grouped = grouped.reset_index()
    grouped["calibration_gap"] = grouped["mean_predicted_probability"] - grouped["empirical_win_rate"]
    return grouped



def _resolve_calibration_cv(y_train: np.ndarray, calibration: bool) -> int:
    if not calibration:
        return 0
    class_counts = np.bincount(y_train)
    if len(class_counts) == 0:
        return 0
    min_class_count = int(class_counts.min()) if class_counts.size > 0 else 0
    return max(0, min(3, min_class_count))



def _has_required_class_coverage(y_train: np.ndarray, target_type: str) -> bool:
    unique_classes = np.unique(y_train)
    if target_type == "three_way":
        return len(unique_classes) == 3
    return len(unique_classes) == 2



def evaluate_time_series(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    target_type: str,
    binary_market: str,
    n_splits: int,
    random_state: int,
    calibration: bool,
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    X = dataset[feature_columns]
    y, _, class_order = prepare_target(dataset, target_type, binary_market)
    if target_type == "three_way":
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
    else:
        y_encoded = y.astype(int)
        encoder = None

    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds: list[dict[str, float]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        if not _has_required_class_coverage(y_train, target_type):
            continue
        calibration_cv = _resolve_calibration_cv(y_train, calibration)
        pipeline = make_pipeline(categorical_columns, random_state, target_type, calibration, calibration_cv, params)
        pipeline.fit(X_train, y_train)
        proba = predict_probabilities(pipeline, X_test)
        fold_log_loss = log_loss(y_test, proba, labels=np.arange(proba.shape[1]))
        if target_type == "binary":
            fold_brier = brier_score_loss(y_test, proba[:, 1])
        else:
            indicator = np.eye(proba.shape[1])[y_test]
            fold_brier = float(np.mean(np.sum((proba - indicator) ** 2, axis=1)))
        folds.append({"fold": float(fold_idx), "log_loss": float(fold_log_loss), "brier_score": float(fold_brier)})

    if not folds:
        raise ValueError(
            "No valid chronological folds were available with sufficient class coverage. Increase history or reduce complexity."
        )

    metrics = {
        "cv_log_loss_mean": float(np.mean([fold["log_loss"] for fold in folds])),
        "cv_brier_mean": float(np.mean([fold["brier_score"] for fold in folds])),
    }
    if target_type == "three_way" and encoder is not None:
        metrics["class_count"] = float(len(encoder.classes_))
    else:
        metrics["class_count"] = 2.0
    return metrics, folds



def train_final_model(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    target_type: str,
    binary_market: str,
    random_state: int,
    calibration: bool,
    params: dict[str, Any] | None = None,
) -> TrainArtifacts:
    y, target_name, class_order = prepare_target(dataset, target_type, binary_market)
    X = dataset[feature_columns]
    if target_type == "three_way":
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y)
        classes = list(encoder.classes_)
    else:
        y_train = y.astype(int)
        classes = class_order
    calibration_cv = _resolve_calibration_cv(y_train, calibration)
    pipeline = make_pipeline(categorical_columns, random_state, target_type, calibration, calibration_cv, params)
    pipeline.fit(X, y_train)
    predictions = predict_probabilities(pipeline, X)
    metrics = {
        "train_log_loss": float(log_loss(y_train, predictions, labels=np.arange(predictions.shape[1]))),
        "target_type": 1.0 if target_type == "binary" else 3.0,
    }
    if target_type == "binary":
        metrics["train_brier_score"] = float(brier_score_loss(y_train, predictions[:, 1]))
    else:
        indicator = np.eye(predictions.shape[1])[y_train]
        metrics["train_brier_score"] = float(np.mean(np.sum((predictions - indicator) ** 2, axis=1)))
    calibration_table = build_calibration_table(y_train, predictions, target_type)
    return TrainArtifacts(
        estimator=pipeline,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        metrics=metrics,
        classes_=classes,
        target_name=target_name,
        binary_market=binary_market if target_type == "binary" else None,
        calibration_table=calibration_table,
    )



def save_train_artifacts(artifacts: TrainArtifacts, output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.estimator, path / "model.joblib")
    artifacts.calibration_table.to_csv(path / "calibration_diagnostics.csv", index=False)
    metadata = {
        "feature_columns": artifacts.feature_columns,
        "categorical_columns": artifacts.categorical_columns,
        "metrics": artifacts.metrics,
        "classes": artifacts.classes_,
        "target_name": artifacts.target_name,
        "binary_market": artifacts.binary_market,
    }
    with (path / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)



def load_model_artifacts(output_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    path = Path(output_dir)
    model = joblib.load(path / "model.joblib")
    with (path / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata: dict[str, Any] = json.load(handle)
    return model, metadata
