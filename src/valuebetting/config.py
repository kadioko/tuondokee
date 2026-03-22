from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass
class PathsConfig:
    matches_csv: str
    odds_csv: str
    artifacts_dir: str


@dataclass
class FeatureConfig:
    snapshot_minutes_before_kickoff: int = 60
    form_windows: tuple[int, ...] = (5, 10)
    elo_k: float = 20.0
    min_history_matches: int = 3
    league_filters: tuple[str, ...] = ()


@dataclass
class ModelConfig:
    market: str = "1x2"
    target: Literal["three_way", "binary"] = "three_way"
    binary_market: Literal["over_0_5", "over_1_5", "over_2_5", "btts", "home_win"] = "over_2_5"
    calibration: bool = True
    n_splits: int = 5
    gap_matches: int = 0
    random_state: int = 42


@dataclass
class TuningConfig:
    enabled: bool = True
    n_trials: int = 25
    timeout_seconds: int | None = None


@dataclass
class BettingConfig:
    initial_bankroll: float = 1000.0
    staking: Literal["flat", "fractional_kelly"] = "flat"
    flat_stake: float = 25.0
    fractional_kelly_fraction: float = 0.25
    minimum_edge: float = 0.03
    max_stake_pct: float = 0.05
    bookmaker: str | None = None


@dataclass
class AppConfig:
    paths: PathsConfig
    features: FeatureConfig
    model: ModelConfig
    tuning: TuningConfig
    betting: BettingConfig


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "matches_csv": "data/sample/matches.csv",
        "odds_csv": "data/sample/odds_snapshots.csv",
        "artifacts_dir": "artifacts",
    },
    "features": {
        "snapshot_minutes_before_kickoff": 60,
        "form_windows": [5, 10],
        "elo_k": 20.0,
        "min_history_matches": 3,
        "league_filters": [],
    },
    "model": {
        "market": "1x2",
        "target": "three_way",
        "binary_market": "over_2_5",
        "calibration": True,
        "n_splits": 4,
        "gap_matches": 0,
        "random_state": 42,
    },
    "tuning": {
        "enabled": True,
        "n_trials": 20,
        "timeout_seconds": 600,
    },
    "betting": {
        "initial_bankroll": 1000.0,
        "staking": "flat",
        "flat_stake": 25.0,
        "fractional_kelly_fraction": 0.25,
        "minimum_edge": 0.03,
        "max_stake_pct": 0.05,
        "bookmaker": None,
    },
}


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = json.load(handle)
    return AppConfig(
        paths=PathsConfig(**raw["paths"]),
        features=FeatureConfig(
            snapshot_minutes_before_kickoff=raw["features"]["snapshot_minutes_before_kickoff"],
            form_windows=tuple(raw["features"]["form_windows"]),
            elo_k=raw["features"]["elo_k"],
            min_history_matches=raw["features"]["min_history_matches"],
            league_filters=tuple(raw["features"].get("league_filters", [])),
        ),
        model=ModelConfig(**raw["model"]),
        tuning=TuningConfig(**raw["tuning"]),
        betting=BettingConfig(**raw["betting"]),
    )


def write_default_config(path: str | Path) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(DEFAULT_CONFIG, handle, indent=2)
