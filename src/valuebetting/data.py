from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_MATCH_COLUMNS = {
    "match_id",
    "kickoff_time",
    "league",
    "season",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
}

REQUIRED_ODDS_COLUMNS = {
    "match_id",
    "snapshot_time",
    "bookmaker",
    "market",
    "selection",
    "decimal_odds",
    "is_opening",
}


class DataValidationError(ValueError):
    """Raised when input data is missing required structure or values."""



def _validate_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise DataValidationError(f"{name} is missing required columns: {sorted(missing)}")



def load_matches(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _validate_columns(frame, REQUIRED_MATCH_COLUMNS, "matches")
    frame = frame.copy()
    frame["kickoff_time"] = pd.to_datetime(frame["kickoff_time"], format="mixed", utc=True).dt.tz_localize(None)
    frame["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce")
    frame["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce")
    frame = frame.sort_values(["kickoff_time", "match_id"]).reset_index(drop=True)
    if frame["match_id"].duplicated().any():
        raise DataValidationError("matches.csv contains duplicate match_id values.")
    frame["is_completed"] = frame[["home_goals", "away_goals"]].notna().all(axis=1).astype(int)
    partial_scores = frame[["home_goals", "away_goals"]].notna().any(axis=1) & (frame["is_completed"] == 0)
    if partial_scores.any():
        raise DataValidationError("matches.csv contains partial score rows. home_goals and away_goals must both be filled or both be blank.")
    frame["total_goals"] = frame["home_goals"] + frame["away_goals"]
    frame["home_win"] = np.where(frame["is_completed"] == 1, (frame["home_goals"] > frame["away_goals"]).astype(float), np.nan)
    frame["draw"] = np.where(frame["is_completed"] == 1, (frame["home_goals"] == frame["away_goals"]).astype(float), np.nan)
    frame["away_win"] = np.where(frame["is_completed"] == 1, (frame["home_goals"] < frame["away_goals"]).astype(float), np.nan)
    frame["over_0_5"] = np.where(frame["is_completed"] == 1, (frame["total_goals"] > 0.5).astype(float), np.nan)
    frame["over_1_5"] = np.where(frame["is_completed"] == 1, (frame["total_goals"] > 1.5).astype(float), np.nan)
    frame["over_2_5"] = np.where(frame["is_completed"] == 1, (frame["total_goals"] > 2.5).astype(float), np.nan)
    frame["btts"] = np.where(frame["is_completed"] == 1, ((frame["home_goals"] > 0) & (frame["away_goals"] > 0)).astype(float), np.nan)
    return frame



def load_odds_snapshots(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _validate_columns(frame, REQUIRED_ODDS_COLUMNS, "odds_snapshots")
    frame = frame.copy()
    frame["snapshot_time"] = pd.to_datetime(frame["snapshot_time"], format="mixed", utc=True).dt.tz_localize(None)
    frame["decimal_odds"] = frame["decimal_odds"].astype(float)
    frame["is_opening"] = frame["is_opening"].astype(int)
    invalid_odds = frame.loc[frame["decimal_odds"] <= 1.0]
    if not invalid_odds.empty:
        raise DataValidationError("Decimal odds must be greater than 1.0.")
    return frame.sort_values(["snapshot_time", "match_id", "market", "selection"]).reset_index(drop=True)



def select_snapshot_before_kickoff(
    matches: pd.DataFrame,
    odds: pd.DataFrame,
    minutes_before_kickoff: int,
) -> pd.DataFrame:
    merged = odds.merge(matches[["match_id", "kickoff_time"]], on="match_id", how="inner")
    merged["minutes_to_kickoff"] = (merged["kickoff_time"] - merged["snapshot_time"]).dt.total_seconds() / 60.0
    eligible = merged.loc[merged["minutes_to_kickoff"] >= minutes_before_kickoff].copy()
    if eligible.empty:
        return pd.DataFrame(columns=list(odds.columns) + ["kickoff_time", "minutes_to_kickoff"])
    eligible = eligible.sort_values(
        ["match_id", "bookmaker", "market", "selection", "snapshot_time"],
        ascending=[True, True, True, True, False],
    )
    selected = eligible.drop_duplicates(["match_id", "bookmaker", "market", "selection"], keep="first")
    return selected.reset_index(drop=True)



def opening_snapshots(odds: pd.DataFrame) -> pd.DataFrame:
    frame = odds.loc[odds["is_opening"] == 1].copy()
    frame = frame.sort_values(["match_id", "bookmaker", "market", "selection", "snapshot_time"])
    return frame.drop_duplicates(["match_id", "bookmaker", "market", "selection"], keep="first").reset_index(drop=True)
