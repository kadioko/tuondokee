from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRESETS_DIR = ROOT / "config" / "presets"
PRESETS_DIR.mkdir(parents=True, exist_ok=True)

LEAGUES = {
    "epl": "EPL",
    "laliga": "La Liga",
    "bundesliga": "Bundesliga",
    "ucl": "UCL",
    "uel": "UEL",
}

MARKETS = {
    "1x2": {"market": "1x2", "target": "three_way", "binary_market": "over_2_5", "artifacts_suffix": "1x2", "staking": "flat"},
    "over05": {"market": "over_0_5", "target": "binary", "binary_market": "over_0_5", "artifacts_suffix": "over05", "staking": "fractional_kelly"},
    "over15": {"market": "over_1_5", "target": "binary", "binary_market": "over_1_5", "artifacts_suffix": "over15", "staking": "flat"},
    "over25": {"market": "over_2_5", "target": "binary", "binary_market": "over_2_5", "artifacts_suffix": "over25", "staking": "flat"},
    "btts": {"market": "btts", "target": "binary", "binary_market": "btts", "artifacts_suffix": "btts", "staking": "fractional_kelly"},
    "homewin": {"market": "home_win", "target": "binary", "binary_market": "home_win", "artifacts_suffix": "homewin", "staking": "flat"},
}

for league_key, league_label in LEAGUES.items():
    for preset_key, market in MARKETS.items():
        config = {
            "paths": {
                "matches_csv": "data/sample/matches.csv",
                "odds_csv": "data/sample/odds_snapshots.csv",
                "artifacts_dir": f"artifacts_{league_key}_{market['artifacts_suffix']}",
            },
            "features": {
                "snapshot_minutes_before_kickoff": 60,
                "form_windows": [5, 10],
                "elo_k": 20.0,
                "min_history_matches": 3,
                "league_filters": [league_label],
            },
            "model": {
                "market": market["market"],
                "target": market["target"],
                "binary_market": market["binary_market"],
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
                "staking": market["staking"],
                "flat_stake": 25.0,
                "fractional_kelly_fraction": 0.25,
                "minimum_edge": 0.03,
                "max_stake_pct": 0.05,
                "bookmaker": None,
            },
        }
        output_path = PRESETS_DIR / f"{league_key}_{preset_key}_config.json"
        output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

print(f"Wrote {len(LEAGUES) * len(MARKETS)} preset configs to {PRESETS_DIR}")
