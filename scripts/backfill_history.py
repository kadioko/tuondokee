"""Backfill real EPL historical match data from football-data.org."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from valuebetting.fetch import (
    COMPETITION_CONFIG,
    FOOTBALL_DATA_BASE_URL,
    _competition_settings,
    _get_env_api_key,
    _http_json,
    football_data_matches_to_frame,
    update_matches_csv,
)


def fetch_season(competition: str, season_year: int, api_key: str):
    settings = _competition_settings(competition)
    url = f"{FOOTBALL_DATA_BASE_URL}/{settings['football_data_code']}/matches?season={season_year}"
    payload = _http_json(url, headers={"X-Auth-Token": api_key})
    frame = football_data_matches_to_frame(payload, competition=competition)
    print(f"  {competition} {season_year}: {len(frame)} matches")
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill historical matches from football-data.org")
    parser.add_argument("--competitions", nargs="*", default=["EPL"], help=f"Competitions to backfill: {', '.join(COMPETITION_CONFIG)}")
    parser.add_argument("--seasons", nargs="*", type=int, default=[2025], help="Season start years to fetch, e.g. 2023 2024 2025")
    parser.add_argument("--matches-csv", default="data/sample/matches.csv")
    parser.add_argument("--sleep-seconds", type=int, default=7)
    return parser


def main():
    args = build_parser().parse_args()
    api_key = _get_env_api_key("FOOTBALL_DATA_API_KEY")
    matches_csv = Path(args.matches_csv)
    results: list[dict[str, object]] = []

    print(f"Backfilling competitions {args.competitions} for seasons: {args.seasons}")

    for competition in args.competitions:
        for season_year in args.seasons:
            try:
                frame = fetch_season(competition, season_year, api_key)
                if not frame.empty:
                    update_matches_csv(matches_csv, frame)
                    print(f"  -> Updated {matches_csv}")
                results.append(
                    {
                        "competition": competition,
                        "season": season_year,
                        "status": "ok",
                        "matches_fetched": int(len(frame)),
                    }
                )
            except HTTPError as exc:
                results.append(
                    {
                        "competition": competition,
                        "season": season_year,
                        "status": "failed",
                        "error_type": "http_error",
                        "status_code": int(exc.code),
                        "message": "Competition or season unavailable on current API tier.",
                    }
                )
                print(f"  {competition} {season_year}: skipped ({exc.code})")
            except Exception as exc:
                results.append(
                    {
                        "competition": competition,
                        "season": season_year,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                print(f"  {competition} {season_year}: failed ({exc})")
            time.sleep(args.sleep_seconds)

    import pandas as pd

    final = pd.read_csv(matches_csv)
    completed = final["home_goals"].notna().sum()
    upcoming = final["home_goals"].isna().sum()
    print(f"\nFinal dataset: {len(final)} total rows, {completed} completed, {upcoming} upcoming")
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
