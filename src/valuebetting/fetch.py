from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4/competitions"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"
COMPETITION_CONFIG: dict[str, dict[str, str]] = {
    "EPL": {"football_data_code": "PL", "odds_api_sport": "soccer_epl", "match_prefix": "FD_PL", "league_label": "EPL"},
    "LaLiga": {"football_data_code": "PD", "odds_api_sport": "soccer_spain_la_liga", "match_prefix": "FD_PD", "league_label": "La Liga"},
    "Bundesliga": {"football_data_code": "BL1", "odds_api_sport": "soccer_germany_bundesliga", "match_prefix": "FD_BL1", "league_label": "Bundesliga"},
    "UCL": {"football_data_code": "CL", "odds_api_sport": "soccer_uefa_champs_league", "match_prefix": "FD_CL", "league_label": "UEFA Champions League"},
    "UEL": {"football_data_code": "EL", "odds_api_sport": "soccer_uefa_europa_league", "match_prefix": "FD_EL", "league_label": "UEFA Europa League"},
}


def _competition_settings(competition: str) -> dict[str, str]:
    if competition not in COMPETITION_CONFIG:
        raise FetchError(f"Unsupported competition: {competition}. Supported values: {sorted(COMPETITION_CONFIG)}")
    return COMPETITION_CONFIG[competition]


class FetchError(ValueError):
    pass



def _get_env_api_key(env_name: str) -> str:
    value = os.getenv(env_name)
    if not value:
        raise FetchError(f"Missing required API key in environment variable: {env_name}")
    return value



def _http_json(url: str, headers: dict[str, str] | None = None) -> Any:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))



def _season_label_from_match(match: dict[str, Any]) -> str:
    season = match.get("season") or {}
    start = str(season.get("startDate", ""))[:4]
    end = str(season.get("endDate", ""))[:4]
    if start and end:
        return f"{start}/{end}"
    utc_date = str(match.get("utcDate", ""))[:4]
    return f"{utc_date}/{utc_date}" if utc_date else "unknown/unknown"



def football_data_matches_to_frame(payload: dict[str, Any], competition: str = "EPL") -> pd.DataFrame:
    settings = _competition_settings(competition)
    rows: list[dict[str, Any]] = []
    for match in payload.get("matches", []):
        status = str(match.get("status", ""))
        full_time = (match.get("score") or {}).get("fullTime") or {}
        home_goals = full_time.get("home") if status == "FINISHED" else None
        away_goals = full_time.get("away") if status == "FINISHED" else None
        rows.append(
            {
                "match_id": f"{settings['match_prefix']}_{match['id']}",
                "kickoff_time": pd.to_datetime(match["utcDate"]).tz_convert(None) if pd.Timestamp(match["utcDate"]).tzinfo else pd.to_datetime(match["utcDate"]),
                "league": settings["league_label"],
                "season": _season_label_from_match(match),
                "home_team": (match.get("homeTeam") or {}).get("name", ""),
                "away_team": (match.get("awayTeam") or {}).get("name", ""),
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["match_id", "kickoff_time", "league", "season", "home_team", "away_team", "home_goals", "away_goals"])
    return frame.sort_values(["kickoff_time", "match_id"]).reset_index(drop=True)



def _normalize_team_name(name: str) -> str:
    cleaned = "".join(ch.lower() for ch in name if ch.isalnum())
    return cleaned.replace("fc", "")



def _build_fixture_lookup(fixtures: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    lookup: dict[tuple[str, str, str], str] = {}
    for _, row in fixtures.iterrows():
        kickoff_key = pd.Timestamp(row["kickoff_time"]).strftime("%Y-%m-%dT%H:%M")
        key = (_normalize_team_name(str(row["home_team"])), _normalize_team_name(str(row["away_team"])), kickoff_key)
        lookup[key] = str(row["match_id"])
    return lookup



def odds_api_events_to_frame(payload: list[dict[str, Any]], fixtures: pd.DataFrame, snapshot_time: pd.Timestamp | None = None) -> pd.DataFrame:
    lookup = _build_fixture_lookup(fixtures)
    snapshot = snapshot_time or pd.Timestamp(datetime.now(timezone.utc)).tz_convert(None)
    rows: list[dict[str, Any]] = []
    for event in payload:
        kickoff = pd.to_datetime(event["commence_time"])
        kickoff_naive = kickoff.tz_convert(None) if kickoff.tzinfo is not None else kickoff
        kickoff_key = kickoff_naive.strftime("%Y-%m-%dT%H:%M")
        match_id = lookup.get(
            (
                _normalize_team_name(str(event.get("home_team", ""))),
                _normalize_team_name(str(event.get("away_team", ""))),
                kickoff_key,
            )
        )
        if not match_id:
            continue
        for bookmaker in event.get("bookmakers", []):
            bookmaker_name = str(bookmaker.get("title", "Unknown"))
            markets = bookmaker.get("markets", [])
            for market in markets:
                market_key = str(market.get("key", ""))
                outcomes = market.get("outcomes", [])
                if market_key == "h2h":
                    for outcome in outcomes:
                        outcome_name = str(outcome.get("name", ""))
                        price = float(outcome["price"])
                        if outcome_name == event.get("home_team"):
                            selection = "home"
                        elif outcome_name == event.get("away_team"):
                            selection = "away"
                        elif outcome_name.lower() == "draw":
                            selection = "draw"
                        else:
                            continue
                        rows.append(
                            {
                                "match_id": match_id,
                                "snapshot_time": snapshot,
                                "bookmaker": bookmaker_name,
                                "market": "1x2",
                                "selection": selection,
                                "decimal_odds": price,
                                "is_opening": 0,
                            }
                        )
                    home_price = next((float(item["price"]) for item in outcomes if str(item.get("name")) == event.get("home_team")), None)
                    draw_price = next((float(item["price"]) for item in outcomes if str(item.get("name", "")).lower() == "draw"), None)
                    away_price = next((float(item["price"]) for item in outcomes if str(item.get("name")) == event.get("away_team")), None)
                    if home_price is not None and draw_price is not None and away_price is not None:
                        not_home_prob = (1.0 / draw_price) + (1.0 / away_price)
                        if not_home_prob > 0:
                            rows.append(
                                {
                                    "match_id": match_id,
                                    "snapshot_time": snapshot,
                                    "bookmaker": bookmaker_name,
                                    "market": "home_win",
                                    "selection": "home",
                                    "decimal_odds": home_price,
                                    "is_opening": 0,
                                }
                            )
                            rows.append(
                                {
                                    "match_id": match_id,
                                    "snapshot_time": snapshot,
                                    "bookmaker": bookmaker_name,
                                    "market": "home_win",
                                    "selection": "not_home",
                                    "decimal_odds": 1.0 / not_home_prob,
                                    "is_opening": 0,
                                }
                            )
                if market_key == "btts":
                    btts_yes = next((item for item in outcomes if str(item.get("name", "")).lower() in {"yes", "both teams to score"}), None)
                    btts_no = next((item for item in outcomes if str(item.get("name", "")).lower() == "no"), None)
                    if btts_yes and btts_no:
                        rows.append(
                            {
                                "match_id": match_id,
                                "snapshot_time": snapshot,
                                "bookmaker": bookmaker_name,
                                "market": "btts",
                                "selection": "btts_yes",
                                "decimal_odds": float(btts_yes["price"]),
                                "is_opening": 0,
                            }
                        )
                        rows.append(
                            {
                                "match_id": match_id,
                                "snapshot_time": snapshot,
                                "bookmaker": bookmaker_name,
                                "market": "btts",
                                "selection": "btts_no",
                                "decimal_odds": float(btts_no["price"]),
                                "is_opening": 0,
                            }
                        )
                if market_key == "totals":
                    over_05 = next((item for item in outcomes if float(item.get("point", -1)) == 0.5 and str(item.get("name", "")).lower() == "over"), None)
                    under_05 = next((item for item in outcomes if float(item.get("point", -1)) == 0.5 and str(item.get("name", "")).lower() == "under"), None)
                    over_15 = next((item for item in outcomes if float(item.get("point", -1)) == 1.5 and str(item.get("name", "")).lower() == "over"), None)
                    under_15 = next((item for item in outcomes if float(item.get("point", -1)) == 1.5 and str(item.get("name", "")).lower() == "under"), None)
                    over = next((item for item in outcomes if float(item.get("point", -1)) == 2.5 and str(item.get("name", "")).lower() == "over"), None)
                    under = next((item for item in outcomes if float(item.get("point", -1)) == 2.5 and str(item.get("name", "")).lower() == "under"), None)
                    if over_05 and under_05:
                        rows.append(
                            {
                                "match_id": match_id,
                                "snapshot_time": snapshot,
                                "bookmaker": bookmaker_name,
                                "market": "over_0_5",
                                "selection": "over_0_5",
                                "decimal_odds": float(over_05["price"]),
                                "is_opening": 0,
                            }
                        )
                        rows.append(
                            {
                                "match_id": match_id,
                                "snapshot_time": snapshot,
                                "bookmaker": bookmaker_name,
                                "market": "over_0_5",
                                "selection": "under_0_5",
                                "decimal_odds": float(under_05["price"]),
                                "is_opening": 0,
                            }
                        )
                    if over_15 and under_15:
                        rows.append(
                            {
                                "match_id": match_id,
                                "snapshot_time": snapshot,
                                "bookmaker": bookmaker_name,
                                "market": "over_1_5",
                                "selection": "over_1_5",
                                "decimal_odds": float(over_15["price"]),
                                "is_opening": 0,
                            }
                        )
                        rows.append(
                            {
                                "match_id": match_id,
                                "snapshot_time": snapshot,
                                "bookmaker": bookmaker_name,
                                "market": "over_1_5",
                                "selection": "under_1_5",
                                "decimal_odds": float(under_15["price"]),
                                "is_opening": 0,
                            }
                        )
                    if over and under:
                        rows.append(
                            {
                                "match_id": match_id,
                                "snapshot_time": snapshot,
                                "bookmaker": bookmaker_name,
                                "market": "over_2_5",
                                "selection": "over_2_5",
                                "decimal_odds": float(over["price"]),
                                "is_opening": 0,
                            }
                        )
                        rows.append(
                            {
                                "match_id": match_id,
                                "snapshot_time": snapshot,
                                "bookmaker": bookmaker_name,
                                "market": "over_2_5",
                                "selection": "under_2_5",
                                "decimal_odds": float(under["price"]),
                                "is_opening": 0,
                            }
                        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["match_id", "snapshot_time", "bookmaker", "market", "selection", "decimal_odds", "is_opening"])
    return frame.sort_values(["snapshot_time", "match_id", "market", "selection"]).reset_index(drop=True)



def fetch_competition_fixtures(
    competition: str,
    date_from: str,
    date_to: str,
    api_key_env: str = "FOOTBALL_DATA_API_KEY",
) -> pd.DataFrame:
    api_key = _get_env_api_key(api_key_env)
    settings = _competition_settings(competition)
    query = urlencode({"dateFrom": date_from, "dateTo": date_to})
    payload = _http_json(
        f"{FOOTBALL_DATA_BASE_URL}/{settings['football_data_code']}/matches?{query}",
        headers={"X-Auth-Token": api_key},
    )
    return football_data_matches_to_frame(payload, competition=competition)



def fetch_competition_odds(
    fixtures: pd.DataFrame,
    competition: str,
    api_key_env: str = "THE_ODDS_API_KEY",
    regions: str = "uk",
) -> pd.DataFrame:
    api_key = _get_env_api_key(api_key_env)
    settings = _competition_settings(competition)
    query = urlencode({"apiKey": api_key, "regions": regions, "markets": "h2h,totals,btts", "oddsFormat": "decimal", "dateFormat": "iso"})
    payload = _http_json(f"{ODDS_BASE_URL}/{settings['odds_api_sport']}/odds/?{query}")
    return odds_api_events_to_frame(payload, fixtures)



def fetch_epl_fixtures(date_from: str, date_to: str, api_key_env: str = "FOOTBALL_DATA_API_KEY") -> pd.DataFrame:
    return fetch_competition_fixtures("EPL", date_from, date_to, api_key_env=api_key_env)



def fetch_epl_odds(fixtures: pd.DataFrame, api_key_env: str = "THE_ODDS_API_KEY", regions: str = "uk") -> pd.DataFrame:
    return fetch_competition_odds(fixtures, "EPL", api_key_env=api_key_env, regions=regions)



def _upsert_by_key(existing: pd.DataFrame, incoming: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    if existing.empty:
        return incoming.copy()
    if incoming.empty:
        return existing.copy()
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.drop_duplicates(subset=key_columns, keep="last")
    return combined.reset_index(drop=True)



def update_matches_csv(path: str | Path, fixtures: pd.DataFrame) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = pd.read_csv(target)
    else:
        existing = pd.DataFrame(columns=fixtures.columns)
    merged = _upsert_by_key(existing, fixtures, ["match_id"])
    merged.to_csv(target, index=False)
    return target



def update_odds_csv(path: str | Path, odds: pd.DataFrame) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = pd.read_csv(target)
    else:
        existing = pd.DataFrame(columns=odds.columns)
    merged = _upsert_by_key(existing, odds, ["match_id", "snapshot_time", "bookmaker", "market", "selection"])
    merged.to_csv(target, index=False)
    return target
