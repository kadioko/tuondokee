from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MarketType = Literal["binary", "three_way"]
SelectionType = Literal["home", "draw", "away", "over_2_5", "under_2_5", "not_home"]


@dataclass(frozen=True)
class MatchSchema:
    match_id: str = "match_id"
    kickoff_time: str = "kickoff_time"
    league: str = "league"
    season: str = "season"
    home_team: str = "home_team"
    away_team: str = "away_team"
    home_goals: str = "home_goals"
    away_goals: str = "away_goals"


@dataclass(frozen=True)
class OddsSchema:
    match_id: str = "match_id"
    snapshot_time: str = "snapshot_time"
    bookmaker: str = "bookmaker"
    market: str = "market"
    selection: str = "selection"
    decimal_odds: str = "decimal_odds"
    is_opening: str = "is_opening"


@dataclass(frozen=True)
class BacktestBet:
    match_id: str
    market: str
    selection: str
    bookmaker: str
    placed_time: str
    odds: float
    fair_probability: float
    model_probability: float
    edge: float
    stake: float
    result: int
    pnl: float
    league: str
    season: str
