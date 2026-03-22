from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd


from valuebetting.backtest import build_prediction_frame, _kelly_fraction
from valuebetting.config import AppConfig, load_config
from valuebetting.data import load_matches, load_odds_snapshots, select_snapshot_before_kickoff
from valuebetting.features import build_feature_dataset
from valuebetting.modeling import load_model_artifacts, predict_probabilities


_SELECTION_LABELS: dict[str, str] = {
    "home": "Home Win",
    "draw": "Draw",
    "away": "Away Win",
    "over_0_5": "Over 0.5 Goals",
    "under_0_5": "Under 0.5 Goals",
    "over_1_5": "Over 1.5 Goals",
    "under_1_5": "Under 1.5 Goals",
    "over_2_5": "Over 2.5 Goals",
    "under_2_5": "Under 2.5 Goals",
    "btts_yes": "BTTS Yes",
    "btts_no": "BTTS No",
    "not_home": "Draw or Away",
}

_MINIMUM_REPORT_RELIABILITY_SCORE = 0.45


def _confidence_tier(edge: float, minimum_edge: float, report_rank_score: float, calibration_score: float) -> str:
    if edge < minimum_edge or report_rank_score < _MINIMUM_REPORT_RELIABILITY_SCORE:
        return "pass"
    if edge >= minimum_edge + 0.06 and report_rank_score >= 0.65 and calibration_score >= 0.60:
        return "strong"
    if edge >= minimum_edge + 0.03 and report_rank_score >= 0.52 and calibration_score >= 0.45:
        return "medium"
    return "watch"


def _stake_suggestion(edge: float, minimum_edge: float, flat_stake: float) -> str:
    if edge < minimum_edge:
        return "No bet"
    if edge >= minimum_edge + 0.06:
        return f"Full stake ({flat_stake:.0f} units)"
    if edge >= minimum_edge + 0.03:
        return f"Half stake ({flat_stake * 0.5:.1f} units)"
    return f"Small stake ({flat_stake * 0.25:.2f} units)"


def _format_kickoff_time(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    return timestamp.strftime("%a %d %b %H:%M")


def _enrich_best_prices(selection_frame: pd.DataFrame) -> pd.DataFrame:
    if selection_frame.empty:
        return selection_frame
    grouped = selection_frame.groupby(["match_id", "market", "selection"], as_index=False)
    summary = grouped.agg(
        best_decimal_odds=("decimal_odds", "max"),
        market_average_odds=("decimal_odds", "mean"),
        bookmaker_count=("bookmaker", "nunique"),
    )
    ranked = selection_frame.sort_values(
        ["match_id", "market", "selection", "decimal_odds", "edge"],
        ascending=[True, True, True, False, False],
    ).copy()
    ranked["bookmaker_rank"] = ranked.groupby(["match_id", "market", "selection"]).cumcount() + 1
    best = ranked.loc[ranked["bookmaker_rank"] == 1].copy()
    second_best = ranked.loc[ranked["bookmaker_rank"] == 2, ["match_id", "market", "selection", "decimal_odds"]].rename(
        columns={"decimal_odds": "second_best_odds"}
    )
    best = best.merge(summary, on=["match_id", "market", "selection"], how="left")
    best = best.merge(second_best, on=["match_id", "market", "selection"], how="left")
    best["decimal_odds"] = best["best_decimal_odds"]
    return best


def _stake_details(row: pd.Series, config: AppConfig) -> tuple[str, float, float, float]:
    bankroll_cap_pct = float(config.betting.max_stake_pct)
    raw_kelly_pct = _kelly_fraction(float(row["model_probability"]), float(row["decimal_odds"]))
    safe_kelly_pct = min(raw_kelly_pct * float(config.betting.fractional_kelly_fraction), bankroll_cap_pct)
    aggressive_kelly_pct = min(raw_kelly_pct, bankroll_cap_pct)
    safe_units = safe_kelly_pct * float(config.betting.initial_bankroll)
    if config.betting.staking == "flat":
        display_text = _stake_suggestion(float(row["edge"]), config.betting.minimum_edge, config.betting.flat_stake)
    else:
        display_text = f"Safe Kelly ({safe_units:.2f} units)"
    return display_text, safe_units, safe_kelly_pct, aggressive_kelly_pct


def _load_oos_reliability(artifacts_dir: str | Path) -> dict[str, float]:
    path = Path(artifacts_dir)
    calibration_path = path / "calibration_diagnostics_oos.csv"
    summary_path = path / "backtest_summary.json"
    reliability = {
        "oos_calibration_gap": 0.10,
        "oos_yield": 0.0,
        "oos_hit_rate": 0.0,
    }
    if calibration_path.exists():
        calibration = pd.read_csv(calibration_path)
        if not calibration.empty and "calibration_gap" in calibration.columns and "sample_count" in calibration.columns:
            weights = calibration["sample_count"].fillna(0.0).astype(float)
            gaps = calibration["calibration_gap"].fillna(0.0).abs().astype(float)
            total = float(weights.sum())
            if total > 0:
                reliability["oos_calibration_gap"] = float((gaps * weights).sum() / total)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        reliability["oos_yield"] = float(summary.get("yield", 0.0))
        reliability["oos_hit_rate"] = float(summary.get("hit_rate", 0.0))
    return reliability


def _apply_reliability_scores(selection_frame: pd.DataFrame) -> pd.DataFrame:
    if selection_frame.empty:
        return selection_frame
    frame = selection_frame.copy()
    frame["agreement_count"] = frame.groupby("match_id")["market"].transform("count")
    frame["agreement_score"] = np.clip((frame["agreement_count"].astype(float) - 1.0) / 2.0, 0.0, 1.0)
    frame["odds_quality_score"] = np.clip(
        (frame["decimal_odds"].astype(float) - frame["market_average_odds"].astype(float).fillna(frame["decimal_odds"].astype(float)))
        / frame["decimal_odds"].astype(float).replace(0.0, np.nan),
        0.0,
        1.0,
    ).fillna(0.0)
    frame["calibration_score"] = np.clip(1.0 - (frame["oos_calibration_gap"].astype(float) / 0.20), 0.0, 1.0)
    frame["yield_score"] = np.clip((frame["oos_yield"].astype(float) + 0.10) / 0.20, 0.0, 1.0)
    frame["edge_score"] = np.clip(frame["edge"].astype(float) / 0.15, 0.0, 1.0)
    frame["report_rank_score"] = (
        0.35 * frame["edge_score"]
        + 0.20 * frame["calibration_score"]
        + 0.15 * frame["yield_score"]
        + 0.15 * frame["agreement_score"]
        + 0.15 * frame["odds_quality_score"]
    )
    return frame


def _get_probability_column_map(classes: list[str]) -> dict[str, str]:
    return {class_name: f"model_prob_{class_name}" for class_name in classes}



def build_scored_matches(config: AppConfig) -> pd.DataFrame:
    model, metadata = load_model_artifacts(config.paths.artifacts_dir)
    feature_artifacts = build_feature_dataset(
        matches_csv=config.paths.matches_csv,
        odds_csv=config.paths.odds_csv,
        snapshot_minutes_before_kickoff=config.features.snapshot_minutes_before_kickoff,
        form_windows=config.features.form_windows,
        elo_k=config.features.elo_k,
        league_filters=config.features.league_filters,
        include_incomplete_matches=True,
    )
    dataset = feature_artifacts.dataset.copy()
    X = dataset[metadata["feature_columns"]]
    probabilities = predict_probabilities(model, X)
    scored = dataset[["match_id", "kickoff_time", "league", "season", "home_team", "away_team", "is_completed"]].copy()
    for idx, class_name in enumerate(metadata["classes"]):
        scored[f"model_prob_{class_name}"] = probabilities[:, idx]
    return scored



def _build_selection_level_frame(scored_matches: pd.DataFrame, selected_odds: pd.DataFrame, classes: list[str], binary_market: str | None) -> pd.DataFrame:
    probability_columns = _get_probability_column_map(classes)
    predicted = np.column_stack([scored_matches[probability_columns[class_name]].to_numpy() for class_name in classes])
    dataset_columns = [
        column
        for column in ["match_id", "kickoff_time", "league", "season", "home_team", "away_team", "home_goals", "away_goals"]
        if column in scored_matches.columns
    ]
    dataset = scored_matches[dataset_columns].copy()
    return build_prediction_frame(
        dataset=dataset,
        predicted_probabilities=predicted,
        classes=classes,
        selected_odds=selected_odds,
        bookmaker=None,
        binary_market=binary_market,
    )



def _score_one_model(config: AppConfig) -> pd.DataFrame:
    """Score upcoming fixtures with one model and return a selection-level frame."""
    scored_matches = build_scored_matches(config)
    upcoming = scored_matches.loc[scored_matches["is_completed"] == 0].copy()
    if upcoming.empty:
        return pd.DataFrame()

    matches = load_matches(config.paths.matches_csv)
    odds = load_odds_snapshots(config.paths.odds_csv)
    selected_odds = select_snapshot_before_kickoff(matches, odds, config.features.snapshot_minutes_before_kickoff)
    selected_odds = selected_odds.loc[selected_odds["match_id"].isin(upcoming["match_id"])].copy()
    if config.betting.bookmaker is not None:
        selected_odds = selected_odds.loc[selected_odds["bookmaker"] == config.betting.bookmaker].copy()
    if selected_odds.empty:
        return pd.DataFrame()

    model, metadata = load_model_artifacts(config.paths.artifacts_dir)
    del model
    selection_frame = _build_selection_level_frame(
        scored_matches=upcoming,
        selected_odds=selected_odds,
        classes=list(metadata["classes"]),
        binary_market=metadata.get("binary_market"),
    )
    if selection_frame.empty:
        return selection_frame
    reliability = _load_oos_reliability(config.paths.artifacts_dir)
    for key, value in reliability.items():
        selection_frame[key] = value
    selection_frame["source_artifacts_dir"] = str(config.paths.artifacts_dir)
    return selection_frame



def _friendly_selection(selection: str) -> str:
    return _SELECTION_LABELS.get(selection, selection)


def _model_opinion(row: pd.Series) -> str:
    if str(row.get("recommendation", "pass")) == "bet":
        return f"VALUE BET [{str(row['strength_label']).upper()}]"
    if float(row.get("edge", 0.0)) > 0.0:
        return f"WATCHLIST [{str(row['strength_label']).upper()}]"
    return "PASS"


def _build_html_report(report_frame: pd.DataFrame, report_text: str) -> str:
    escaped_text = report_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if report_frame.empty:
        table_html = "<p>No report rows available.</p>"
    else:
        html_frame = report_frame.copy()
        html_frame = html_frame.assign(
            matchup=html_frame["home_team"].astype(str) + " vs " + html_frame["away_team"].astype(str),
            kickoff_display=html_frame["kickoff_time"].map(_format_kickoff_time),
            selection_label=html_frame["selection"].map(_friendly_selection),
            model_opinion=html_frame.apply(_model_opinion, axis=1),
            edge_pct=(html_frame["edge"].astype(float) * 100.0).round(2),
            reliability_score=(html_frame["report_rank_score"].astype(float) * 100.0).round(1),
            model_probability_pct=(html_frame["model_probability"].astype(float) * 100.0).round(1),
            fair_probability_pct=(html_frame["fair_prob"].astype(float) * 100.0).round(1),
        )
        html_frame = html_frame[
            [
                "league",
                "matchup",
                "kickoff_display",
                "market",
                "selection_label",
                "model_opinion",
                "bookmaker",
                "decimal_odds",
                "edge_pct",
                "reliability_score",
                "strength_label",
                "model_probability_pct",
                "fair_probability_pct",
                "stake_suggestion",
            ]
        ]
        table_html = html_frame.to_html(index=False, border=0, classes="report-table")
    return """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Value Betting Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; color: #1f2937; background: #f8fafc; }
    h1 { margin-bottom: 8px; }
    .summary { white-space: pre-wrap; background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; }
    .report-table { border-collapse: collapse; width: 100%; margin-top: 20px; background: white; }
    .report-table th, .report-table td { border: 1px solid #e5e7eb; padding: 10px; text-align: left; }
    .report-table th { background: #eef2ff; }
    .note { color: #6b7280; margin-top: 8px; }
  </style>
</head>
<body>
  <h1>Shareable Value Report</h1>
  <p class=\"note\">This HTML version is intended for easier sharing outside the terminal.</p>
  <div class=\"summary\">""" + escaped_text + """</div>
  """ + table_html + """
</body>
</html>
"""


def _report_title(report_rows: pd.DataFrame, fallback_frame: pd.DataFrame) -> str:
    base = report_rows if not report_rows.empty else fallback_frame
    if base.empty:
        return "MODEL VALUE REPORT"
    leagues = sorted({str(value) for value in base["league"].dropna().tolist()})
    league_label = leagues[0] if len(leagues) == 1 else "MULTI-LEAGUE"
    kickoff_min = pd.Timestamp(base["kickoff_time"].min())
    kickoff_max = pd.Timestamp(base["kickoff_time"].max())
    if kickoff_min.normalize() == kickoff_max.normalize():
        period_label = kickoff_min.strftime("%d %b %Y")
    else:
        period_label = f"{kickoff_min.strftime('%d %b')} to {kickoff_max.strftime('%d %b %Y')}"
    return f"{league_label} VALUE REPORT — {period_label}"



def generate_value_report(
    config: AppConfig,
    top_n: int = 10,
    extra_config_paths: list[str | Path] | None = None,
) -> tuple[pd.DataFrame, str]:
    frames: list[pd.DataFrame] = []

    # Score with the primary model (1x2)
    primary_frame = _score_one_model(config)
    if not primary_frame.empty:
        frames.append(primary_frame)

    # Score with any extra models (e.g. over_2_5)
    for extra_path in extra_config_paths or []:
        extra_config = load_config(extra_path)
        extra_frame = _score_one_model(extra_config)
        if not extra_frame.empty:
            frames.append(extra_frame)

    if not frames:
        empty = pd.DataFrame()
        return empty, "No upcoming fixtures could be scored. Check that data has been fetched and models have been trained."

    combined = pd.concat(frames, ignore_index=True)
    combined = _enrich_best_prices(combined)
    combined = _apply_reliability_scores(combined)
    combined = combined.sort_values(["report_rank_score", "edge", "model_probability"], ascending=[False, False, False]).reset_index(drop=True)
    combined["recommendation"] = np.where(
        (combined["edge"] >= config.betting.minimum_edge) & (combined["report_rank_score"] >= _MINIMUM_REPORT_RELIABILITY_SCORE),
        "bet",
        "pass",
    )
    combined["confidence_label"] = pd.cut(
        combined["edge"],
        bins=[-np.inf, 0.0, config.betting.minimum_edge, config.betting.minimum_edge + 0.03, np.inf],
        labels=["negative", "watchlist", "small_edge", "strong_edge"],
    ).astype(str)
    combined["strength_label"] = combined.apply(
        lambda row: _confidence_tier(
            float(row["edge"]),
            config.betting.minimum_edge,
            float(row["report_rank_score"]),
            float(row["calibration_score"]),
        ),
        axis=1,
    )
    stake_details = combined.apply(lambda row: _stake_details(row, config), axis=1, result_type="expand")
    combined[["stake_suggestion", "safe_stake_units", "safe_kelly_pct", "aggressive_kelly_pct"]] = stake_details

    combined["market_category"] = combined["market"]
    all_views = combined.drop_duplicates(subset=["match_id", "market"], keep="first").reset_index(drop=True)
    bets_only = all_views.loc[all_views["recommendation"] == "bet"].copy()
    summary_rows = bets_only.head(top_n).copy() if not bets_only.empty else all_views.head(top_n).copy()
    title = _report_title(summary_rows, all_views)

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"  {title}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("How to read this report:")
    lines.append("  - By default, this report shows only bets that cleared the edge threshold.")
    lines.append("  - 'Value bet' means our model thinks the true chance is")
    lines.append("    higher than what the bookmaker's odds imply — the price")
    lines.append("    is generous enough to bet on.")
    lines.append("  - Strength labels help you separate stronger edges from marginal ones.")
    lines.append("  - 'Edge' is how much extra probability our model sees")
    lines.append("    compared to the bookmaker (bigger = better value).")
    lines.append("  - 'WATCH' means the bet is still positive-value, but only by a smaller margin.")
    lines.append("  - A bet must clear both the edge threshold and a minimum reliability score.")
    lines.append("  - No model can produce sure winners; this report only ranks better-supported opinions.")
    lines.append("")
    lines.append("Best bets summary:")
    if summary_rows.empty:
        lines.append("  - No bets cleared the current edge threshold. Consider this a watchlist slate.")
    else:
        for _, row in summary_rows.iterrows():
            sel_label = _friendly_selection(str(row["selection"]))
            lines.append(
                f"  - {row['home_team']} vs {row['away_team']}: {sel_label} @ {float(row['decimal_odds']):.2f} "
                f"[{str(row['strength_label']).upper()}]"
            )
    lines.append("")
    lines.append("-" * 60)
    lines.append("All model views:")

    # Group opinions by match for cleaner output
    match_groups = all_views.groupby(["home_team", "away_team"], sort=False)
    for (home, away), group in match_groups:
        lines.append(f"\n  {home}  vs  {away}")
        lines.append(f"  {'─' * 40}")
        for _, row in group.iterrows():
            edge_pct = float(row["edge"]) * 100.0
            model_pct = float(row["model_probability"]) * 100.0
            fair_pct = float(row["fair_prob"]) * 100.0
            sel_label = _friendly_selection(str(row["selection"]))
            odds_str = f"{float(row['decimal_odds']):.2f}"
            kickoff_str = _format_kickoff_time(row["kickoff_time"])
            bookmaker = str(row["bookmaker"])
            verdict = _model_opinion(row)

            lines.append(f"    Kickoff: {kickoff_str}")
            lines.append(f"    {str(row['market'])}: {sel_label} @ {odds_str} with {bookmaker}  →  {verdict}")
            lines.append(f"      Our model: {model_pct:.1f}%  |  Market fair: {fair_pct:.1f}%  |  Edge: {edge_pct:+.1f}%")
            lines.append(
                f"      Reliability score: {float(row['report_rank_score']) * 100.0:.1f}/100  |  Agreement: {int(row['agreement_count'])} market(s)"
            )
            lines.append(f"      Model thinks: {verdict}")
            lines.append(f"      Suggested stake: {row['stake_suggestion']}")
            lines.append(
                f"      Safe Kelly: {float(row['safe_stake_units']):.2f} units ({float(row['safe_kelly_pct']) * 100.0:.2f}% bankroll)"
            )
            lines.append(
                f"      Aggressive Kelly cap: {float(row['aggressive_kelly_pct']) * 100.0:.2f}% bankroll"
            )
            if not pd.isna(row.get('second_best_odds')):
                lines.append(
                    f"      Market comparison: best {float(row['decimal_odds']):.2f} | second-best {float(row['second_best_odds']):.2f} | average {float(row['market_average_odds']):.2f}"
                )
            else:
                lines.append(
                    f"      Market comparison: best {float(row['decimal_odds']):.2f} | average {float(row['market_average_odds']):.2f}"
                )

    lines.append("")
    lines.append("-" * 60)
    lines.append("Disclaimer: These are model-generated opinions, not financial")
    lines.append("advice. Always bet responsibly and within your means.")
    lines.append("=" * 60)

    return all_views, "\n".join(lines)



def write_value_report(output_dir: str | Path, report_frame: pd.DataFrame, report_text: str) -> tuple[Path, Path, Path, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / "value_opinions.csv"
    best_bets_path = target_dir / "best_bets.csv"
    txt_path = target_dir / "value_opinions.txt"
    html_path = target_dir / "value_opinions.html"
    report_frame.to_csv(csv_path, index=False)
    best_bets = report_frame.copy()
    if not best_bets.empty:
        best_bets = best_bets.assign(
            matchup=best_bets["home_team"].astype(str) + " vs " + best_bets["away_team"].astype(str),
            kickoff_display=best_bets["kickoff_time"].map(_format_kickoff_time),
            selection_label=best_bets["selection"].map(_friendly_selection),
            model_opinion=best_bets.apply(_model_opinion, axis=1),
            edge_pct=(best_bets["edge"].astype(float) * 100.0).round(2),
            safe_kelly_bankroll_pct=(best_bets["safe_kelly_pct"].astype(float) * 100.0).round(2),
            aggressive_kelly_bankroll_pct=(best_bets["aggressive_kelly_pct"].astype(float) * 100.0).round(2),
        )
        best_bets = best_bets.loc[best_bets["recommendation"] == "bet"].copy()
        export_columns = [
            "match_id",
            "league",
            "season",
            "matchup",
            "kickoff_display",
            "market",
            "selection_label",
            "model_opinion",
            "bookmaker",
            "decimal_odds",
            "second_best_odds",
            "market_average_odds",
            "model_probability",
            "fair_prob",
            "edge_pct",
            "strength_label",
            "stake_suggestion",
            "safe_stake_units",
            "safe_kelly_bankroll_pct",
            "aggressive_kelly_bankroll_pct",
        ]
        best_bets = best_bets[export_columns]
    best_bets.to_csv(best_bets_path, index=False)
    txt_path.write_text(report_text, encoding="utf-8")
    html_path.write_text(_build_html_report(report_frame, report_text), encoding="utf-8")
    return csv_path, best_bets_path, txt_path, html_path
