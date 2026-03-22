"""Remove toy M1-M10 rows from matches and odds CSVs, keeping only real FD_PL_ data."""
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[1]
matches_path = root / "data" / "sample" / "matches.csv"
odds_path = root / "data" / "sample" / "odds_snapshots.csv"

# Clean matches
matches = pd.read_csv(matches_path)
before = len(matches)
matches = matches[matches["match_id"].str.startswith("FD_PL_")].reset_index(drop=True)
after = len(matches)
matches.to_csv(matches_path, index=False)
print(f"Matches: {before} -> {after} rows (removed {before - after} toy rows)")

# Clean odds
odds = pd.read_csv(odds_path)
before_odds = len(odds)
odds = odds[odds["match_id"].str.startswith("FD_PL_")].reset_index(drop=True)
after_odds = len(odds)
odds.to_csv(odds_path, index=False)
print(f"Odds: {before_odds} -> {after_odds} rows (removed {before_odds - after_odds} toy rows)")

# Summary
completed = matches["home_goals"].notna().sum()
upcoming = matches["home_goals"].isna().sum()
unique_teams = sorted(set(matches["home_team"].unique()) | set(matches["away_team"].unique()))
print(f"\nCompleted: {completed}, Upcoming: {upcoming}")
print(f"Unique teams: {len(unique_teams)}")
for t in unique_teams:
    print(f"  {t}")
