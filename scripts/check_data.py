import pandas as pd

df = pd.read_csv("data/sample/matches.csv")
print(f"Total: {len(df)}")
print(f"With scores: {df['home_goals'].notna().sum()}")
print(f"Without scores: {df['home_goals'].isna().sum()}")
print("\nFirst 3 rows:")
print(df.head(3).to_string())
print("\nLast 3 rows:")
print(df.tail(3).to_string())
print("\nUnique teams:", sorted(df["home_team"].unique()))
