import sys
sys.path.insert(0, 'd:/Github/MFS')

from TA.LW_CheckHFH import calHFH
import pandas as pd
import numpy as np

print("=== Quick HFH Test ===")

df = pd.read_csv('d:/Github/MFS/../Sdata/P_YFdata/L/P_0001.HK.csv', index_col=0)
print(f"Loaded {len(df)} rows")

# Calculate all required EMAs
df['EMA10'] = df['Close'].ewm(span=10, min_periods=10, adjust=False).mean()
df['EMA22'] = df['Close'].ewm(span=22, min_periods=22, adjust=False).mean()
df['EMA50'] = df['Close'].ewm(span=50, min_periods=50, adjust=False).mean()
df['EMA100'] = df['Close'].ewm(span=100, min_periods=100, adjust=False).mean()
df['EMA250'] = df['Close'].ewm(span=250, min_periods=250, adjust=False).mean()

# Call calHFH with new defaults
result = calHFH(df.copy())
hfhs = result['HFH'].sum()
print(f"\nTotal HFH signals: {hfhs}")

if hfhs > 0:
    signals = result[result['HFH'] == True]
    print(f"\nFirst 5 signals:")
    for i, (date, row) in enumerate(signals.head(5).iterrows()):
        print(f"  {i+1}. {date}: FlatCount={row['FlatCount']}, PreHighCount={row['PreHighCount']}, Quality={row['BreakoutQuality']:.1f}")

print("\n=== Done ===")