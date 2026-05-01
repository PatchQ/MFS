import sys
sys.path.insert(0, 'd:/Github/MFS')

# Direct import
from TA.LW_CheckHFH import calHFH
import pandas as pd
import numpy as np

print("Starting fresh test...")

df = pd.read_csv('d:/Github/MFS/../SData/P_YFdata/L/P_0001.HK.csv', index_col=0)
print(f"Loaded {len(df)} rows")

# Calculate all required EMAs
df['EMA10'] = df['Close'].ewm(span=10, min_periods=10, adjust=False).mean()
df['EMA22'] = df['Close'].ewm(span=22, min_periods=22, adjust=False).mean()
df['EMA50'] = df['Close'].ewm(span=50, min_periods=50, adjust=False).mean()
df['EMA100'] = df['Close'].ewm(span=100, min_periods=100, adjust=False).mean()
df['EMA250'] = df['Close'].ewm(span=250, min_periods=250, adjust=False).mean()

# Test with defaults (now next_day_confirm=False)
result = calHFH(df.copy())
print(f'HFH with defaults: {result["HFH"].sum()}')

# Check pre_high_count distribution
if 'PreHighCount' in result.columns:
    print(f"PreHighCount distribution: {result['PreHighCount'].value_counts().to_dict()}")

# Check FlatCount
if 'FlatCount' in result.columns:
    print(f"FlatCount distribution: {result['FlatCount'].value_counts().to_dict()}")

# Check if there are any signals at all
if result['HFH'].sum() == 0:
    print("No HFH signals detected. Debugging...")
    
    # Check EMA alignment
    uptrend = (df['EMA22'] > df['EMA50']) & (df['EMA50'] > df['EMA100']) & (df['EMA100'] > df['EMA250'])
    print(f"Uptrend days (EMA22>EMA50>EMA100>EMA250): {uptrend.sum()}")
    
    # Check strong bullish
    bodies = np.abs(df['Close'] - df['Open'])
    ranges = df['High'] - df['Low']
    body_pct = np.where(ranges > 0, bodies / ranges, 0)
    strong_bullish = (df['Close'] > df['Open']) & (body_pct >= 0.5)
    print(f"Strong bullish days: {strong_bullish.sum()}")
    
    # Check flat zones
    max_flat_pct = 0.12
    for i in range(100, min(200, len(df))):
        left = max(0, i-10)
        w_high = df['High'].iloc[left:i+1].max()
        w_low = df['Low'].iloc[left:i+1].min()
        if w_low > 0:
            flat_pct = (w_high - w_low) / w_low
            if flat_pct <= max_flat_pct:
                print(f"Found flat zone at index {i}: {flat_pct:.4f}")
                break

print("Done!")