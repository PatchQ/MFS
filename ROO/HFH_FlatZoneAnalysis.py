import sys
sys.path.insert(0, 'd:/Github/MFS')

from TA.LW_CheckHFH import calHFH
import pandas as pd
import numpy as np

print("=== HFH Detailed Diagnosis ===")

df = pd.read_csv('d:/Github/MFS/../SData/P_YFdata/L/P_0001.HK.csv', index_col=0)
print(f"Loaded {len(df)} rows, date range: {df.index[0]} to {df.index[-1]}")

# Calculate all required EMAs
df['EMA10'] = df['Close'].ewm(span=10, min_periods=10, adjust=False).mean()
df['EMA22'] = df['Close'].ewm(span=22, min_periods=22, adjust=False).mean()
df['EMA50'] = df['Close'].ewm(span=50, min_periods=50, adjust=False).mean()
df['EMA100'] = df['Close'].ewm(span=100, min_periods=100, adjust=False).mean()
df['EMA250'] = df['Close'].ewm(span=250, min_periods=250, adjust=False).mean()

# Run calHFH
result = calHFH(df.copy())
print(f"\nTotal HFH signals: {result['HFH'].sum()}")

# Check each filtering stage
highs = result['High'].values
lows = result['Low'].values
closes = result['Close'].values
opens = result['Open'].values
volumes = result['Volume'].values

bodies = np.abs(closes - opens)
candle_ranges = highs - lows
body_pct = np.where(candle_ranges > 0, bodies / candle_ranges, np.nan)
is_bullish = closes > opens
vol_ma20 = pd.Series(volumes).rolling(window=20, min_periods=1).mean().values

# Uptrend
uptrend_condition = (
    (result['EMA22'].values > result['EMA50'].values) &
    (result['EMA50'].values > result['EMA100'].values) &
    (result['EMA100'].values > result['EMA250'].values)
)

# Check flat zone detection manually
max_flat_pct = 0.12
min_flat_length = 4

print("\n=== Checking flat zone detection ===")
flat_starts = np.zeros(len(result), dtype=int)
left = 0
for right in range(len(result)):
    while left < right:
        w_high = np.max(highs[left:right+1])
        w_low = np.min(lows[left:right+1])
        if w_low > 0 and (w_high - w_low) / w_low <= max_flat_pct:
            break
        left += 1
    flat_starts[right] = left

# Find flat zones that are long enough
flat_zones = []
for i in range(100, min(300, len(result))):
    curr_start = flat_starts[i]
    flat_len = i - curr_start
    if flat_len >= min_flat_length:
        flat_zones.append((i, curr_start, flat_len, result.index[i]))

print(f"Found {len(flat_zones)} flat zones with length >= {min_flat_length}")
if flat_zones:
    print("\nFirst 5 flat zones:")
    for idx, (end, start, length, date) in enumerate(flat_zones[:5]):
        flat_high = np.max(highs[start:end])
        flat_low = np.min(lows[start:end])
        print(f"  {idx+1}. End: {date}, Start: {result.index[start]}, Length: {length}, Range: {flat_low:.2f}-{flat_high:.2f}")

# Check why flat zones don't become signals
print("\n=== Analyzing first valid flat zone ===")
if flat_zones:
    i, curr_start, flat_len, date = flat_zones[0]
    print(f"Date: {date}, Index: {i}")
    print(f"  Flat zone: {result.index[curr_start]} to {date}, Length: {flat_len}")
    
    # Check conditions
    flat_bodies = bodies[curr_start:i]
    flat_ranges = candle_ranges[curr_start:i]
    flat_body_pcts = body_pct[curr_start:i]
    
    print(f"  Bodies: {flat_bodies}")
    print(f"  Body%: {flat_body_pcts}")
    
    # Strong bullish count before flat
    is_strong_bullish = is_bullish & (body_pct >= 0.5)
    pre_count = 0
    for j in range(curr_start - 1, max(0, curr_start - 10), -1):
        if is_strong_bullish[j]:
            pre_count += 1
        else:
            break
    print(f"  Pre-high count: {pre_count}")
    print(f"  Min required: 2")
    
    # Check uptrend
    print(f"  Uptrend: {uptrend_condition[i]}")
    
    # Check breakout
    flat_high = np.max(highs[curr_start:i])
    print(f"  Flat high: {flat_high}, Close: {closes[i]}, Breakout: {closes[i] > flat_high}")
    
    # Check close strength
    daily_range = highs[i] - lows[i]
    close_strength = (closes[i] - lows[i]) / daily_range if daily_range > 0 else 0
    print(f"  Close strength: {close_strength:.3f}, Min: 0.5")
    
    # Check upper wick
    upper_wick = highs[i] - closes[i]
    upper_wick_ratio = upper_wick / daily_range if daily_range > 0 else 0
    print(f"  Upper wick ratio: {upper_wick_ratio:.3f}, Max: 0.35")
    
    # Check volume
    vol_ratio = volumes[i] / vol_ma20[i]
    print(f"  Volume ratio: {vol_ratio:.3f}, Min: 1.2")

print("\n=== Done ===")