import sys
sys.path.insert(0, 'd:/Github/MFS')

from TA.LW_CheckHFH import calHFH
import pandas as pd
import numpy as np

print("=== Finding the breakout candidate ===")

df = pd.read_csv('d:/Github/MFS/../SData/P_YFdata/L/P_0001.HK.csv', index_col=0)
print(f"Loaded {len(df)} rows")

# Calculate all required EMAs
df['EMA10'] = df['Close'].ewm(span=10, min_periods=10, adjust=False).mean()
df['EMA22'] = df['Close'].ewm(span=22, min_periods=22, adjust=False).mean()
df['EMA50'] = df['Close'].ewm(span=50, min_periods=50, adjust=False).mean()
df['EMA100'] = df['Close'].ewm(span=100, min_periods=100, adjust=False).mean()
df['EMA250'] = df['Close'].ewm(span=250, min_periods=250, adjust=False).mean()

# Get result
result = calHFH(df.copy())

highs = result['High'].values
lows = result['Low'].values
closes = result['Close'].values
opens = result['Open'].values
volumes = result['Volume'].values

bodies = np.abs(closes - opens)
candle_ranges = highs - lows
body_pct = np.where(candle_ranges > 0, bodies / candle_ranges, np.nan)
is_bullish = closes > opens
is_strong_bullish = is_bullish & (body_pct >= 0.5)
vol_ma20 = pd.Series(volumes).rolling(window=20, min_periods=1).mean().values

uptrends = (
    (result['EMA22'].values > result['EMA50'].values) &
    (result['EMA50'].values > result['EMA100'].values) &
    (result['EMA100'].values > result['EMA250'].values)
)

# Calculate flat_starts
max_flat_pct = 0.12
min_flat_length = 4

left = 0
flat_starts = np.zeros(len(result), dtype=int)
for right in range(len(result)):
    while left < right:
        w_high = np.max(highs[left:right+1])
        w_low = np.min(lows[left:right+1])
        if w_low > 0 and (w_high - w_low) / w_low <= max_flat_pct:
            break
        left += 1
    flat_starts[right] = left

# Find the one candidate that passed breakout
for i in range(250, min(1000, len(result))):
    curr_start = flat_starts[i]
    flat_len = i - curr_start
    
    if flat_len < min_flat_length:
        continue
    
    # Pre-high
    pre_high = 0
    if curr_start > 0 and curr_start < len(result):
        for j in range(curr_start - 1, max(0, curr_start - 10), -1):
            if is_strong_bullish[j]:
                pre_high += 1
            else:
                break
    if pre_high < 2:
        continue
    
    if not uptrends[i]:
        continue
    
    flat_high = np.max(highs[curr_start:i])
    if closes[i] <= flat_high:
        continue
    
    # Found breakout! Let's analyze it
    print(f"\n*** Breakout candidate at {result.index[i]} (index {i}) ***")
    print(f"  Flat zone: {result.index[curr_start]} to {result.index[i]}, length={flat_len}")
    print(f"  Pre-high: {pre_high}")
    print(f"  Uptrend: {uptrends[i]}")
    print(f"  Breakout: close={closes[i]:.2f} > flat_high={flat_high:.2f}")
    
    # Check subsequent conditions
    daily_range = highs[i] - lows[i]
    close_strength = (closes[i] - lows[i]) / daily_range if daily_range > 0 else 0
    print(f"  Close strength: {close_strength:.3f} (need >= 0.5)")
    
    upper_wick = highs[i] - closes[i]
    upper_wick_ratio = upper_wick / daily_range if daily_range > 0 else 0
    print(f"  Upper wick ratio: {upper_wick_ratio:.3f} (need <= 0.35)")
    
    vol_ratio = volumes[i] / vol_ma20[i]
    print(f"  Volume ratio: {vol_ratio:.3f} (need >= 1.2)")
    
    # Volume trend
    if i >= 3:
        vol_trend = volumes[i-2:i+1]
        vol_trend_ma = np.mean(volumes[max(0, i-5):i])
        print(f"  Vol trend mean: {np.mean(vol_trend):.0f}")
        print(f"  Vol trend MA: {vol_trend_ma:.0f}")
        print(f"  Vol trend check: {np.mean(vol_trend):.0f} >= {vol_trend_ma * 0.5:.0f}? {np.mean(vol_trend) >= vol_trend_ma * 0.5}")
    
    # Body deviation check
    flat_bodies = bodies[curr_start:i]
    flat_ranges = candle_ranges[curr_start:i]
    flat_body_pcts = body_pct[curr_start:i]
    valid_mask = ~np.isnan(flat_body_pcts) & (flat_ranges > 0)
    
    if np.sum(valid_mask) >= min_flat_length:
        avg_body = np.mean(flat_bodies[valid_mask])
        body_devs = np.abs(flat_bodies[valid_mask] - avg_body) / avg_body
        print(f"  Body deviation max: {np.max(body_devs):.3f} (need <= 0.3)")
        print(f"  Min body pct: {np.min(flat_body_pcts[valid_mask]):.3f} (need >= 0.3)")

print("\n=== Done ===")