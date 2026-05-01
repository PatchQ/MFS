import sys
sys.path.insert(0, 'd:/Github/MFS')

from TA.LW_CheckHFH import calHFH
import pandas as pd
import numpy as np

print("=== HFH Signal Missing Root Cause ===")

df = pd.read_csv('d:/Github/MFS/../SData/P_YFdata/L/P_0001.HK.csv', index_col=0)
print(f"Loaded {len(df)} rows")

# Calculate all required EMAs
df['EMA10'] = df['Close'].ewm(span=10, min_periods=10, adjust=False).mean()
df['EMA22'] = df['Close'].ewm(span=22, min_periods=22, adjust=False).mean()
df['EMA50'] = df['Close'].ewm(span=50, min_periods=50, adjust=False).mean()
df['EMA100'] = df['Close'].ewm(span=100, min_periods=100, adjust=False).mean()
df['EMA250'] = df['Close'].ewm(span=250, min_periods=250, adjust=False).mean()

# Get result from calHFH
result = calHFH(df.copy())

# Now manually trace through all conditions for each row
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

# Uptrend
uptrends = (
    (result['EMA22'].values > result['EMA50'].values) &
    (result['EMA50'].values > result['EMA100'].values) &
    (result['EMA100'].values > result['EMA250'].values)
)

# Calculate flat_starts manually
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

# For each condition, count how many rows pass
condition_stats = {
    'total_rows': len(result),
    'flat_len >= 4': 0,
    'pre_high >= 2': 0,
    'uptrend': 0,
    'close > flat_high': 0,
    'close_strength >= 0.5': 0,
    'upper_wick <= 0.35': 0,
    'volume_ratio >= 1.2': 0,
    'volume_trend_ok': 0,
    'flat_body_dev <= 0.3': 0,
    'flat_min_body_pct >= 0.3': 0,
}

fail_reasons = {
    'flat_len': 0,
    'pre_high': 0,
    'uptrend': 0,
    'breakout': 0,
    'close_strength': 0,
    'upper_wick': 0,
    'volume': 0,
    'vol_trend': 0,
    'body_dev': 0,
    'body_pct': 0,
}

# Check each potential flat zone
for i in range(250, min(1000, len(result))):
    curr_start = flat_starts[i]
    flat_len = i - curr_start
    
    # Condition 1: Flat length
    if flat_len < min_flat_length:
        continue
    condition_stats['flat_len >= 4'] += 1
    
    # Condition 2: Pre-high count
    if curr_start > 0 and curr_start < len(result):
        pre_high = 0
        for j in range(curr_start - 1, max(0, curr_start - 10), -1):
            if is_strong_bullish[j]:
                pre_high += 1
            else:
                break
        if pre_high < 2:
            fail_reasons['pre_high'] += 1
            continue
    else:
        continue
    condition_stats['pre_high >= 2'] += 1
    
    # Condition 3: Uptrend
    if not uptrends[i]:
        fail_reasons['uptrend'] += 1
        continue
    condition_stats['uptrend'] += 1
    
    # Condition 4: Breakout
    flat_high = np.max(highs[curr_start:i])
    if closes[i] <= flat_high:
        fail_reasons['breakout'] += 1
        continue
    condition_stats['close > flat_high'] += 1
    
    # Condition 5: Close strength
    daily_range = highs[i] - lows[i]
    close_strength = (closes[i] - lows[i]) / daily_range if daily_range > 0 else 0
    if close_strength < 0.5:
        fail_reasons['close_strength'] += 1
        continue
    condition_stats['close_strength >= 0.5'] += 1
    
    # Condition 6: Upper wick
    upper_wick = highs[i] - closes[i]
    upper_wick_ratio = upper_wick / daily_range if daily_range > 0 else 0
    if upper_wick_ratio > 0.35:
        fail_reasons['upper_wick'] += 1
        continue
    condition_stats['upper_wick <= 0.35'] += 1
    
    # Condition 7: Volume ratio
    vol_ratio = volumes[i] / vol_ma20[i]
    if vol_ratio < 1.2:
        fail_reasons['volume'] += 1
        continue
    condition_stats['volume_ratio >= 1.2'] += 1
    
    # Condition 8: Volume trend
    if i >= 3:
        vol_trend = volumes[i-2:i+1]
        vol_trend_ma = np.mean(volumes[max(0, i-5):i])
        if np.mean(vol_trend) < vol_trend_ma * 0.8:
            fail_reasons['vol_trend'] += 1
            continue
    condition_stats['volume_trend_ok'] += 1
    
    # Condition 9: Flat body deviation
    flat_bodies = bodies[curr_start:i]
    flat_ranges = candle_ranges[curr_start:i]
    flat_body_pcts = body_pct[curr_start:i]
    valid_mask = ~np.isnan(flat_body_pcts) & (flat_ranges > 0)
    
    if np.sum(valid_mask) < min_flat_length:
        continue
    
    avg_body = np.mean(flat_bodies[valid_mask])
    if avg_body <= 0:
        continue
    
    body_devs = np.abs(flat_bodies[valid_mask] - avg_body) / avg_body
    if np.max(body_devs) > 0.30:
        fail_reasons['body_dev'] += 1
        continue
    condition_stats['flat_body_dev <= 0.3'] += 1
    
    # Condition 10: Flat body pct
    if np.min(flat_body_pcts[valid_mask]) < 0.30:
        fail_reasons['body_pct'] += 1
        continue
    condition_stats['flat_min_body_pct >= 0.3'] += 1
    
    # ALL CONDITIONS PASSED - This should be an HFH signal!
    print(f"\n*** HFH Signal found at {result.index[i]} (index {i}) ***")
    print(f"  Flat zone: {result.index[curr_start]} to {result.index[i]}, length={flat_len}")
    print(f"  Pre-high: {pre_high}, Uptrend: {uptrends[i]}")
    print(f"  Breakout: {closes[i]:.2f} > {flat_high:.2f}")
    print(f"  Close strength: {close_strength:.3f}")
    print(f"  Upper wick: {upper_wick_ratio:.3f}")
    print(f"  Volume ratio: {vol_ratio:.3f}")

print("\n=== Condition Stats ===")
for k, v in condition_stats.items():
    print(f"  {k}: {v}")

print("\n=== Failure Reasons (when pre-high was ok) ===")
for k, v in fail_reasons.items():
    print(f"  {k}: {v}")

print("\n=== Done ===")