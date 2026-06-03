import sys
sys.path.insert(0, 'd:/Github/MFS')

from TA.LW_CheckHFH import calHFH
import pandas as pd
import numpy as np

print("=== Full Dataset HFH Search ===")

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
print(f"calHFH result: {result['HFH'].sum()} signals")

# Manual tracing with ACTUAL params
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

max_flat_pct = 0.12
min_flat_length = 4
min_strong_bullish = 2
max_body_deviation = 0.50
min_flat_body_ratio = 0.20

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

# Find ALL candidates through the entire pipeline
candidates = []
for i in range(250, len(result)):
    curr_start = flat_starts[i]
    flat_len = i - curr_start
    
    if flat_len < min_flat_length:
        continue
    
    pre_high = 0
    if curr_start > 0 and curr_start < len(result):
        for j in range(curr_start - 1, max(0, curr_start - 10), -1):
            if is_strong_bullish[j]:
                pre_high += 1
            else:
                break
    if pre_high < min_strong_bullish:
        continue
    
    if not uptrends[i]:
        continue
    
    flat_high = np.max(highs[curr_start:i])
    if closes[i] <= flat_high:
        continue
    
    daily_range = highs[i] - lows[i]
    close_strength = (closes[i] - lows[i]) / daily_range if daily_range > 0 else 0
    
    upper_wick = highs[i] - closes[i]
    upper_wick_ratio = upper_wick / daily_range if daily_range > 0 else 0
    
    vol_ratio = volumes[i] / vol_ma20[i]
    
    flat_bodies = bodies[curr_start:i]
    flat_ranges = candle_ranges[curr_start:i]
    flat_body_pcts = body_pct[curr_start:i]
    valid_mask = ~np.isnan(flat_body_pcts) & (flat_ranges > 0)
    
    avg_body = np.mean(flat_bodies[valid_mask]) if np.sum(valid_mask) >= min_flat_length else 0
    body_devs = np.abs(flat_bodies[valid_mask] - avg_body) / avg_body if avg_body > 0 else [999]
    min_body_pct = np.min(flat_body_pcts[valid_mask]) if np.sum(valid_mask) >= min_flat_length else 0
    
    candidates.append({
        'date': result.index[i],
        'i': i,
        'flat_start': result.index[curr_start],
        'flat_len': flat_len,
        'pre_high': pre_high,
        'close': closes[i],
        'flat_high': flat_high,
        'close_strength': close_strength,
        'upper_wick_ratio': upper_wick_ratio,
        'vol_ratio': vol_ratio,
        'body_dev_max': np.max(body_devs),
        'min_body_pct': min_body_pct,
    })

print(f"\nFound {len(candidates)} candidates through breakout stage")
for c in candidates:
    print(f"\n  {c['date']}: flat={c['flat_start']} to {c['date']}, len={c['flat_len']}")
    print(f"    pre_high={c['pre_high']}, breakout: {c['close']:.2f} > {c['flat_high']:.2f}")
    print(f"    close_str={c['close_strength']:.3f}, wick={c['upper_wick_ratio']:.3f}, vol={c['vol_ratio']:.3f}")
    print(f"    body_dev={c['body_dev_max']:.3f} (max 0.5), min_body_pct={c['min_body_pct']:.3f} (min 0.2)")

# Now apply remaining filters
print("\n\n=== Filtering candidates ===")
pass_filters = []
fail_filters = []
for c in candidates:
    if c['body_dev_max'] <= max_body_deviation and c['min_body_pct'] >= min_flat_body_ratio:
        pass_filters.append(c)
    else:
        fail = []
        if c['body_dev_max'] > max_body_deviation:
            fail.append(f"body_dev={c['body_dev_max']:.3f}>0.5")
        if c['min_body_pct'] < min_flat_body_ratio:
            fail.append(f"body_pct={c['min_body_pct']:.3f}<0.2")
        fail_filters.append((c['date'], fail))

print(f"Passed all filters: {len(pass_filters)}")
for c in pass_filters:
    print(f"  {c['date']}")

print(f"\nFailed at final filter: {len(fail_filters)}")
for date, reasons in fail_filters[:10]:
    print(f"  {date}: {', '.join(reasons)}")

print("\n=== Done ===")