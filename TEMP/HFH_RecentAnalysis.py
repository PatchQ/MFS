import sys
sys.path.insert(0, 'd:/Github/MFS')

from TA.LW_CheckHFH import calHFH
import pandas as pd
import numpy as np

print("=== HFH Signal Location Check ===")

df = pd.read_csv('d:/Github/MFS/../Sdata/P_YFdata/L/P_0001.HK.csv', index_col=0)
print(f"Loaded {len(df)} rows")

# Calculate all required EMAs
df['EMA10'] = df['Close'].ewm(span=10, min_periods=10, adjust=False).mean()
df['EMA22'] = df['Close'].ewm(span=22, min_periods=22, adjust=False).mean()
df['EMA50'] = df['Close'].ewm(span=50, min_periods=50, adjust=False).mean()
df['EMA100'] = df['Close'].ewm(span=100, min_periods=100, adjust=False).mean()
df['EMA250'] = df['Close'].ewm(span=250, min_periods=250, adjust=False).mean()

# Check EMA validity
print(f"\nEMA values at recent dates (index 6400-6410):")
for i in range(6400, 6410):
    date = df.index[i]
    ema22 = df['EMA22'].iloc[i]
    ema50 = df['EMA50'].iloc[i]
    ema100 = df['EMA100'].iloc[i]
    ema250 = df['EMA250'].iloc[i]
    uptrend = (ema22 > ema50) and (ema50 > ema100) and (ema100 > ema250)
    print(f"  {date}: EMA22={ema22:.2f}, EMA50={ema50:.2f}, EMA100={ema100:.2f}, EMA250={ema250:.2f}, Uptrend={uptrend}")

# Check uptrend availability - how many rows have valid EMA250?
valid_ema250 = df['EMA250'].notna() & (df['EMA250'] > 0)
print(f"\nRows with valid EMA250: {valid_ema250.sum()} out of {len(df)}")

# Check how many rows have uptrend
uptrend_mask = (
    (df['EMA22'] > df['EMA50']) &
    (df['EMA50'] > df['EMA100']) &
    (df['EMA100'] > df['EMA250'])
)
print(f"Rows with uptrend (EMA22>EMA50>EMA100>EMA250): {uptrend_mask.sum()}")

# Check distribution of PreHighCount
result = calHFH(df.copy())
print(f"\nPreHighCount distribution (non-zero only):")
prehigh_counts = result['PreHighCount'].value_counts().sort_index()
for count, freq in prehigh_counts.items():
    if count > 0:
        print(f"  {count}: {freq} rows")

print(f"\nFlatCount distribution (non-zero only):")
flat_counts = result['FlatCount'].value_counts().sort_index()
for count, freq in flat_counts.items():
    if count > 0:
        print(f"  {count}: {freq} rows")

# Find rows that have everything except something
# Check: PreHighCount >= 2, FlatCount >= 4, Uptrend = True, but HFH = False
potential_signal = (
    (result['PreHighCount'] >= 2) &
    (result['FlatCount'] >= 4) &
    uptrend_mask &
    (~result['HFH'])
)
print(f"\nRows with PreHigh>=2, Flat>=4, Uptrend=True, but HFH=False: {potential_signal.sum()}")

# Check why these potential signals didn't trigger
if potential_signal.sum() > 0:
    potential_indices = result[potential_signal].index.tolist()
    print(f"First 5 potential signal dates: {potential_indices[:5]}")
    
    # Analyze first potential signal
    i = result.index.get_loc(potential_indices[0])
    
    highs = result['High'].values
    lows = result['Low'].values
    closes = result['Close'].values
    volumes = result['Volume'].values
    vol_ma20 = pd.Series(volumes).rolling(window=20, min_periods=1).mean().values
    
    curr_start = result['FlatCount'].iloc[i]  # This is wrong, need to recalc
    
    # Manually check conditions
    flat_len = result['FlatCount'].iloc[i]
    flat_start_idx = i - flat_len
    
    print(f"\n=== Analyzing potential signal at {potential_indices[0]} ===")
    print(f"  Flat length: {flat_len}")
    print(f"  PreHighCount: {result['PreHighCount'].iloc[i]}")
    print(f"  Uptrend: {uptrend_mask.iloc[i]}")
    
    # Recalculate flat_starts for this
    max_flat_pct = 0.12
    left = 0
    flat_starts_manual = np.zeros(len(result), dtype=int)
    for right in range(len(result)):
        while left < right:
            w_high = np.max(highs[left:right+1])
            w_low = np.min(lows[left:right+1])
            if w_low > 0 and (w_high - w_low) / w_low <= max_flat_pct:
                break
            left += 1
        flat_starts_manual[right] = left
    
    curr_start = flat_starts_manual[i]
    flat_high = np.max(highs[curr_start:i])
    flat_low = np.min(lows[curr_start:i])
    
    print(f"  Flat start: {result.index[curr_start]}")
    print(f"  Flat range: {flat_low:.2f} - {flat_high:.2f}")
    print(f"  Close: {closes[i]}")
    print(f"  Breakout: {closes[i] > flat_high}")
    
    # Check breakout conditions
    daily_range = highs[i] - lows[i]
    close_strength = (closes[i] - lows[i]) / daily_range if daily_range > 0 else 0
    upper_wick = highs[i] - closes[i]
    upper_wick_ratio = upper_wick / daily_range if daily_range > 0 else 0
    vol_ratio = volumes[i] / vol_ma20[i]
    
    print(f"  Close strength: {close_strength:.3f} (min 0.5)")
    print(f"  Upper wick ratio: {upper_wick_ratio:.3f} (max 0.35)")
    print(f"  Volume ratio: {vol_ratio:.3f} (min 1.2)")
    
    # Check flat body similarity
    bodies = np.abs(closes - opens)
    candle_ranges = highs - lows
    body_pct = np.where(candle_ranges > 0, bodies / candle_ranges, np.nan)
    
    flat_bodies = bodies[curr_start:i]
    flat_ranges = candle_ranges[curr_start:i]
    flat_body_pcts = body_pct[curr_start:i]
    
    valid_mask = ~np.isnan(flat_body_pcts) & (flat_ranges > 0)
    avg_body = np.mean(flat_bodies[valid_mask])
    body_devs = np.abs(flat_bodies[valid_mask] - avg_body) / avg_body if avg_body > 0 else [999]
    
    print(f"  Flat body deviation: max={np.max(body_devs):.3f} (max 0.30)")
    print(f"  Flat body pcts: {flat_body_pcts[valid_mask]}")
    print(f"  Min flat body ratio: {np.min(flat_body_pcts[valid_mask]):.3f} (min 0.30)")

print("\n=== Done ===")