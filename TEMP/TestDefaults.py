import sys
sys.path.insert(0, 'd:/Github/MFS')

# Force reimport
import importlib
if 'TA.LW_CheckHFH' in sys.modules:
    del sys.modules['TA.LW_CheckHFH']
if 'TA' in sys.modules:
    del sys.modules['TA']

from TA.LW_CheckHFH import calHFH
import pandas as pd

print("Starting test...")

df = pd.read_csv('d:/Github/MFS/../SData/P_YFdata/L/P_0001.HK.csv', index_col=0)
df['EMA22'] = df['Close'].ewm(span=22, min_periods=22, adjust=False).mean()
df['EMA50'] = df['Close'].ewm(span=50, min_periods=50, adjust=False).mean()
df['EMA100'] = df['Close'].ewm(span=100, min_periods=100, adjust=False).mean()
df['EMA250'] = df['Close'].ewm(span=250, min_periods=250, adjust=False).mean()

print(f"Data loaded: {len(df)} rows")

# Test with defaults
result = calHFH(df.copy())
print(f'HFH with defaults: {result["HFH"].sum()}')

# Test with relaxed params
result2 = calHFH(df.copy(), 
    min_strong_bullish=2,
    require_consecutive_higher=False,
    min_flat_length=4,
    max_flat_pct=0.12,
    max_body_deviation=999,
    min_flat_body_ratio=0,
    min_close_strength=0.5,
    max_upper_wick=0.35,
    min_volume_ratio=1.2,
    next_day_confirm=False,
    use_dynamic_flat_pct=False
)
print(f'HFH with relaxed params: {result2["HFH"].sum()}')

# Check FlatCount distribution
if 'FlatCount' in result.columns:
    print(f"FlatCount distribution: {result['FlatCount'].value_counts().to_dict()}")

print("Done!")