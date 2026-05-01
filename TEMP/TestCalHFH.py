"""Test calHFH directly"""
import sys
import os
sys.path.append('..')

import UTIL.CommonConfig as cc
import pandas as pd

# Read a stock
df = pd.read_csv('../SData/P_YFdata/L/P_0001.HK.csv', index_col=0)
df = cc.calEMA(df)

print("Testing calHFH with new parameters:")
print(f"EMA22 > EMA50: {(df['EMA22'] > df['EMA50']).sum()}")
print(f"EMA50 > EMA100: {(df['EMA50'] > df['EMA100']).sum()}")
print(f"EMA100 > EMA250: {(df['EMA100'] > df['EMA250']).sum()}")
print()

# Test calHFH with explicit parameters
df_test = cc.calHFH(df.copy(),
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
                    use_dynamic_flat_pct=False)

print(f"HFH signals: {df_test['HFH'].sum()}")
print(f"HFH dates: {[str(d)[:10] for d in df_test[df_test['HFH']].index[:10]]}")
print()

# Test with default parameters
df_default = cc.calHFH(df.copy())
print(f"HFH signals with defaults: {df_default['HFH'].sum()}")