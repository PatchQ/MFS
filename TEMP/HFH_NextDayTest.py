"""
HFH 隔日確認測試
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  
import numpy as np
import pandas as pd

# 讀取一支股票
df = pd.read_csv(f"{cc.OUTPATH}/L/P_0001.HK.csv", index_col=0)
df.index = pd.to_datetime(df.index)
df = cc.calEMA(df)

print("=== 測試 next_day_confirm 影響 ===")
print()

# 測試 1: 禁用隔日確認
print("測試 1: 禁用隔日確認 (next_day_confirm=False)")
df1 = cc.calHFH(df.copy(), 
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
print(f"  HFH 信號數: {df1['HFH'].sum()}")

# 測試 2: 啟用隔日確認
print("\n測試 2: 啟用隔日確認 (next_day_confirm=True)")
df2 = cc.calHFH(df.copy(), 
                min_strong_bullish=2,
                require_consecutive_higher=False,
                min_flat_length=4,
                max_flat_pct=0.12,
                max_body_deviation=999,
                min_flat_body_ratio=0,
                min_close_strength=0.5,
                max_upper_wick=0.35,
                min_volume_ratio=1.2,
                next_day_confirm=True,
                use_dynamic_flat_pct=False)
print(f"  HFH 信號數: {df2['HFH'].sum()}")

print()
print("=== 測試完成 ===")