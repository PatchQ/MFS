"""
HFH 最小測試 - 只保留核心條件
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

print("=== 測試不同參數組合 ===")

# 測試 1: 原始參數
print("\n測試 1: 原始參數 (min_flat_body_ratio=0.30)")
df1 = cc.calHFH(df.copy(), 
                min_strong_bullish=2,
                require_consecutive_higher=False,
                min_flat_length=4,
                max_flat_pct=0.12,
                min_flat_body_ratio=0.30,  # 這個可能太嚴格
                use_dynamic_flat_pct=False)
print(f"  HFH 信號數: {df1['HFH'].sum()}")

# 測試 2: 放寬燭身比例要求
print("\n測試 2: 放寬燭身比例 (min_flat_body_ratio=0.10)")
df2 = cc.calHFH(df.copy(), 
                min_strong_bullish=2,
                require_consecutive_higher=False,
                min_flat_length=4,
                max_flat_pct=0.12,
                min_flat_body_ratio=0.10,  # 放寬
                use_dynamic_flat_pct=False)
print(f"  HFH 信號數: {df2['HFH'].sum()}")

# 測試 3: 禁用燭身比例檢測
print("\n測試 3: 禁用燭身比例檢測 (min_flat_body_ratio=0)")
df3 = cc.calHFH(df.copy(), 
                min_strong_bullish=2,
                require_consecutive_higher=False,
                min_flat_length=4,
                max_flat_pct=0.12,
                min_flat_body_ratio=0,  # 完全禁用
                use_dynamic_flat_pct=False)
print(f"  HFH 信號數: {df3['HFH'].sum()}")

# 測試 4: 禁用燭身相似度檢測
print("\n測試 4: 禁用燭身相似度 (max_body_deviation=999)")
df4 = cc.calHFH(df.copy(), 
                min_strong_bullish=2,
                require_consecutive_higher=False,
                min_flat_length=4,
                max_flat_pct=0.12,
                max_body_deviation=999,  # 禁用
                use_dynamic_flat_pct=False)
print(f"  HFH 信號數: {df4['HFH'].sum()}")

# 測試 5: 全部放寬
print("\n測試 5: 全部放寬")
df5 = cc.calHFH(df.copy(), 
                min_strong_bullish=1,
                require_consecutive_higher=False,
                min_flat_length=3,
                max_flat_pct=0.20,
                max_body_deviation=999,
                min_flat_body_ratio=0,
                min_close_strength=0.4,
                max_upper_wick=0.4,
                min_volume_ratio=1.0,
                use_dynamic_flat_pct=False)
print(f"  HFH 信號數: {df5['HFH'].sum()}")

# 檢查 FlatCount 分布
print("\n=== FlatCount 分布 (測試 3) ===")
if 'FlatCount' in df3.columns:
    print(df3['FlatCount'].value_counts().sort_index().head(20))
print()

print("=== 測試完成 ===")