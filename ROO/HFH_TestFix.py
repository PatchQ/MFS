"""
HFH 簡化測試 - 直接禁用 ATR 動態調整
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

print("=== 基本信息 ===")
print(f"總行數: {len(df)}")
print()

# 計算 EMA
df = cc.calEMA(df)

# 直接禁用 ATR 動態調整，使用 max_flat_pct=0.12
df_test = cc.calHFH(df.copy(), 
                    use_dynamic_flat_pct=False,  # 禁用 ATR
                    max_flat_pct=0.15)  # 放寬到 15%

hfhh_count = df_test['HFH'].sum()
print(f"禁用 ATR 動態調整後的 HFH 信號數: {hfhh_count}")
print()

# 檢查 PreHighCount 分布
if 'PreHighCount' in df_test.columns:
    print("=== PreHighCount 分布 ===")
    print(df_test['PreHighCount'].value_counts().sort_index().head(10))
    print()

# 檢查 FlatCount 分布
if 'FlatCount' in df_test.columns:
    print("=== FlatCount 分布 ===")
    print(df_test['FlatCount'].value_counts().sort_index().head(10))
    print()

# 檢查 uptrend
uptrend = (df['EMA22'] > df['EMA50']) & (df['EMA50'] > df['EMA100']) & (df['EMA100'] > df['EMA250'])
print(f"=== Uptrend 統計 ===")
print(f"Uptrend=True 天數: {uptrend.sum()}")
print()

# 測試 PreHighCount >= 2 且 uptrend=True 的數量
prehigh2_uptrend = (df_test['PreHighCount'] >= 2) & uptrend
print(f"PreHighCount>=2 且 Uptrend=True 的天數: {prehigh2_uptrend.sum()}")
print()

print("=== 測試完成 ===")