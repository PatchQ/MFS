"""
HFH 診斷腳本 - 檢查數據和信號情況
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
print(f"日期範圍: {df.index[0]} 到 {df.index[-1]}")
print()

# 檢查 HFH 信號（用默認參數）
hfhh_count = df['HFH'].sum()
print(f"舊版 HFH 信號數: {hfhh_count}")
print()

# 用默認參數測試新 calHFH
df_test = cc.calHFH(df.copy())
new_hfh_count = df_test['HFH'].sum()
print(f"新版 HFH 信號數 (默認參數): {new_hfh_count}")
print()

# 測試不同參數
test_params = [
    {'min_strong_bullish': 3, 'body_ratio': 0.5, 'min_flat_length': 5, 'max_flat_pct': 0.10, 'min_close_strength': 0.6, 'min_volume_ratio': 1.2},
    {'min_strong_bullish': 2, 'body_ratio': 0.4, 'min_flat_length': 5, 'max_flat_pct': 0.10, 'min_close_strength': 0.5, 'min_volume_ratio': 1.0},
    {'min_strong_bullish': 5, 'body_ratio': 0.6, 'min_flat_length': 7, 'max_flat_pct': 0.08, 'min_close_strength': 0.7, 'min_volume_ratio': 1.5},
]

print("=== 不同參數下的 HFH 信號數 ===")
for i, params in enumerate(test_params):
    df_test = cc.calHFH(df.copy(), **params)
    count = df_test['HFH'].sum()
    print(f"組合 {i+1}: {count} 信號")
    print(f"  參數: {params}")
    print()

# 檢查 PreHighCount 分布
if 'PreHighCount' in df_test.columns:
    print("=== PreHighCount 分布 ===")
    print(df_test['PreHighCount'].value_counts().sort_index().head(10))
    print()

# 檢查 BreakoutQuality 分布
if 'BreakoutQuality' in df_test.columns:
    print("=== BreakoutQuality 分布 ===")
    bq = df_test['BreakoutQuality']
    print(f"範圍: {bq.min():.1f} - {bq.max():.1f}")
    print(f"平均: {bq.mean():.1f}")
    print(f"非零數量: {(bq > 0).sum()}")
    print()

# 檢查 FalseBreakout 分布
if 'FalseBreakout' in df_test.columns:
    print("=== FalseBreakout 分布 ===")
    print(df_test['FalseBreakout'].value_counts())
    print()

print("=== 測試完成 ===")