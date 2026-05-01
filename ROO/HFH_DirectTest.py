"""
HFH 直接測試 - 使用修改後的 calHFH 函數
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

# 計算 EMA (calHFH 需要 EMA 數值)
df = cc.calEMA(df)

# 用修改後的 calHFH 測試
df_test = cc.calHFH(df.copy())

hfhh_count = df_test['HFH'].sum()
print(f"修改後 calHFH 的 HFH 信號數: {hfhh_count}")
print()

# 檢查各階段統計
if 'PreHighCount' in df_test.columns:
    print("=== PreHighCount 分布 ===")
    print(df_test['PreHighCount'].value_counts().sort_index().head(10))
    print()

if 'FlatCount' in df_test.columns:
    print("=== FlatCount 分布 ===")
    print(df_test['FlatCount'].value_counts().sort_index().head(10))
    print()

if 'BreakoutQuality' in df_test.columns:
    print("=== BreakoutQuality 分布 ===")
    bq = df_test['BreakoutQuality']
    print(f"範圍: {bq.min():.1f} - {bq.max():.1f}")
    print(f"平均: {bq.mean():.1f}")
    print(f"非零數量: {(bq > 0).sum()}")
    print()

if 'FalseBreakout' in df_test.columns:
    print("=== FalseBreakout 分布 ===")
    print(df_test['FalseBreakout'].value_counts())
    print()

# 測試更多股票
print("=== 測試多支股票 ===")
import os
slist = [f.replace("P_", "").replace(".csv", "") for f in os.listdir(f"{cc.OUTPATH}/L/") if f.startswith("P_")]
slist = slist[:20]  # 只測試前 20 支

total_hfh = 0
for sno in slist:
    try:
        tdf = pd.read_csv(f"{cc.OUTPATH}/L/P_{sno}.csv", index_col=0)
        tdf.index = pd.to_datetime(tdf.index)
        tdf = cc.calEMA(tdf)
        tdf = cc.calHFH(tdf)
        count = tdf['HFH'].sum()
        if count > 0:
            print(f"{sno}: {count} HFH 信號")
        total_hfh += count
    except Exception as e:
        pass

print(f"\n總 HFH 信號數: {total_hfh}")
print("=== 測試完成 ===")