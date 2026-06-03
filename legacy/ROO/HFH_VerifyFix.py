"""
HFH 驗證修復 - 使用默認參數
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

print("=== 測試使用更新後的默認參數 ===")
print()

# 使用 calHFH 的默認參數
df_test = cc.calHFH(df.copy())

hfhh_count = df_test['HFH'].sum()
print(f"使用默認參數的 HFH 信號數: {hfhh_count}")
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

# 測試多支股票
print("=== 測試多支股票 (使用默認參數) ===")
import os
slist = [f.replace("P_", "").replace(".csv", "") for f in os.listdir(f"{cc.OUTPATH}/L/") if f.startswith("P_")]
slist = slist[:30]  # 測試前 30 支

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