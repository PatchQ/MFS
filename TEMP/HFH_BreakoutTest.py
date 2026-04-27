"""
HFH 突破日條件瓶頸分析
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

print("=== 突破日條件瓶頸測試 ===")

# 基線參數 (全部放寬到能產生信號)
base_params = {
    'min_strong_bullish': 1,
    'require_consecutive_higher': False,
    'min_flat_length': 3,
    'max_flat_pct': 0.20,
    'max_body_deviation': 999,
    'min_flat_body_ratio': 0,
    'min_close_strength': 0.4,
    'max_upper_wick': 0.4,
    'min_volume_ratio': 1.0,
    'next_day_confirm': False,  # 禁用隔日確認
    'use_dynamic_flat_pct': False
}

# 測試 1: 基線
df1 = cc.calHFH(df.copy(), **base_params)
print(f"基線 (全部放寬): {df1['HFH'].sum()} 信號")

# 測試 2: 恢復 min_close_strength=0.6
params2 = base_params.copy()
params2['min_close_strength'] = 0.6
df2 = cc.calHFH(df.copy(), **params2)
print(f"恢復 min_close_strength=0.6: {df2['HFH'].sum()} 信號")

# 測試 3: 恢復 max_upper_wick=0.2
params3 = base_params.copy()
params3['max_upper_wick'] = 0.2
df3 = cc.calHFH(df.copy(), **params3)
print(f"恢復 max_upper_wick=0.2: {df3['HFH'].sum()} 信號")

# 測試 4: 恢復 min_volume_ratio=1.3
params4 = base_params.copy()
params4['min_volume_ratio'] = 1.3
df4 = cc.calHFH(df.copy(), **params4)
print(f"恢復 min_volume_ratio=1.3: {df4['HFH'].sum()} 信號")

# 測試 5: 恢復 min_close_strength=0.6 和 max_upper_wick=0.2
params5 = base_params.copy()
params5['min_close_strength'] = 0.6
params5['max_upper_wick'] = 0.2
df5 = cc.calHFH(df.copy(), **params5)
print(f"恢復 min_close_strength=0.6 + max_upper_wick=0.2: {df5['HFH'].sum()} 信號")

# 測試 6: 恢復 min_close_strength=0.6 和 min_volume_ratio=1.3
params6 = base_params.copy()
params6['min_close_strength'] = 0.6
params6['min_volume_ratio'] = 1.3
df6 = cc.calHFH(df.copy(), **params6)
print(f"恢復 min_close_strength=0.6 + min_volume_ratio=1.3: {df6['HFH'].sum()} 信號")

# 測試 7: 恢復 max_upper_wick=0.2 和 min_volume_ratio=1.3
params7 = base_params.copy()
params7['max_upper_wick'] = 0.2
params7['min_volume_ratio'] = 1.3
df7 = cc.calHFH(df.copy(), **params7)
print(f"恢復 max_upper_wick=0.2 + min_volume_ratio=1.3: {df7['HFH'].sum()} 信號")

# 測試 8: 恢復全部三個條件
params8 = base_params.copy()
params8['min_close_strength'] = 0.6
params8['max_upper_wick'] = 0.2
params8['min_volume_ratio'] = 1.3
df8 = cc.calHFH(df.copy(), **params8)
print(f"恢復全部三個條件: {df8['HFH'].sum()} 信號")

print()
print("=== 結論 ===")
print("如果測試 2-4 中某個突然降到 0，說明那個參數是主要瓶頸")
print("如果測試 5-7 逐步下降，說明這些參數都有影響")
print()

# 顯示測試 1 的 HFH 信號日期
if df1['HFH'].sum() > 0:
    print("=== 基線 HFH 信號日期 (前 10 個) ===")
    hfh_dates = df1[df1['HFH']].index[:10]
    for d in hfh_dates:
        print(f"  {d.strftime('%Y-%m-%d')}")
print()
print("=== 測試完成 ===")