"""
HFH 盤整區間診斷 - 追蹤 flat_starts 計算
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

# 手動實現盤整區間計算邏輯
highs = df['High'].values
lows = df['Low'].values
n = len(df)

max_flat_pct = 0.12
min_flat_length = 4

# 計算滾動窗口盤整區間
left = 0
flat_starts = np.zeros(n, dtype=int)

for right in range(n):
    while left < right:
        w_high = np.max(highs[left:right+1])
        w_low = np.min(lows[left:right+1])
        
        if w_low > 0 and (w_high - w_low) / w_low <= max_flat_pct:
            break
        left += 1
    flat_starts[right] = left

# 統計盤整區間長度
flat_lengths = np.array([right - flat_starts[right] for right in range(n)])

print("=== 盤整區間長度分布 ===")
print(pd.Series(flat_lengths).value_counts().sort_index().head(20))
print()
print(f"盤整區間 >= {min_flat_length} 的數量: {(flat_lengths >= min_flat_length).sum()}")
print()

# 測試使用不同的 max_flat_pct
for pct in [0.10, 0.12, 0.15, 0.20, 0.30]:
    left = 0
    temp_flat_starts = np.zeros(n, dtype=int)
    for right in range(n):
        while left < right:
            w_high = np.max(highs[left:right+1])
            w_low = np.min(lows[left:right+1])
            if w_low > 0 and (w_high - w_low) / w_low <= pct:
                break
            left += 1
        temp_flat_starts[right] = left
    
    temp_lengths = np.array([right - temp_flat_starts[right] for right in range(n)])
    count = (temp_lengths >= min_flat_length).sum()
    print(f"max_flat_pct={pct}: 盤整區間長度>={min_flat_length} 的數量 = {count}")
print()

# 測試 PreHighCount >= 2 的情況
PreHighCount = np.zeros(n, dtype=int)
body_ratio = 0.5
opens = df['Open'].values
closes = df['Close'].values
bodies = np.abs(closes - opens)
candle_ranges = highs - lows
body_pct = np.where(candle_ranges > 0, bodies / candle_ranges, np.nan)
is_bullish = closes > opens
is_strong_bullish = is_bullish & (body_pct >= body_ratio)

for i in range(n):
    count = 0
    j = i - 1
    while j >= 0 and is_strong_bullish[j]:
        count += 1
        j -= 1
    PreHighCount[i] = count

print("=== PreHighCount >= 2 的日期統計 ===")
prehigh2_dates = np.where(PreHighCount >= 2)[0]
print(f"PreHighCount >= 2 的總天數: {len(prehigh2_dates)}")
print(f"對應的日期範圍: {df.index[prehigh2_dates[0]]} 到 {df.index[prehigh2_dates[-1]]}")
print()

# 測試：當 PreHighCount >= 2 且盤整區間足夠長時
uptrend = (df['EMA22'] > df['EMA50']) & (df['EMA50'] > df['EMA100']) & (df['EMA100'] > df['EMA250'])
print(f"=== Uptrend 統計 ===")
print(f"Uptrend=True 天數: {uptrend.sum()}")
print()

# 候選天數
candidate = (PreHighCount >= 2) & uptrend & (flat_lengths >= min_flat_length)
print(f"同時滿足 PreHighCount>=2, Uptrend=True, 盤整區間>={min_flat_length} 的天數: {candidate.sum()}")
print()

print("=== 測試完成 ===")