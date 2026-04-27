"""
HFH 深度診斷 - 檢查各階段過濾情況
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

# 計算 EMA
df = cc.calEMA(df)

# 檢查均線多頭排列 (條件 A)
uptrend_condition = (
    (df['Close'] > df['EMA10']) &
    (df['EMA10'] > df['EMA22']) &
    (df['EMA22'] > df['EMA50']) &
    (df['EMA50'] > df['EMA100']) &
    (df['EMA100'] > df['EMA250'])
)

print("=== 均線多頭排列 (uptrend) 統計 ===")
print(f"uptrend=True 的天數: {uptrend_condition.sum()}")
print(f"uptrend=False 的天數: {(~uptrend_condition).sum()}")
print(f"uptrend 比例: {uptrend_condition.mean()*100:.2f}%")
print()

# 檢查 EMA 值
print("=== EMA 值檢查 (前 5 行) ===")
print(df[['Close', 'EMA10', 'EMA22', 'EMA50', 'EMA100', 'EMA250']].head())
print()

# 檢查 EMA250 是否為 NaN
print("=== EMA250 NaN 情況 ===")
print(f"EMA250 NaN 數量: {df['EMA250'].isna().sum()}")
print()

# 強陽燭識別
opens = df['Open'].values
highs = df['High'].values
lows = df['Low'].values
closes = df['Close'].values
bodies = np.abs(closes - opens)
candle_ranges = highs - lows
body_pct = np.where(candle_ranges > 0, bodies / candle_ranges, np.nan)
is_bullish = closes > opens

is_strong_bullish = is_bullish & (body_pct >= 0.5)

print("=== 強陽燭統計 ===")
print(f"強陽燭數量: {is_strong_bullish.sum()}")
print(f"強陽燭比例: {is_strong_bullish.mean()*100:.2f}%")
print()

# 計算 pre_high_count
n = len(df)
pre_high_count = np.zeros(n, dtype=int)
require_consecutive_higher = False

for i in range(n):
    count = 0
    j = i - 1
    while j >= 0 and is_strong_bullish[j]:
        if require_consecutive_higher:
            if j > 0 and closes[j] <= closes[j-1]:
                break
        count += 1
        j -= 1
    pre_high_count[i] = count

print("=== PreHighCount 分布 ===")
print(pd.Series(pre_high_count).value_counts().sort_index())
print()

# 計算盤整區間
left = 0
flat_starts = np.zeros(n, dtype=int)
max_flat_pct = 0.12

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
print(pd.Series(flat_lengths).value_counts().sort_index().head(10))
print(f"盤整區間 >= 4 的數量: {(flat_lengths >= 4).sum()}")
print()

# 最終診斷：同時滿足 uptrend 和 pre_high_count >= 2 的天數
print("=== 關鍵瓶頸分析 ===")
uptrend_with_prehigh = uptrend_condition & (pre_high_count >= 2)
print(f"同時滿足 uptrend=True 且 pre_high_count>=2 的天數: {uptrend_with_prehigh.sum()}")

# 測試放寬 EMA 條件會如何
print()
print("=== 測試不同的 uptrend 條件 ===")

# 測試 EMA3 (較寬鬆的條件: EMA22>EMA50>EMA100>EMA250)
uptrend_ema3 = (df['EMA22'] > df['EMA50']) & (df['EMA50'] > df['EMA100']) & (df['EMA100'] > df['EMA250'])
print(f"使用 EMA3 條件 (EMA22>EMA50>EMA100>EMA250): uptrend=True 天數 = {uptrend_ema3.sum()}")

# 測試 EMA2 (更寬鬆: EMA10>EMA22>EMA50>EMA100>EMA250)
uptrend_ema2 = (df['EMA10'] > df['EMA22']) & (df['EMA22'] > df['EMA50']) & (df['EMA50'] > df['EMA100']) & (df['EMA100'] > df['EMA250'])
print(f"使用 EMA2 條件 (EMA10>EMA22>EMA50>EMA100>EMA250): uptrend=True 天數 = {uptrend_ema2.sum()}")

print()
print("=== 診斷完成 ===")