"""
ATR 計算診斷
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

# 計算 ATR (手動實現)
highs = df['High'].values
lows = df['Low'].values
closes = df['Close'].values
n = len(df)

tr1 = highs - lows
tr2 = np.abs(highs[1:] - closes[:-1])
tr3 = np.abs(lows[1:] - closes[:-1])
tr = np.zeros(n)
tr[0] = tr1[0]
tr[1:] = np.maximum(np.maximum(tr1[1:], tr2), tr3)

atr14 = pd.Series(tr).rolling(window=14, min_periods=1).mean()

print("=== ATR 計算結果 ===")
print(f"ATR14 前 20 個值:")
print(atr14.head(20).values)
print()
print(f"ATR14 最大值: {atr14.max():.4f}")
print(f"ATR14 平均值: {atr14.mean():.4f}")
print()

# 計算 ATR flat multiplier
atr_flat_multiplier = 1.5
w_low_sample = lows[100]  # 取一個樣本
current_max_flat_sample = (atr14.iloc[100] * atr_flat_multiplier) / w_low_sample if w_low_sample > 0 else 0.10
print(f"樣本計算 (index 100):")
print(f"  w_low = {w_low_sample:.2f}")
print(f"  atr14 = {atr14.iloc[100]:.4f}")
print(f"  current_max_flat = {current_max_flat_sample:.4f} ({current_max_flat_sample*100:.2f}%)")
print()

# 測試靜態 max_flat_pct = 0.12
max_flat_pct = 0.12
print(f"靜態 max_flat_pct = {max_flat_pct} ({max_flat_pct*100}%)")
print()

# 測試 ATR 動態調整後的實際效果
print("=== ATR 動態調整測試 ===")
dynamic_count = 0
static_count = 0

for i in range(100, min(500, n)):
    w_high = np.max(highs[max(0, i-10):i+1])
    w_low = np.min(lows[max(0, i-10):i+1])
    
    # ATR 動態
    if w_low > 0:
        current_max_flat = (atr14.iloc[i] * atr_flat_multiplier) / w_low
        current_max_flat = min(current_max_flat, max_flat_pct)
    else:
        current_max_flat = max_flat_pct
    
    flat_pct = (w_high - w_low) / w_low if w_low > 0 else 0
    
    if flat_pct <= current_max_flat:
        dynamic_count += 1
    
    if flat_pct <= max_flat_pct:
        static_count += 1

print(f"100-500 區間:")
print(f"  ATR 動態滿足條件的數量: {dynamic_count}")
print(f"  靜態 max_flat_pct 滿足條件的數量: {static_count}")