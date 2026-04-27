"""
HFH 除錯腳本 - 深入分析信號為何為 0
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

print("=== 分析為何 HFH 為 0 ===")
print()

# 創建測試用的完整數據
opens = df['Open'].values
highs = df['High'].values
lows = df['Low'].values
closes = df['Close'].values
volumes = df['Volume'].values
n = len(df)

bodies = np.abs(closes - opens)
candle_ranges = highs - lows
body_pct = np.where(candle_ranges > 0, bodies / candle_ranges, np.nan)
is_bullish = closes > opens

# 強陽燭條件
body_ratio = 0.5
require_consecutive_higher = True
is_strong_bullish = is_bullish & (body_pct >= body_ratio)

if require_consecutive_higher:
    consecutive_higher = closes > np.roll(closes, 1)
    consecutive_higher[0] = False
    is_strong_bullish = is_strong_bullish & consecutive_higher

# 計算 pre_high_count
pre_high_count = np.zeros(n, dtype=int)
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

print("PreHighCount >= 3 的位置:")
high_prehigh_indices = np.where(pre_high_count >= 3)[0]
print(f"總共 {len(high_prehigh_indices)} 個位置")
for idx in high_prehigh_indices[:10]:
    body_pct_val = body_pct[idx] if not np.isnan(body_pct[idx]) else 0
    print(f"  index={idx}, date={df.index[idx]}, pre_high_count={pre_high_count[idx]}")
    print(f"    收盤={closes[idx]}, 前一天收盤={closes[idx-1] if idx>0 else 'N/A'}")
    print(f"    強陽燭={is_strong_bullish[idx]}, body_pct={body_pct_val:.2f}")
print()

# 計算盤整區間
max_flat_pct = 0.10
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

print("=== 檢查盤整區間邏輯 ===")
# 找到 HFH 可能的位置
min_flat_length = 5
for idx in high_prehigh_indices[:5]:
    # 往前看是否能形成盤整區間
    for test_i in range(idx + 5, min(idx + 20, n)):
        prev_start = flat_starts[test_i - 1]
        flat_len = test_i - prev_start
        
        if flat_len >= min_flat_length and prev_start <= idx:
            # 找到了可能的 HFH 組合
            flat_high = np.max(highs[prev_start:test_i])
            
            print(f"\n潛在 HFH 位置:")
            print(f"  盤整起點: {prev_start}, 當前: {test_i}, 盤整長度: {flat_len}")
            print(f"  PreHighCount at prev_start: {pre_high_count[prev_start]}")
            print(f"  Flat high: {flat_high}, 收盤: {closes[test_i]}")
            print(f"  突破條件: {closes[test_i] > flat_high}")
            
            # 檢查均線條件
            ema10 = df['EMA10'].values[test_i]
            ema22 = df['EMA22'].values[test_i]
            ema50 = df['EMA50'].values[test_i]
            ema100 = df['EMA100'].values[test_i]
            ema250 = df['EMA250'].values[test_i]
            
            uptrend = (closes[test_i] > ema10 and ema10 > ema22 and 
                     ema22 > ema50 and ema50 > ema100 and ema100 > ema250)
            print(f"  均線多頭: {uptrend}")
            
            # 檢查突破日質量
            daily_range = highs[test_i] - lows[test_i]
            if daily_range > 0:
                close_strength = (closes[test_i] - lows[test_i]) / daily_range
                upper_wick = highs[test_i] - closes[test_i]
                upper_wick_ratio = upper_wick / daily_range
                
                print(f"  收盤位置: {close_strength:.2f} (需要 >= 0.6)")
                print(f"  上影線比例: {upper_wick_ratio:.2f} (需要 <= 0.2)")
                
                vol_ma20 = pd.Series(volumes).rolling(window=20, min_periods=1).mean().values[test_i]
                vol_ratio = volumes[test_i] / vol_ma20 if vol_ma20 > 0 else 0
                print(f"  成交量比例: {vol_ratio:.2f} (需要 >= 1.2)")
            
            break
    else:
        continue
    break

print("\n=== 測試 calHFH 函數本身 ===")
# 直接調用 calHFH 並打印中間結果
df_test = df.copy()
df_test = cc.calHFH(df_test)

print(f"HFH 信號數: {df_test['HFH'].sum()}")
print(f"FlatCount 非零: {(df_test['FlatCount'] > 0).sum()}")

# 顯示所有 HFH 為 True 的行
hfhh_rows = df_test[df_test['HFH'] == True]
if len(hfhh_rows) > 0:
    print("\nHFH 信號詳情:")
    print(hfhh_rows[['Close', 'FlatCount', 'PreHighCount']].head())
else:
    print("\n沒有 HFH 信號，檢查條件失敗的原因...")
    
    # 測試每個股票的 HFH 信號
    snolist = [s.replace(".csv", "") for s in os.listdir(cc.OUTPATH+"/L")][:10]
    for sno in snolist:
        try:
            temp_df = pd.read_csv(f"{cc.OUTPATH}/L/P_{sno}.csv", index_col=0)
            temp_df.index = pd.to_datetime(temp_df.index)
            temp_df = cc.calHFH(temp_df)
            hfh_count = temp_df['HFH'].sum()
            if hfh_count > 0:
                print(f"{sno}: {hfh_count} HFH 信號")
        except:
            pass