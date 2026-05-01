"""
HFH 詳細 Debug - 逐步檢查每個條件
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  
import numpy as np
import pandas as pd

# 讀取一支股票
df = pd.read_csv(f"../SData/P_YFData/L/P_0001.HK.csv", index_col=0)
df.index = pd.to_datetime(df.index)

# 複製數據並調用 calHFH
df_test = df.copy()
df_result = cc.calHFH(df_test)

print("=== calHFH 結果 ===")
print(f"HFH 數量: {df_result['HFH'].sum()}")
print(f"FlatCount 非零: {(df_result['FlatCount'] > 0).sum()}")
print(f"PreHighCount 最大: {df_result['PreHighCount'].max()}")
print()

# 手動重新計算並逐步檢查
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

# 計算 pre_high_count（從該位置往前數有多少連續強陽燭）
pre_high_count = np.zeros(n, dtype=int)
for i in range(n):
    count = 0
    j = i
    while j >= 0 and is_strong_bullish[j]:
        if require_consecutive_higher:
            if j > 0 and closes[j] <= closes[j-1]:
                break
        count += 1
        j -= 1
    pre_high_count[i] = count

# 計算盤整區間起點
max_flat_pct = 0.10
min_flat_length = 5

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

# 均線條件
uptrend_condition = (
    (closes > df['EMA10'].values) &
    (df['EMA10'].values > df['EMA22'].values) &
    (df['EMA22'].values > df['EMA50'].values) &
    (df['EMA50'].values > df['EMA100'].values) &
    (df['EMA100'].values > df['EMA250'].values)
)
uptrends = uptrend_condition

# 逐步檢查每個條件
min_strong_bullish = 3
min_close_strength = 0.6
max_upper_wick = 0.2
min_volume_ratio = 1.2
next_day_confirm = True
next_day_max_drop = 0.03
max_body_deviation = 0.30
min_flat_body_ratio = 0.30

vol_ma20 = pd.Series(volumes).rolling(window=20, min_periods=1).mean().values

# 找到潛在的 HFH 信號（使用與 calHFH 相同的邏輯）
print("=== 逐步檢查每個候選位置 ===")
found_count = 0
fail_reasons = {}

for i in range(1, n):
    prev_start = flat_starts[i-1]
    flat_len = i - prev_start
    
    # 條件 1：盤整區間長度
    if flat_len < min_flat_length:
        fail_reasons['flat_len'] = fail_reasons.get('flat_len', 0) + 1
        continue
    
    flat_high = np.max(highs[prev_start:i])
    
    # 條件 2：強陽燭序列
    pre_high = pre_high_count[prev_start] if prev_start > 0 else 0
    if pre_high < min_strong_bullish:
        fail_reasons['pre_high'] = fail_reasons.get('pre_high', 0) + 1
        continue
    
    # 條件 3：燭身相似度
    flat_bodies = bodies[prev_start:i]
    flat_ranges = candle_ranges[prev_start:i]
    flat_body_pcts = body_pct[prev_start:i]
    
    valid_mask = ~np.isnan(flat_body_pcts) & (flat_ranges > 0)
    if np.sum(valid_mask) < min_flat_length:
        fail_reasons['valid_mask'] = fail_reasons.get('valid_mask', 0) + 1
        continue
    
    avg_body = np.mean(flat_bodies[valid_mask])
    if avg_body <= 0:
        fail_reasons['avg_body'] = fail_reasons.get('avg_body', 0) + 1
        continue
    
    body_devs = np.abs(flat_bodies[valid_mask] - avg_body) / avg_body
    if np.max(body_devs) > max_body_deviation:
        fail_reasons['body_deviation'] = fail_reasons.get('body_deviation', 0) + 1
        continue
    
    if np.min(flat_body_pcts[valid_mask]) < min_flat_body_ratio:
        fail_reasons['body_ratio'] = fail_reasons.get('body_ratio', 0) + 1
        continue
    
    # 條件 4：均線多頭排列
    if not uptrends[i]:
        fail_reasons['uptrend'] = fail_reasons.get('uptrend', 0) + 1
        continue
    
    # 條件 5：價格突破
    if closes[i] <= flat_high:
        fail_reasons['breakout'] = fail_reasons.get('breakout', 0) + 1
        continue
    
    # 條件 6：突破日質量檢測
    daily_range = highs[i] - lows[i]
    if daily_range > 0:
        close_strength = (closes[i] - lows[i]) / daily_range
        upper_wick = highs[i] - closes[i]
        upper_wick_ratio = upper_wick / daily_range
        
        if close_strength < min_close_strength:
            fail_reasons['close_strength'] = fail_reasons.get('close_strength', 0) + 1
            continue
        
        if upper_wick_ratio > max_upper_wick:
            fail_reasons['upper_wick'] = fail_reasons.get('upper_wick', 0) + 1
            continue
        
        if volumes[i] < vol_ma20[i] * min_volume_ratio:
            fail_reasons['volume'] = fail_reasons.get('volume', 0) + 1
            continue
        
        if next_day_confirm and (i + 1) < n:
            next_close = closes[i + 1]
            breakout_price = closes[i]
            if next_close < breakout_price * (1 - next_day_max_drop):
                fail_reasons['next_day'] = fail_reasons.get('next_day', 0) + 1
                continue
    
    # 全部條件滿足
    found_count += 1
    if found_count <= 5:
        print(f"\n候選位置 {i} ({df.index[i]}):")
        print(f"  盤整起點: {prev_start}, 長度: {flat_len}")
        print(f"  pre_high: {pre_high}, flat_high: {flat_high:.2f}, close: {closes[i]:.2f}")
        print(f"  收盤位置: {close_strength:.2f}, 上影線: {upper_wick_ratio:.2f}")
        print(f"  成交量比例: {volumes[i]/vol_ma20[i]:.2f}")
        print(f"  全部條件通過!")

print(f"\n=== 統計結果 ===")
print(f"總共找到 {found_count} 個符合所有條件的位置")
print(f"\n失敗原因統計:")
for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
    print(f"  {reason}: {count}")