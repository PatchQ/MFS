"""
HFH 完整過濾條件追蹤
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

# 測試參數
params = {
    'min_strong_bullish': 2,
    'body_ratio': 0.5,
    'require_consecutive_higher': False,
    'min_flat_length': 4,
    'max_flat_pct': 0.12,
    'max_body_deviation': 0.30,
    'min_flat_body_ratio': 0.30,
    'min_close_strength': 0.6,
    'max_upper_wick': 0.2,
    'min_volume_ratio': 1.3,
    'next_day_confirm': True,
    'next_day_max_drop': 0.03,
    'use_dynamic_flat_pct': False,
    'atr_period': 14,
    'atr_flat_multiplier': 1.5
}

# 手動實現 HFH 邏輯並追蹤每個階段
highs = df['High'].values
lows = df['Low'].values
opens = df['Open'].values
closes = df['Close'].values
volumes = df['Volume'].values
n = len(df)

# 計算 EMA uptrend (修改後的版本)
uptrend_condition = (
    (df['EMA22'].values > df['EMA50'].values) &
    (df['EMA50'].values > df['EMA100'].values) &
    (df['EMA100'].values > df['EMA250'].values)
)

# 燭身計算
bodies = np.abs(closes - opens)
candle_ranges = highs - lows
body_pct = np.where(candle_ranges > 0, bodies / candle_ranges, np.nan)
is_bullish = closes > opens

# 強陽燭識別
is_strong_bullish = is_bullish & (body_pct >= params['body_ratio'])

# PreHighCount 計算
PreHighCount = np.zeros(n, dtype=int)
for i in range(n):
    count = 0
    j = i - 1
    while j >= 0 and is_strong_bullish[j]:
        count += 1
        j -= 1
    PreHighCount[i] = count

# 盤整區間計算
left = 0
flat_starts = np.zeros(n, dtype=int)
for right in range(n):
    while left < right:
        w_high = np.max(highs[left:right+1])
        w_low = np.min(lows[left:right+1])
        if w_low > 0 and (w_high - w_low) / w_low <= params['max_flat_pct']:
            break
        left += 1
    flat_starts[right] = left

flat_lengths = np.array([right - flat_starts[right] for right in range(n)])

# 均量計算
vol_ma20 = pd.Series(volumes).rolling(window=20, min_periods=1).mean().values

# 過濾統計
stats = {
    'total': 0,
    'flat_len_pass': 0,
    'pre_high_pass': 0,
    'uptrend_pass': 0,
    'body_dev_pass': 0,
    'body_ratio_pass': 0,
    'close_strength_pass': 0,
    'upper_wick_pass': 0,
    'volume_pass': 0,
    'next_day_pass': 0,
    'final': 0
}

hfh_signals = np.zeros(n, dtype=bool)

for i in range(1, n):
    stats['total'] += 1
    
    curr_start = flat_starts[i]
    flat_len = i - curr_start
    
    # 條件 1：盤整區間長度
    if flat_len >= params['min_flat_length']:
        stats['flat_len_pass'] += 1
    else:
        continue
    
    flat_high = np.max(highs[curr_start:i])
    flat_low = np.min(lows[curr_start:i])
    
    # 條件 2：PreHighCount
    pre_high = PreHighCount[curr_start] if curr_start > 0 else 0
    if pre_high >= params['min_strong_bullish']:
        stats['pre_high_pass'] += 1
    else:
        continue
    
    # 條件 3：Uptrend
    if not uptrend_condition[i]:
        continue
    stats['uptrend_pass'] += 1
    
    # 條件 4：燭身相似度
    flat_bodies = bodies[curr_start:i]
    flat_ranges = candle_ranges[curr_start:i]
    flat_body_pcts = body_pct[curr_start:i]
    
    valid_mask = ~np.isnan(flat_body_pcts) & (flat_ranges > 0)
    if np.sum(valid_mask) < params['min_flat_length']:
        continue
    
    avg_body = np.mean(flat_bodies[valid_mask])
    if avg_body <= 0:
        continue
    
    body_devs = np.abs(flat_bodies[valid_mask] - avg_body) / avg_body
    if np.max(body_devs) > params['max_body_deviation']:
        stats['body_dev_pass'] += 1  # 計入已經通過前面條件的
    else:
        stats['body_dev_pass'] += 1
        stats['body_ratio_pass'] += 1
        # 這裡繼續
    
    # 燭身比例檢測
    if np.min(flat_body_pcts[valid_mask]) < params['min_flat_body_ratio']:
        continue
    stats['body_ratio_pass'] += 1
    
    # 突破日檢測
    if closes[i] <= flat_high:
        continue
    
    daily_range = highs[i] - lows[i]
    close_strength = (closes[i] - lows[i]) / daily_range if daily_range > 0 else 0
    if close_strength < params['min_close_strength']:
        continue
    stats['close_strength_pass'] += 1
    
    upper_wick = highs[i] - closes[i]
    upper_wick_ratio = upper_wick / daily_range if daily_range > 0 else 0
    if upper_wick_ratio > params['max_upper_wick']:
        continue
    stats['upper_wick_pass'] += 1
    
    if volumes[i] < vol_ma20[i] * params['min_volume_ratio']:
        continue
    stats['volume_pass'] += 1
    
    # 隔日確認
    if params['next_day_confirm'] and (i + 1) < n:
        next_close = closes[i + 1]
        breakout_price = closes[i]
        if next_close < breakout_price * (1 - params['next_day_max_drop']):
            continue
    stats['next_day_pass'] += 1
    
    # 全部通過
    hfh_signals[i] = True
    stats['final'] += 1

print("=== HFH 過濾條件統計 ===")
print(f"總候選天數: {stats['total']}")
print(f"盤整區間長度通過: {stats['flat_len_pass']}")
print(f"PreHighCount 通過: {stats['pre_high_pass']}")
print(f"Uptrend 通過: {stats['uptrend_pass']}")
print(f"燭身相似度通過: {stats['body_dev_pass']}")
print(f"燭身比例通過: {stats['body_ratio_pass']}")
print(f"收盤位置通過: {stats['close_strength_pass']}")
print(f"上影線通過: {stats['upper_wick_pass']}")
print(f"成交量通過: {stats['volume_pass']}")
print(f"隔日確認通過: {stats['next_day_pass']}")
print(f"最終 HFH 信號: {stats['final']}")
print()

# 找出瓶頸
bottlenecks = [
    ('盤整區間長度', stats['total'] - stats['flat_len_pass']),
    ('PreHighCount', stats['flat_len_pass'] - stats['pre_high_pass']),
    ('Uptrend', stats['pre_high_pass'] - stats['uptrend_pass']),
    ('燭身相似度', stats['uptrend_pass'] - stats['body_dev_pass']),
    ('燭身比例', stats['body_dev_pass'] - stats['body_ratio_pass']),
    ('收盤位置', stats['body_ratio_pass'] - stats['close_strength_pass']),
    ('上影線', stats['close_strength_pass'] - stats['upper_wick_pass']),
    ('成交量', stats['upper_wick_pass'] - stats['volume_pass']),
    ('隔日確認', stats['volume_pass'] - stats['next_day_pass']),
]

print("=== 瓶頸分析 ===")
for name, count in bottlenecks:
    if count > 0:
        print(f"{name}: 過濾掉 {count} ({count/stats['total']*100:.1f}%)")

print()
print("=== 測試完成 ===")