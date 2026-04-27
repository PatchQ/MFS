"""
HFH 修正版 - 修復 pre_high_count 的邏輯問題
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  
import numpy as np
import pandas as pd

# 測試修正後的邏輯
def test_fixed_logic():
    # 讀取一支股票
    df = pd.read_csv(f"../Sdata/P_YFData/L/P_0001.HK.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    
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
    
    # 計算每個位置的 pre_high_count（從該位置往前數有多少連續強陽燭）
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
    
    print("=== 測試修正後的邏輯 ===")
    print(f"總共 {n} 根K線")
    print(f"強陽燭總數: {is_strong_bullish.sum()}")
    print()
    
    # 找到符合 HFH 條件的位置
    min_strong_bullish = 3
    hfh_indices = []
    
    for i in range(1, n):
        prev_start = flat_starts[i-1]
        flat_len = i - prev_start
        
        if flat_len >= min_flat_length:
            pre_high = pre_high_count[prev_start] if prev_start > 0 else 0
            if pre_high >= min_strong_bullish:
                flat_high = np.max(highs[prev_start:i])
                if closes[i] > flat_high:
                    hfh_indices.append(i)
    
    print(f"找到 {len(hfh_indices)} 個潛在 HFH 信號")
    for idx in hfh_indices[:5]:
        print(f"  index={idx}, date={df.index[idx]}")
        print(f"    pre_high_count at flat_start ({flat_starts[idx-1]}): {pre_high_count[flat_starts[idx-1]]}")
    
    return hfh_indices

# 測試原始數據的回測情況
def test_backtest_on_stock():
    df = pd.read_csv(f"../Sdata/P_YFData/L/P_0001.HK.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # 使用 calHFH 函數
    df_test = cc.calHFH(df.copy())
    
    print("\n=== calHFH 回測測試 ===")
    print(f"HFH 信號數: {df_test['HFH'].sum()}")
    
    # 顯示 HFH 信號的詳情
    hfhh_rows = df_test[df_test['HFH'] == True]
    if len(hfhh_rows) > 0:
        print(f"\n找到 {len(hfhh_rows)} 個 HFH 信號:")
        for idx, row in hfhh_rows.iterrows():
            print(f"  {idx}: Close={row['Close']:.2f}, FlatCount={row['FlatCount']}")
    else:
        print("\n沒有 HFH 信號")
        print(f"FlatCount 非零: {(df_test['FlatCount'] > 0).sum()}")
        print(f"PreHighCount 最大值: {df_test['PreHighCount'].max() if 'PreHighCount' in df_test.columns else 'N/A'}")
        
        # 顯示 FlatCount 的分布
        if 'FlatCount' in df_test.columns:
            fc = df_test['FlatCount']
            print(f"FlatCount 分布: 0={ (fc==0).sum()}, 5-10={ ((fc>=5)&(fc<=10)).sum()}, >10={ (fc>10).sum()}")

if __name__ == '__main__':
    test_fixed_logic()
    test_backtest_on_stock()