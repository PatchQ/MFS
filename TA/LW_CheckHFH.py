import pandas as pd
import numpy as np  

def calHFH(df, min_flat_length=5, max_flat_pct=0.10):
    """
    #找出 High-Flat-High 型態 (動態盤整長度版)
    
    min_flat_length : int
        定義盤整區間的最少 K 線數量，預設為至少 5 支 candle。
    max_flat_pct : float
        盤整區間最高與最低的容許誤差，預設為 0.10 (即 10%)。
    """        
    # 條件 A: 判斷強升勢 (High)
    # 價格 > EMA10 > EMA22 > EMA50 > EMA100 > EMA250
    uptrend_condition = (
        (df['Close'] > df['EMA10']) &
        (df['EMA10'] > df['EMA22']) &
        (df['EMA22'] > df['EMA50']) &
        (df['EMA50'] > df['EMA100']) &
        (df['EMA100'] > df['EMA250'])
    )
    
    # 條件 B: 判斷盤整區間 (Flat)
    # 為了提升效能，將 DataFrame 欄位轉為 numpy array 來進行滑動視窗運算
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    uptrends = uptrend_condition.values
    
    n = len(df)
    hfh_signals = np.zeros(n, dtype=bool)
    flat_counts = np.zeros(n, dtype=int)
    
    # 紀錄每天「往前看」能維持在 10% 波動內的最早起始索引 (left_idx)
    left = 0
    flat_starts = np.zeros(n, dtype=int)
    
    # 步驟 1：計算每天的盤整區間起點
    for right in range(n):
        # 如果當前視窗 [left, right] 的高低差大於 10%，就把 left 往右移
        while left < right:
            w_high = np.max(highs[left:right+1])
            w_low = np.min(lows[left:right+1])
            # 如果最低價大於 0 且波動在允許範圍內，說明這是個合格的區間，停止移動 left
            if w_low > 0 and (w_high - w_low) / w_low <= max_flat_pct:
                break
            left += 1
        flat_starts[right] = left
        
    # 步驟 2：判斷每天是否為突破日
    # 從第 1 天開始判斷，因為我們需要看「前一天」的盤整狀況
    for i in range(1, n):
        # 取得「昨天為止」的盤整起點與長度
        prev_start = flat_starts[i-1]
        flat_len = i - prev_start 
        
        # 如果累積的盤整天數大於等於我們設定的最小值 (5天)
        if flat_len >= min_flat_length:
            # 取得這段盤整期間的最高價
            flat_high = np.max(highs[prev_start:i])
            
            # 條件 B & C: 判斷今天收市價是否突破盤整區間最高價，且均線呈多頭排列
            if closes[i] > flat_high and uptrends[i]:
                hfh_signals[i] = True
                flat_counts[i] = flat_len # 記錄組成的 K 線數量
                
    # 將結果寫回 DataFrame
    df['HFH'] = hfh_signals
    df['FlatCount'] = flat_counts
    
    return df    
    

