import pandas as pd
import numpy as np  

def calHFH(df, 
           # === First High (強勢上升段) ===
           min_strong_bullish=3,        # 必需連續強陽燭數量
           body_ratio=0.5,              # 燭身 / 燭桿 最小比例
           require_consecutive_higher=True,  # 是否要求收盤價遞增
           
           # === Flat Zone (盤整區) ===
           min_flat_length=5,          # 盤整最少蠟燭數
           max_flat_pct=0.10,           # High-Low 最大範圍 (10%)
           max_body_deviation=0.30,     # 燭身大小偏差上限
           min_flat_body_ratio=0.30,    # 盤整區燭身最小比例
           
           # === Breakout (突破) ===
           min_close_strength=0.6,     # 收盤位置 (Low為0, High為1)
           max_upper_wick=0.2,          # 上影線最大比例
           min_volume_ratio=1.2,        # 成交量 / 均量 最小比例
           next_day_confirm=True,       # 是否需要隔日確認
           next_day_max_drop=0.03       # 隔日最大允許跌幅 (3%)
           ):
    """
    找出 High-Flat-High 型態 (加強驗證版)
    
    策略架構：
    1. HIGH-1/2/3: 盤整前需有連續強陽燭序列，確認上升動能
    2. FLAT: 盤整區間價格波動 ≤10%，燭身大小相似
    3. HIGH: 突破日需滿足收盤位置、上影線、成交量等嚴格條件
    
    參數說明：
    -----------
    min_strong_bullish : int
        盤整前必需連續出現的強陽燭數量，預設 3 支。
    body_ratio : float
        強陽燭的燭身 / 燭桿 最小比例，預設 0.5 (50%)。
    require_consecutive_higher : bool
        是否要求每支陽燭收盤價高於前一支，預設 True。
    min_flat_length : int
        盤整區間的最少 K 線數量，預設至少 5 支 candle。
    max_flat_pct : float
        盤整區間最高與最低的容許誤差，預設為 0.10 (即 10%)。
    max_body_deviation : float
        盤整區間內各燭身與平均燭身的偏差上限，預設 0.30 (30%)。
    min_flat_body_ratio : float
        盤整區間內各燭身的燭身 / 燭桿 最小比例，預設 0.30 (30%)。
    min_close_strength : float
        突破日收盤位置（相對燭桿），預設 0.6（需在燭體上半部）。
    max_upper_wick : float
        突破日上影線相對燭桿的最大比例，預設 0.2（20%）。
    min_volume_ratio : float
        突破日成交量相對均量的最小倍數，預設 1.2。
    next_day_confirm : bool
        是否需要隔日確認（排除假突破），預設 True。
    next_day_max_drop : float
        隔日最大允許跌幅（相對於突破日收盤），預設 0.03（3%）。
    
    新增輸出欄位：
    --------------
    HFH : bool
        最終 HFH 信號。
    FlatCount : int
        盤整區的蠟燭數量。
    PreHighCount : int
        盤整前的強陽燭序列數量。
    BreakoutQuality : float
        突破質量分數 (0-100)。
    FalseBreakout : bool
        是否為假突破標記。
    """
    
    # --- 前置計算：各項數值化指標 ---
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    volumes = df['Volume'].values
    n = len(df)
    
    # 燭身與燭桿計算
    bodies = np.abs(closes - opens)                   # 燭身大小
    candle_ranges = highs - lows                      # 燭桿（整體幅度）
    
    # 避免除以零：燭桿為 0 時設為 NaN 後續排除
    body_pct = np.where(candle_ranges > 0, bodies / candle_ranges, np.nan)
    
    # 陽燭標記：收盤 > 開盤
    is_bullish = closes > opens
    
    # 計算 20 日均量（用於成交量驗證）
    vol_ma20 = pd.Series(volumes).rolling(window=20, min_periods=1).mean().values
    
    # 條件 A: 判斷強升勢 (High)
    # 價格 > EMA10 > EMA22 > EMA50 > EMA100 > EMA250
    uptrend_condition = (
        (closes > df['EMA10'].values) &
        (df['EMA10'].values > df['EMA22'].values) &
        (df['EMA22'].values > df['EMA50'].values) &
        (df['EMA50'].values > df['EMA100'].values) &
        (df['EMA100'].values > df['EMA250'].values)
    )
    uptrends = uptrend_condition  # 已是 numpy array，不需要 .values
    
    # =========================================================================
    # 第一階段：識別強陽燭序列（用於判斷盤整前的上升動能）
    # =========================================================================
    # 強陽燭條件：陽燭 + 燭身比例達標
    is_strong_bullish = is_bullish & (body_pct >= body_ratio)
    
    # 如果要求收盤價遞增，則額外過濾：每支陽燭收盤需高於前一支
    if require_consecutive_higher:
        # 收盤價需高於前一根收盤價（才能算作遞增上升）
        consecutive_higher = closes > np.roll(closes, 1)
        consecutive_higher[0] = False  # 第一根無法比較
        is_strong_bullish = is_strong_bullish & consecutive_higher
    
    # 向前追蹤連續強陽燭：計算每個位置往前算有幾根連續強陽燭
    # 從盤整區間的起點往前數，需要有至少 min_strong_bullish 支連續強陽燭
    pre_high_count = np.zeros(n, dtype=int)
    for i in range(n):
        count = 0
        j = i - 1  # 從盤整起點的前一根開始往前算
        while j >= 0 and is_strong_bullish[j]:
            # 嚴格遞增：每根強陽燭的收盤需高於前一根（如果要求）
            if require_consecutive_higher:
                if j > 0 and closes[j] <= closes[j-1]:
                    break
            count += 1
            j -= 1
        pre_high_count[i] = count
    
    # =========================================================================
    # 第二階段：計算盤整區間起點（滾動視窗法）
    # =========================================================================
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
    
    # =========================================================================
    # 第三階段：突破日判定（加入多項嚴格條件）
    # =========================================================================
    hfh_signals = np.zeros(n, dtype=bool)
    flat_counts = np.zeros(n, dtype=int)
    breakout_quality = np.zeros(n, dtype=float)
    false_breakout = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        prev_start = flat_starts[i-1]
        flat_len = i - prev_start
        
        # 條件 1：盤整區間長度達標
        if flat_len >= min_flat_length:
            flat_high = np.max(highs[prev_start:i])
            flat_low = np.min(lows[prev_start:i])
            
            # 條件 2：盤整前的強陽燭序列數量達標
            pre_high = pre_high_count[prev_start] if prev_start > 0 else 0
            if pre_high < min_strong_bullish:
                continue
            
            # 條件 3：盤整區間內的燭身相似度檢測
            flat_bodies = bodies[prev_start:i]
            flat_ranges = candle_ranges[prev_start:i]
            flat_body_pcts = body_pct[prev_start:i]
            
            # 排除 NaN 值
            valid_mask = ~np.isnan(flat_body_pcts) & (flat_ranges > 0)
            if np.sum(valid_mask) < min_flat_length:
                continue
            
            # 計算平均燭身與偏差
            avg_body = np.mean(flat_bodies[valid_mask])
            if avg_body <= 0:
                continue
            
            body_devs = np.abs(flat_bodies[valid_mask] - avg_body) / avg_body
            if np.max(body_devs) > max_body_deviation:
                continue  # 燭身大小差異過大
            
            # 燭身比例檢測（避免十字星）
            if np.min(flat_body_pcts[valid_mask]) < min_flat_body_ratio:
                continue
            
            # 條件 4：均線多頭排列
            if not uptrends[i]:
                continue
            
            # 條件 5：價格突破盤整高點
            if closes[i] <= flat_high:
                continue
            
            # =========================================================================
            # 條件 6：突破日質量檢測（新增）
            # =========================================================================
            # 6a. 收盤位置：需要在燭體上半部
            daily_range = highs[i] - lows[i]
            close_strength = (closes[i] - lows[i]) / daily_range if daily_range > 0 else 0
            if close_strength < min_close_strength:
                false_breakout[i] = True
                continue
            
            # 6b. 上影線比例：不能太大
            upper_wick = highs[i] - closes[i]
            upper_wick_ratio = upper_wick / daily_range if daily_range > 0 else 0
            if upper_wick_ratio > max_upper_wick:
                false_breakout[i] = True
                continue
            
            # 6c. 成交量放大確認
            if volumes[i] < vol_ma20[i] * min_volume_ratio:
                false_breakout[i] = True
                continue
            
            # 6d. 隔日確認（防止假突破）
            if next_day_confirm and (i + 1) < n:
                next_close = closes[i + 1]
                breakout_price = closes[i]
                # 隔日收盤不能低於突破價格的 (1 - next_day_max_drop)
                if next_close < breakout_price * (1 - next_day_max_drop):
                    false_breakout[i] = True
                    continue
            
            # 突破日質量評分（0-100）
            quality_score = (
                close_strength * 30 +  # 收盤位置權重
                (1 - upper_wick_ratio) * 30 +  # 上影線越少越好
                min(volumes[i] / (vol_ma20[i] * min_volume_ratio), 2) / 2 * 20 +  # 成交量超額加分
                pre_high / max(min_strong_bullish, 5) * 20  # 強陽燭序列加分
            )
            breakout_quality[i] = min(quality_score, 100)
            
            # 全部條件滿足，確認 HFH 信號
            hfh_signals[i] = True
            flat_counts[i] = flat_len
    
    # =========================================================================
    # 寫入結果到 DataFrame
    # =========================================================================
    df['HFH'] = hfh_signals
    df['FlatCount'] = flat_counts
    df['PreHighCount'] = pre_high_count
    df['BreakoutQuality'] = breakout_quality
    df['FalseBreakout'] = false_breakout
    
    return df
