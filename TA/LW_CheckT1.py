import pandas as pd

def calT1(df, days, max_flat_pct=0.10):    
    # 取得過去 window 期間內的最高價與最低價極值
    rolling_max = df['High'].rolling(window=days).max()
    rolling_min = df['Low'].rolling(window=days).min()
    
    # 將盤整區間「往後推一天」，因為我們要在「第 N 天(突破日)」去參考「前 N 天」的盤整狀況
    prev_rolling_max = rolling_max.shift(1)
    prev_rolling_min = rolling_min.shift(1)
    
    # 計算前 5 天的波幅是否小於等於 10%
    flat_condition = ((prev_rolling_max - prev_rolling_min) / prev_rolling_min) <= max_flat_pct
    
    # 今天的收盤價必須大於過去N天的最高價 (收盤站穩)
    breakout_condition = df['Close'] > prev_rolling_max
    
    df["T1_"+str(days)] = flat_condition & breakout_condition
    
    # 將無法計算的前期 NaN 轉換為 False
    df["T1_"+str(days)] = df["T1_"+str(days)].fillna(False)
    
    return df
