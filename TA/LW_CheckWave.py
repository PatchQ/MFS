import pandas as pd

def checkWave(df, sno, stype, swing_analysis):
    """
    檢查波浪型態並將分析結果合併回主 DataFrame
    """
    
    # 確保 df 的 index 是時間格式，這是後續對齊資料的關鍵
    df.index = pd.to_datetime(df.index)

    # 整理 swing_analysis
    swing_analysis = swing_analysis.reset_index(drop=True)
    swing_analysis['Date'] = pd.to_datetime(swing_analysis['Date'])
    
    # 紀錄 sno 與 stype (保留你原本的邏輯)
    swing_analysis['sno'] = sno
    swing_analysis['stype'] = stype

    # =========================================================
    # 步驟 1: 使用向量化 (shift) 取代 for 迴圈來計算前幾期的值
    # =========================================================
    # .shift(n) 表示把資料「往下推」 n 格，也就是獲取前 n 期的歷史資料
    
    # 組合 PATTERN (將前4期到當期，共5期字串相加)
    # .fillna('') 確保如果前面沒有足夠期數時，空值會轉為空字串，不會變成 NaN
    c0 = swing_analysis['Classification'].fillna('')
    c1 = swing_analysis['Classification'].shift(1).fillna('')
    c2 = swing_analysis['Classification'].shift(2).fillna('')
    c3 = swing_analysis['Classification'].shift(3).fillna('')
    c4 = swing_analysis['Classification'].shift(4).fillna('')
    swing_analysis['PATTERN'] = c4 + c3 + c2 + c1 + c0

    # 計算 WLow (前2期) 與 WHigh (前1期) 的價格和日期
    swing_analysis['WLow'] = swing_analysis['Price'].shift(2).fillna(0)
    swing_analysis['WLDate'] = swing_analysis['Date'].shift(2).dt.strftime("%Y-%m-%d").fillna("")
    
    swing_analysis['WHigh'] = swing_analysis['Price'].shift(1).fillna(0)
    swing_analysis['WHDate'] = swing_analysis['Date'].shift(1).dt.strftime("%Y-%m-%d").fillna("")

    # 將 Date 設為 index，準備與主要 df 進行資料庫式合併 (Join)
    swing_analysis.set_index('Date', inplace=True)

    # =========================================================
    # 步驟 2: 使用 join 根據日期對齊資料，取代原本耗時的 date_match
    # =========================================================
    # 定義我們需要從 swing_analysis 貼過來的欄位
    cols_to_join = ['Classification', 'WLow', 'WLDate', 'WHigh', 'WHDate', 'PATTERN']
    
    # 為了避免重複執行時產生欄位衝突 (如出現 _x, _y)，如果 df 已有這些欄位，先捨棄
    df_cols_to_drop = ['classification', 'WLow', 'WLDate', 'WHigh', 'WHDate', 'HHHL_PATTERN', 'HHHL']
    df = df.drop(columns=[c for c in df_cols_to_drop if c in df.columns])

    # 把欄位無縫貼合到 df (左側合併 how='left' 確保 df 原本的列數和順序不變)
    df = df.join(swing_analysis[cols_to_join], how='left')

    # 將合併進來的欄位重新命名，符合你原先程式碼預設的變數名稱
    df.rename(columns={
        'Classification': 'classification', 
        'PATTERN': 'HHHL_PATTERN'
    }, inplace=True)

    # 將那些沒有對應到波浪紀錄的日子，填補你原本設計的空值預設 ("" 或 0)
    df['classification'] = df['classification'].fillna("")
    df['HHHL_PATTERN'] = df['HHHL_PATTERN'].fillna("")
    df['WLDate'] = df['WLDate'].fillna("")
    df['WHDate'] = df['WHDate'].fillna("")
    df['WLow'] = df['WLow'].fillna(0)
    df['WHigh'] = df['WHigh'].fillna(0)

    # =========================================================
    # 步驟 3: 波浪條件判斷 (尋找目標型態)
    # =========================================================
    WAVERule1 = df['HHHL_PATTERN'] == "LLHHHLHHHL"
    WAVERule2 = df['HHHL_PATTERN'] == "HLHHHLHHHL"
    
    # 只要符合規則1或規則2，該日期的 HHHL 標籤就設為 True，否則為 False
    df["HHHL"] = (WAVERule1 | WAVERule2).fillna(False)

    return df