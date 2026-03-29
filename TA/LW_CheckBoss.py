import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from LW_Calindicator import *    
except ImportError:
    from TA.LW_Calindicator import *    

def checkBoss(df, sno, stype, swing_analysis):
    """
    經過效能與勝率優化的 BOSS 策略掃描器
    """
    if df.empty or swing_analysis.empty:
        return df

    # 1. 確保索引格式正確
    df.index = pd.to_datetime(df.index)
    swing_analysis = swing_analysis.reset_index(drop=True)
    swing_analysis['Date'] = pd.to_datetime(swing_analysis['Date'])

    # 初始化必要的欄位
    cols_to_init = [
        'classification', 'BOSS_PATTERN', 'LLLow', 'LLDate', 'HHClose', 'HHDate', 'HHHigh',
        'VOLATILITY', '22DLow', '33DLow', 'BOSS_STATUS', 'HHEMA1', 'HHEMA2', 'HHEMA3',
        'BOSSB', 'BOSSTP1', 'BOSSTP2', 'BOSSTP3', 'BOSSCL1', 'BOSSCL2', 'BOSSTU1', 'BOSSTU2',
        'bullish_ratio', 'bullish_count', 'strong_bullish', 'medium_bullish', 'weak_bullish',
        'buy_price', 'cl_price', 'tp1_price', 'tp2_price', 'tp3_price', 'BOSS1', 'BOSS2'
    ]
    for col in cols_to_init:
        if col not in df.columns:
            # 依據預期型態賦予初始值
            if 'price' in col or col in ['LLLow', 'HHClose', 'HHHigh', 'VOLATILITY', '22DLow', '33DLow']:
                df[col] = np.nan
            elif col in ['bullish_count', 'strong_bullish', 'medium_bullish', 'weak_bullish']:
                df[col] = 0
            elif col == 'bullish_ratio':
                df[col] = 0.00
            elif col in ['BOSSB', 'BOSSTP1', 'BOSSTP2', 'BOSSTP3', 'BOSSCL1', 'BOSSCL2', 'BOSSTU1', 'BOSSTU2', 'BOSS1', 'BOSS2']:
                df[col] = False
            else:
                df[col] = ""

    # ==========================================
    # 步驟 1: 向量化處理波段分析 (消除第一個大迴圈)
    # ==========================================
    
    # 建立一個新 DataFrame 來快速計算波段特徵
    sa = pd.DataFrame()
    sa['Date'] = swing_analysis['Date']
    sa['Classification'] = swing_analysis['Classification']
    
    # 利用 shift 快速拼湊出連續三個波段的特徵 (例如 LH -> LL -> HH)
    sa['PATTERN'] = swing_analysis['Classification'] + swing_analysis['Classification'].shift(-1) + swing_analysis['Classification'].shift(-2)
    sa['LLLow'] = swing_analysis['Price'].shift(-1)
    sa['LLDate'] = swing_analysis['Date'].shift(-1)
    sa['HHClose'] = swing_analysis['Close'].shift(-2)
    sa['HHDate'] = swing_analysis['Date'].shift(-2)
    sa['HHHigh'] = swing_analysis['Price'].shift(-2)
    
    # 過濾出有完整 Pattern 的列
    sa = sa.dropna(subset=['PATTERN'])
    
    # 將波段資訊對應回主 DataFrame (使用 merge 或索引對齊)
    # 設定 Date 為 index 方便映射
    sa_indexed = sa.set_index('Date')
    
    # 使用 pd.Index.intersection 來安全更新
    valid_dates = df.index.intersection(sa_indexed.index)
    for col in ['PATTERN', 'LLLow', 'LLDate', 'HHClose', 'HHDate', 'HHHigh']:
        df.loc[valid_dates, col if col != 'PATTERN' else 'BOSS_PATTERN'] = sa_indexed.loc[valid_dates, col]

    # 計算波動率 (HHHigh - LLLow) / LLLow
    mask_has_boss = df['BOSS_PATTERN'] != ""
    df.loc[mask_has_boss, 'VOLATILITY'] = ((df.loc[mask_has_boss, 'HHHigh'] - df.loc[mask_has_boss, 'LLLow']) / df.loc[mask_has_boss, 'LLLow']).round(2)

    # 向量化計算 22DLow 和 33DLow
    df['rolling_22_low'] = df['Low'].rolling(window=22, min_periods=1).min()
    df['rolling_33_low'] = df['Low'].rolling(window=33, min_periods=1).min()
    
    # ==========================================
    # 步驟 2: 篩選 BOSS1 與優化勝率邏輯
    # ==========================================
    
    # 新增趨勢過濾器 (例如: 價格必須高於長天期均線，這裡暫時使用簡單的 MA150 代表長期趨勢)
    df['MA150'] = df['Close'].rolling(150, min_periods=1).mean()
    
    # 修正 BOSS1 邏輯：
    # 1. 確保型態正確
    BOSS1Rule1 = df['BOSS_PATTERN'].isin(["LHLLHH", "HHLLHH"])
    # 2. 確保最高中收市價高於LH日的最高位
    BOSS1Rule2 = df['HHClose']>df['High']
    # 3. 確保突破動能 (波動率大於 14%)
    BOSS1Rule3 = df['VOLATILITY'] >= 0.14
    # 4. 大趨勢保護：波段高點必須處於多頭排列或價格大於 MA150 (勝率提升關鍵)
    Trend_Filter = df['Close'] > df['MA150']
    
    df['BOSS1'] = BOSS1Rule1 & BOSS1Rule2 & BOSS1Rule3 #& Trend_Filter
    
    # 提取符合 BOSS1 的日期來進行 K 線計算
    boss1_dates = df[df['BOSS1']].index
    
    for date in boss1_dates:
        ll_date = df.loc[date, 'LLDate']
        hh_date = df.loc[date, 'HHDate']
        
        if pd.isna(ll_date) or pd.isna(hh_date): continue
            
        # 擷取波段區間
        fdf = df.loc[(df.index >= ll_date) & (df.index <= hh_date)]
        if fdf.empty: continue
            
        # 填入 22DLow (取 LL_date 前一天的 rolling low)
        try:
            prev_day_idx = df.index.get_loc(ll_date) - 1
            if prev_day_idx >= 0:
                df.loc[date, '22DLow'] = df['rolling_22_low'].iloc[prev_day_idx]
                df.loc[date, '33DLow'] = df['rolling_33_low'].iloc[prev_day_idx]
        except KeyError:
            pass

        # 呼叫外部指標計算 K 線特徵
        bullish_count, bullish_ratio = calCandleStick(fdf)
        strong_bullish, medium_bullish, weak_bullish = calCandleStickBody(fdf)
        
        df.loc[date, 'bullish_count'] = bullish_count
        df.loc[date, 'bullish_ratio'] = bullish_ratio
        df.loc[date, 'strong_bullish'] = strong_bullish
        df.loc[date, 'medium_bullish'] = medium_bullish
        df.loc[date, 'weak_bullish'] = weak_bullish

    # ==========================================
    # 步驟 3: 篩選 BOSS2 與設定交易計畫
    # ==========================================
    
    BOSS2Rule1 = df['LLLow'] <= df['22DLow'] 
    BOSS2Rule2 = df["bullish_ratio"] >= 0.65
    BOSS2Rule3 = df["strong_bullish"] >= 1
    BOSS2Rule4 = df["bullish_count"] >= 4    

    df["BOSS2"] = df["BOSS1"] & BOSS2Rule1 & BOSS2Rule2 & BOSS2Rule3 & BOSS2Rule4
    
    mask_boss2 = df["BOSS2"]
    
    # 計算買入與停利損價位 (加入 1% 停損緩衝區，減少被洗盤機率)
    df.loc[mask_boss2, "buy_price"] = ((df.loc[mask_boss2, "HHHigh"] + df.loc[mask_boss2, "LLLow"]) / 2).round(2)
    df.loc[mask_boss2, "cl_price"] = (df.loc[mask_boss2, "LLLow"] * 0.99).round(2) # 停損下移 1%
    df.loc[mask_boss2, "tp1_price"] = df.loc[mask_boss2, "HHHigh"]
    df.loc[mask_boss2, "tp2_price"] = df.loc[mask_boss2, "buy_price"] + (df.loc[mask_boss2, "HHHigh"] - df.loc[mask_boss2, "buy_price"]) * 2 
    df.loc[mask_boss2, "tp3_price"] = df.loc[mask_boss2, "buy_price"] + (df.loc[mask_boss2, "HHHigh"] - df.loc[mask_boss2, "buy_price"]) * 3 
    
    # 將日期格式化寫入 Status
    boss2_date_strs = df[mask_boss2].index.strftime("%Y/%m/%d")
    df.loc[mask_boss2, "BOSS_STATUS"] = "SB1-" + boss2_date_strs

    # ==========================================
    # 步驟 4: 高效模擬交易路徑 (去除繁重的 loc 日期遮罩)
    # ==========================================
    
    # 轉為 numpy 陣列加快未來查找速度
    dates_array = df.index.values
    lows_array = df['Low'].values
    highs_array = df['High'].values
    closes_array = df['Close'].values
    ema3_array = df['EMA3'].values if 'EMA3' in df.columns else np.ones(len(df), dtype=bool)

    boss2_indices = np.where(df['BOSS2'])[0]
    
    for i in boss2_indices:
        hhdate = df['HHDate'].iloc[i]
        if pd.isna(hhdate): continue
            
        hh_idx = df.index.get_loc(pd.to_datetime(hhdate))
        buydeadline_idx = min(hh_idx + 22, len(df) - 1)
        
        buy_price = df['buy_price'].iloc[i]
        cl_price = df['cl_price'].iloc[i]
        tp1_price = df['tp1_price'].iloc[i]
        tp2_price = df['tp2_price'].iloc[i]
        tp3_price = df['tp3_price'].iloc[i]
        hh_price = tp1_price
        startbossdate = df.index[i].strftime("%Y/%m/%d")

        # 1. 尋找進場點
        search_window = slice(hh_idx + 1, buydeadline_idx + 1)
        buy_mask = (lows_array[search_window] <= buy_price * 1.005) & (ema3_array[search_window] == True)
        high_mask = highs_array[search_window] > hh_price
        
        buy_hit_indices = np.where(buy_mask)[0]
        high_hit_indices = np.where(high_mask)[0]
        
        if len(buy_hit_indices) == 0:
            continue
            
        first_buy_rel_idx = buy_hit_indices[0]
        
        if len(high_hit_indices) > 0 and high_hit_indices[0] < first_buy_rel_idx:
            continue
            
        # 加上 int() 確保絕對是整數
        buy_idx = int(hh_idx + 1 + first_buy_rel_idx)
        buy_date = df.index[buy_idx]
        
        df.loc[buy_date, 'BOSS_STATUS'] = "BY1-" + startbossdate
        df.loc[buy_date, 'BOSSB'] = True
        for col, val in zip(['buy_price', 'cl_price', 'tp1_price', 'tp2_price', 'tp3_price'], 
                            [buy_price, cl_price, tp1_price, tp2_price, tp3_price]):
            df.loc[buy_date, col] = val

        # 2. 尋找 TP1 與 CL1
        tp_deadline_idx = min(buy_idx + 30, len(df) - 1)
        trade_window = slice(buy_idx, tp_deadline_idx + 1)
        
        tp1_mask = highs_array[trade_window] >= tp1_price * 0.995
        cl1_mask = closes_array[trade_window] < cl_price
        
        tp1_hits = np.where(tp1_mask)[0]
        cl1_hits = np.where(cl1_mask)[0]
        
        first_tp1_idx = tp1_hits[0] if len(tp1_hits) > 0 else np.inf
        first_cl1_idx = cl1_hits[0] if len(cl1_hits) > 0 else np.inf
        
        if first_tp1_idx == np.inf and first_cl1_idx == np.inf:
            deadline_date = df.index[tp_deadline_idx]
            if deadline_date < pd.Timestamp(datetime.now().date()):
                if ((lows_array[tp_deadline_idx] - buy_price) / buy_price) >= 0.01:
                    df.loc[deadline_date, 'BOSSTU1'] = True
                    df.loc[deadline_date, 'BOSS_STATUS'] = "TU1-" + startbossdate
                else:
                    df.loc[deadline_date, 'BOSSTU2'] = True
                    df.loc[deadline_date, 'BOSS_STATUS'] = "TU2-" + startbossdate
            continue 
            
        if first_cl1_idx <= first_tp1_idx:
            cl1_date = df.index[int(buy_idx + first_cl1_idx)] # 加上 int() 防禦
            df.loc[cl1_date, 'BOSS_STATUS'] = "CL1-" + startbossdate
            df.loc[cl1_date, 'BOSSCL1'] = True
            continue 
            
        tp1_idx_global = int(buy_idx + first_tp1_idx)
        tp1_date = df.index[tp1_idx_global]
        df.loc[tp1_date, 'BOSS_STATUS'] = "TP1-" + startbossdate
        df.loc[tp1_date, 'BOSSTP1'] = True
        
        # 3. 追蹤 TP2
        tp2_deadline_idx = min(tp1_idx_global + 30, len(df) - 1)
        trade_window_2 = slice(tp1_idx_global, tp2_deadline_idx + 1)
        
        tp2_mask = highs_array[trade_window_2] >= tp2_price * 0.99
        cl2_mask = closes_array[trade_window_2] < cl_price
        
        tp2_hits = np.where(tp2_mask)[0]
        cl2_hits = np.where(cl2_mask)[0]
        
        first_tp2_idx = tp2_hits[0] if len(tp2_hits) > 0 else np.inf
        first_cl2_idx = cl2_hits[0] if len(cl2_hits) > 0 else np.inf
        
        # 【關鍵修復】如果 TP2 期間沒碰到停利也沒碰到停損，直接跳出追蹤
        if first_tp2_idx == np.inf and first_cl2_idx == np.inf:
            continue

        if first_cl2_idx <= first_tp2_idx:
            cl2_date = df.index[int(tp1_idx_global + first_cl2_idx)]
            df.loc[cl2_date, 'BOSS_STATUS'] = "CL2-" + startbossdate
            df.loc[cl2_date, 'BOSSCL2'] = True
            continue
            
        if first_tp2_idx != np.inf:
            tp2_idx_global = int(tp1_idx_global + first_tp2_idx)
            tp2_date = df.index[tp2_idx_global]
            df.loc[tp2_date, 'BOSS_STATUS'] = "TP2-" + startbossdate
            df.loc[tp2_date, 'BOSSTP2'] = True
            
            # 4. 追蹤 TP3
            tp3_deadline_idx = min(tp2_idx_global + 30, len(df) - 1)
            trade_window_3 = slice(tp2_idx_global, tp3_deadline_idx + 1)
            tp3_mask = highs_array[trade_window_3] >= tp3_price * 0.99
            
            tp3_hits = np.where(tp3_mask)[0]
            if len(tp3_hits) > 0:
                tp3_date = df.index[int(tp2_idx_global + tp3_hits[0])]
                df.loc[tp3_date, 'BOSS_STATUS'] = "TP3-" + startbossdate
                df.loc[tp3_date, 'BOSSTP3'] = True
    
    return df