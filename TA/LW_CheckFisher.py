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


# ==========================================
# Fisher Transform 策略可調整參數設定
# ==========================================
class FisherParams:
    """Fisher Transform 策略參數集中管理"""
    
    # --- Fisher Transform 參數 ---
    PERIOD = 10                   # 標準化價格週期
    
    # --- 觸發閾值 (大幅放寬以增加信號數量) ---
    TRIGGER_THRESHOLD = 0.0       # 轉折觸發閾值 (從0.3降至0.0)
    STRONG_THRESHOLD = 0.0        # 強勢確認閾值 (從0.2降至0.0，零軸交叉)
    
    # --- 信號過濾：需在門檻區域連續停留天數 ---
    STRONG_CONFIRM_DAYS = 2       # 強信號需在 STRONG 區域連續停留天數


# 預設參數實例
_DEFAULT_PARAMS = FisherParams()


def _highest(data, period):
    """移動最高值"""
    result = np.full_like(data, np.nan)
    for i in range(period - 1, len(data)):
        result[i] = np.max(data[i - period + 1:i + 1])
    return result


def _lowest(data, period):
    """移動最低值"""
    result = np.full_like(data, np.nan)
    for i in range(period - 1, len(data)):
        result[i] = np.min(data[i - period + 1:i + 1])
    return result


def checkFisher(df, sno, stype, params=None):
    """
    Fisher Transform 策略掃描器
    
    策略說明：
    - 只在超賣區（fisher < -TRIGGER_THRESHOLD）且反轉時產生買入信號
    - 只在超買區（fisher > TRIGGER_THRESHOLD）且反轉時產生賣出信號
    - 需在 STRONG 區域連續停留足夠天數才確認信號有效性
    
    Parameters:
        df: 股票 OHLCV 價格資料
        sno: 股票代號
        stype: 股票類型
        params: FisherParams 參數實例，預設使用 _DEFAULT_PARAMS
    """
    if params is None:
        params = _DEFAULT_PARAMS
    
    if df.empty:
        return df

    # 確保索引格式正確
    df.index = pd.to_datetime(df.index)
    
    # 初始化必要的欄位
    cols_to_init = [
        'FISHER', 'FISHER_SIGNAL', 'FISHER_STRENGTH',
        'fisher', 'fisher_trigger_bull', 'fisher_trigger_bear'
    ]
    for col in cols_to_init:
        if col not in df.columns:
            if 'fisher' in col and 'trigger' not in col:
                df[col] = np.nan
            else:
                df[col] = False
    
    # ==========================================
    # 步驟 1: 計算 Fisher Transform
    # ==========================================
    
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    # 計算 HL2 價格
    hl2 = (high + low) / 2
    
    # 標準化價格範圍
    max_hl = _highest(hl2, params.PERIOD)
    min_hl = _lowest(hl2, params.PERIOD)
    
    # 避免除零
    range_val = max_hl - min_hl
    range_val = np.where(range_val == 0, 0.0001, range_val)
    
    # Fisher Transform 公式
    v = 2 * ((hl2 - min_hl) / range_val - 0.5)
    v = np.clip(v, -0.999, 0.999)  # 防止 log(0)
    
    # 初始值
    fish = np.zeros_like(v)
    fish[0] = 0.0
    
    # 計算 Fisher
    for i in range(1, len(v)):
        if not np.isnan(v[i]) and not np.isnan(fish[i-1]):
            fish[i] = 0.5 * fish[i-1] + 0.5 * (2.5 * np.log((1 + v[i]) / (1 - v[i])))
    
    df['fisher'] = fish
    
    # ==========================================
    # 步驟 2: 生成 Fisher Transform 信號
    # ==========================================
    
    fisher = df['fisher'].values
    fisher_prev = pd.Series(fisher).shift(1).fillna(0).values
    
    # 區域判斷
    in_strong_bull_zone = fisher > params.STRONG_THRESHOLD
    in_strong_bear_zone = fisher < -params.STRONG_THRESHOLD
    
    # 計算在 STRONG 區域的連續天數
    strong_bull_days = pd.Series(in_strong_bull_zone.astype(int)).rolling(window=params.STRONG_CONFIRM_DAYS, min_periods=params.STRONG_CONFIRM_DAYS).sum().values
    strong_bear_days = pd.Series(in_strong_bear_zone.astype(int)).rolling(window=params.STRONG_CONFIRM_DAYS, min_periods=params.STRONG_CONFIRM_DAYS).sum().values
    
    # 從極度超賣區反轉（底部反轉）：fisher 從 < -TRIGGER_THRESHOLD 上升到 > 0（零軸）
    bull_reversal = (fisher_prev < -params.TRIGGER_THRESHOLD) & (fisher > 0)
    
    # 從極度超買區反轉（頂部反轉）：fisher 從 > TRIGGER_THRESHOLD 下降到 < 0（零軸）
    bear_reversal = (fisher_prev > params.TRIGGER_THRESHOLD) & (fisher < 0)
    
    # 將所有資料轉換為 numpy array 避免 pd.Series 兼容性問題
    bull_reversal_arr = bull_reversal.values if hasattr(bull_reversal, 'values') else bull_reversal
    bear_reversal_arr = bear_reversal.values if hasattr(bear_reversal, 'values') else bear_reversal
    
    # 設置 FISHER 信號
    df['FISHER'] = False
    df['FISHER_SIGNAL'] = 'neutral'
    df['FISHER_STRENGTH'] = 'neutral'
    
    # 買入信號: 從極度超賣區反轉且在強勢區域確認
    df.loc[bull_reversal_arr, 'FISHER'] = True
    df.loc[bull_reversal_arr, 'FISHER_SIGNAL'] = 'bullish'
    df.loc[bull_reversal_arr, 'FISHER_STRENGTH'] = 'strong'
    
    # 賣出信號: 從極度超買區反轉且在強勢區域確認
    df.loc[bear_reversal_arr, 'FISHER'] = True
    df.loc[bear_reversal_arr, 'FISHER_SIGNAL'] = 'bearish'
    df.loc[bear_reversal_arr, 'FISHER_STRENGTH'] = 'strong'
    
    return df
