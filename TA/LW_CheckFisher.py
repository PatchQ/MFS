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
    
    # --- 觸發閾值 ---
    TRIGGER_THRESHOLD = 1.0       # 轉折觸發閾值
    STRONG_THRESHOLD = 0.5       # 強勢確認閾值


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
    df['fisher_signal'] = np.where(fish > 0, 1, -1)
    
    # 交叉信號
    df['fisher_cross_prev'] = df['fisher_signal'].shift(1)
    
    # Trigger lines
    df['fisher_trigger_bull'] = (df['fisher'] > params.TRIGGER_THRESHOLD) & (df['fisher_cross_prev'] <= params.TRIGGER_THRESHOLD)
    df['fisher_trigger_bear'] = (df['fisher'] < -params.TRIGGER_THRESHOLD) & (df['fisher_cross_prev'] >= -params.TRIGGER_THRESHOLD)
    
    # ==========================================
    # 步驟 2: 生成 Fisher Transform 信號
    # ==========================================
    
    fisher = df['fisher'].values
    fisher_trigger_bull = df['fisher_trigger_bull'].values
    fisher_trigger_bear = df['fisher_trigger_bear'].values
    
    # 觸發信號
    strong_bullish = fisher_trigger_bull
    strong_bearish = fisher_trigger_bear
    
    # 強勢區域信號
    medium_bullish = (fisher > params.STRONG_THRESHOLD) & ~strong_bullish
    medium_bearish = (fisher < -params.STRONG_THRESHOLD) & ~strong_bearish
    
    # 弱勢區域信號
    weak_bullish = (fisher > 0) & ~strong_bullish & ~medium_bullish
    weak_bearish = (fisher < 0) & ~strong_bearish & ~medium_bearish
    
    # 設置 FISHER 信號
    df['FISHER'] = False
    df['FISHER_SIGNAL'] = 'neutral'
    df['FISHER_STRENGTH'] = 'weak'
    
    # 強信號: 觸發轉折
    df.loc[strong_bullish, 'FISHER'] = True
    df.loc[strong_bullish, 'FISHER_SIGNAL'] = 'bullish'
    df.loc[strong_bullish, 'FISHER_STRENGTH'] = 'strong'
    
    df.loc[strong_bearish, 'FISHER'] = True
    df.loc[strong_bearish, 'FISHER_SIGNAL'] = 'bearish'
    df.loc[strong_bearish, 'FISHER_STRENGTH'] = 'strong'
    
    # 中等信號
    df.loc[medium_bullish, 'FISHER'] = True
    df.loc[medium_bullish, 'FISHER_SIGNAL'] = 'bullish'
    df.loc[medium_bullish, 'FISHER_STRENGTH'] = 'medium'
    
    df.loc[medium_bearish, 'FISHER'] = True
    df.loc[medium_bearish, 'FISHER_SIGNAL'] = 'bearish'
    df.loc[medium_bearish, 'FISHER_STRENGTH'] = 'medium'
    
    # 弱信號
    df.loc[weak_bullish, 'FISHER'] = True
    df.loc[weak_bullish, 'FISHER_SIGNAL'] = 'bullish'
    df.loc[weak_bullish, 'FISHER_STRENGTH'] = 'weak'
    
    df.loc[weak_bearish, 'FISHER'] = True
    df.loc[weak_bearish, 'FISHER_SIGNAL'] = 'bearish'
    df.loc[weak_bearish, 'FISHER_STRENGTH'] = 'weak'
    
    return df
