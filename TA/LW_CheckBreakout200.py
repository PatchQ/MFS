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
# Breakout200Days 策略可調整參數設定
# ==========================================
class Breakout200Params:
    """200日新高突破策略參數集中管理"""
    
    # --- 突破週期參數 ---
    PERIOD = 200                  # 200日移動窗口
    
    # --- 成交量確認參數 ---
    VOLUME_CONFIRM = 1.5          # 放量確認倍數
    
    # --- 動量過濾參數 ---
    PRICE_ABOVE_MA_THRESHOLD = 5   # 價格高於均線百分比閾值


# 預設參數實例
_DEFAULT_PARAMS = Breakout200Params()


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


def _sma(data, period):
    """簡單移動平均"""
    result = np.full_like(data, np.nan)
    for i in range(period - 1, len(data)):
        result[i] = np.mean(data[i - period + 1:i + 1])
    return result


def checkBreakout200(df, sno, stype, params=None):
    """
    200日新高突破策略掃描器
    
    Parameters:
        df: 股票 OHLCV 價格資料
        sno: 股票代號
        stype: 股票類型
        params: Breakout200Params 參數實例，預設使用 _DEFAULT_PARAMS
    """
    if params is None:
        params = _DEFAULT_PARAMS
    
    if df.empty:
        return df

    # 確保索引格式正確
    df.index = pd.to_datetime(df.index)
    
    # 初始化必要的欄位
    cols_to_init = [
        'BREAKOUT200', 'BREAKOUT200_SIGNAL', 'BREAKOUT200_STRENGTH',
        'highest_200', 'lowest_200', 'ma200',
        'price_vs_ma200', 'volume_ma20', 'volume_ratio',
        'new_high_200', 'new_low_200', 'breakout_strength'
    ]
    for col in cols_to_init:
        if col not in df.columns:
            if 'price' in col or 'ma' in col or 'high' in col or 'low' in col or 'ratio' in col or 'strength' in col:
                df[col] = np.nan
            elif 'new_' in col:
                df[col] = False
            else:
                df[col] = np.nan
    
    # ==========================================
    # 步驟 1: 計算 Breakout200 指標
    # ==========================================
    
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    # 200日移動窗口內的最高價和最低價
    df['highest_200'] = _highest(high, params.PERIOD)
    df['lowest_200'] = _lowest(low, params.PERIOD)
    
    # 200日均線
    df['ma200'] = _sma(close, params.PERIOD)
    
    # 價格相對200日均線位置
    df['price_vs_ma200'] = (close - df['ma200'].values) / df['ma200'].values * 100
    
    # 成交量移動平均
    if 'Volume' in df.columns:
        df['volume_ma20'] = _sma(df['Volume'].values, 20)
        df['volume_ratio'] = df['Volume'] / df['volume_ma20']
    else:
        df['volume_ratio'] = 1.0
    
    # 新高新/新低價
    df['new_high_200'] = close >= df['highest_200'].shift(1).values
    df['new_low_200'] = close <= df['lowest_200'].shift(1).values
    
    # 突破強度
    df['breakout_strength'] = close - df['highest_200'].shift(1).values
    
    # ==========================================
    # 步驟 2: 生成 Breakout200 信號
    # ==========================================
    
    volume_ratio = df['volume_ratio'].values
    new_high = df['new_high_200'].values
    new_low = df['new_low_200'].values
    price_vs_ma = df['price_vs_ma200'].values
    
    # 放量突破200日新高
    vol_confirmed_bullish = new_high & (volume_ratio >= params.VOLUME_CONFIRM)
    # 放量跌破200日新低
    vol_confirmed_bearish = new_low & (volume_ratio >= params.VOLUME_CONFIRM)
    
    # 無量突破
    weak_bullish = new_high & ~vol_confirmed_bullish
    weak_bearish = new_low & ~vol_confirmed_bearish
    
    # 價格在均線上方震盪
    price_bullish = (price_vs_ma > params.PRICE_ABOVE_MA_THRESHOLD) & ~new_high
    price_bearish = (price_vs_ma < -params.PRICE_ABOVE_MA_THRESHOLD) & ~new_low
    
    # 設置 BREAKOUT200 信號
    df['BREAKOUT200'] = False
    df['BREAKOUT200_SIGNAL'] = 'neutral'
    df['BREAKOUT200_STRENGTH'] = 'weak'
    
    # 強信號: 放量突破
    df.loc[vol_confirmed_bullish, 'BREAKOUT200'] = True
    df.loc[vol_confirmed_bullish, 'BREAKOUT200_SIGNAL'] = 'bullish'
    df.loc[vol_confirmed_bullish, 'BREAKOUT200_STRENGTH'] = 'strong'
    
    df.loc[vol_confirmed_bearish, 'BREAKOUT200'] = True
    df.loc[vol_confirmed_bearish, 'BREAKOUT200_SIGNAL'] = 'bearish'
    df.loc[vol_confirmed_bearish, 'BREAKOUT200_STRENGTH'] = 'strong'
    
    # 中等信號: 無量突破
    df.loc[weak_bullish, 'BREAKOUT200'] = True
    df.loc[weak_bullish, 'BREAKOUT200_SIGNAL'] = 'bullish'
    df.loc[weak_bullish, 'BREAKOUT200_STRENGTH'] = 'medium'
    
    df.loc[weak_bearish, 'BREAKOUT200'] = True
    df.loc[weak_bearish, 'BREAKOUT200_SIGNAL'] = 'bearish'
    df.loc[weak_bearish, 'BREAKOUT200_STRENGTH'] = 'medium'
    
    # 弱信號: 價格在均線上方/下方
    df.loc[price_bullish, 'BREAKOUT200'] = True
    df.loc[price_bullish, 'BREAKOUT200_SIGNAL'] = 'bullish'
    df.loc[price_bullish, 'BREAKOUT200_STRENGTH'] = 'weak'
    
    df.loc[price_bearish, 'BREAKOUT200'] = True
    df.loc[price_bearish, 'BREAKOUT200_SIGNAL'] = 'bearish'
    df.loc[price_bearish, 'BREAKOUT200_STRENGTH'] = 'weak'
    
    return df
