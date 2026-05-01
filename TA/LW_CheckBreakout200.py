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
    
    # --- 成交量確認參數 (放寬以增加信號數量) ---
    VOLUME_CONFIRM = 1.2          # 放量確認倍數 (從2.0降至1.2)
    
    # --- 動量過濾參數 (放寬) ---
    PRICE_ABOVE_MA_THRESHOLD = 10  # 價格高於均線百分比閾值 (從20降至10)
    
    # --- 信號過濾：需在均線上方連續停留天數 (減少以增加信號) ---
    STRONG_CONFIRM_DAYS = 1      # 強信號需在均線上方停留最少天數 (從5降至1)


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
    
    策略說明：
    - 只在放量突破200日新高時產生買入信號
    - 只在放量跌破200日新低時產生賣出信號
    - 成交量需超過20日均量的2倍確認突破有效性
    
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
    # 步驟 2: 生成 Breakout200 信號（大幅簡化）
    # ==========================================
    
    volume_ratio = df['volume_ratio'].values
    new_high = df['new_high_200'].values
    new_low = df['new_low_200'].values
    price_vs_ma = df['price_vs_ma200'].values
    
    # 放量突破200日新高（需成交量確認，且價格在均線上方）
    vol_confirmed_bullish = new_high & (volume_ratio >= params.VOLUME_CONFIRM) & (price_vs_ma > 0)
    # 放量跌破200日新低（需成交量確認，且價格在均線下方）
    vol_confirmed_bearish = new_low & (volume_ratio >= params.VOLUME_CONFIRM) & (price_vs_ma < 0)
    
    # 將所有資料轉換為 numpy array 避免 pd.Series 兼容性問題
    vol_confirmed_bullish_arr = vol_confirmed_bullish.values if hasattr(vol_confirmed_bullish, 'values') else vol_confirmed_bullish
    vol_confirmed_bearish_arr = vol_confirmed_bearish.values if hasattr(vol_confirmed_bearish, 'values') else vol_confirmed_bearish
    
    # 設置 BREAKOUT200 信號
    df['BREAKOUT200'] = False
    df['BREAKOUT200_SIGNAL'] = 'neutral'
    df['BREAKOUT200_STRENGTH'] = 'neutral'
    
    # 強信號: 放量突破（同時滿足：突破200日新高/新低 + 放量2倍 + 價格在均線同側）
    df.loc[vol_confirmed_bullish_arr, 'BREAKOUT200'] = True
    df.loc[vol_confirmed_bullish_arr, 'BREAKOUT200_SIGNAL'] = 'bullish'
    df.loc[vol_confirmed_bullish_arr, 'BREAKOUT200_STRENGTH'] = 'strong'
    
    df.loc[vol_confirmed_bearish_arr, 'BREAKOUT200'] = True
    df.loc[vol_confirmed_bearish_arr, 'BREAKOUT200_SIGNAL'] = 'bearish'
    df.loc[vol_confirmed_bearish_arr, 'BREAKOUT200_STRENGTH'] = 'strong'
    
    return df
