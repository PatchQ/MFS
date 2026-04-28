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
# GBS22C 策略可調整參數設定
# ==========================================
class GBS22CParams:
    """GBS22C (Gann-Based System) 策略參數集中管理"""
    
    # --- GBS22C 週期參數 ---
    PERIOD = 22                    # 主週期
    ATR_PERIOD = 14               # ATR 週期
    
    # --- 波動率調整參數 ---
    VOLATILITY_THRESHOLD = 2.0     # 高波動性門檻 (%)
    
    # --- 支撐/阻力位參數 ---
    SUPPORT_MULTIPLIER = 2.0       # 支撐位 ATR 倍數
    RESISTANCE_MULTIPLIER = 2.0   # 阻力位 ATR 倍數


# 預設參數實例
_DEFAULT_PARAMS = GBS22CParams()


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


def _atr(high, low, close, period):
    """平均真實波幅 (Average True Range)"""
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    tr = np.concatenate([[np.nan], tr])
    return _ema(tr, period)


def _ema(data, period):
    """指數移動平均"""
    result = np.full_like(data, np.nan)
    if len(data) >= period:
        result[period - 1] = np.nanmean(data[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(data)):
            if not np.isnan(result[i - 1]):
                result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
    return result


def checkGBS22C(df, sno, stype, params=None):
    """
    GBS22C (Gann-Based System) 策略掃描器
    
    Parameters:
        df: 股票 OHLCV 價格資料
        sno: 股票代號
        stype: 股票類型
        params: GBS22CParams 參數實例，預設使用 _DEFAULT_PARAMS
    """
    if params is None:
        params = _DEFAULT_PARAMS
    
    if df.empty:
        return df

    # 確保索引格式正確
    df.index = pd.to_datetime(df.index)
    
    # 初始化必要的欄位
    cols_to_init = [
        'GBS22C', 'GBS22C_SIGNAL', 'GBS22C_STRENGTH',
        'gbs_ma', 'gann_angle', 'volatility',
        'gann_support', 'gann_resistance', 'trend',
        'period_high', 'period_low', 'breakout_up', 'breakout_down'
    ]
    for col in cols_to_init:
        if col not in df.columns:
            if 'price' in col or 'ma' in col or 'angle' in col or 'support' in col or 'resistance' in col or 'high' in col or 'low' in col or 'volatility' in col:
                df[col] = np.nan
            elif 'trend' in col:
                df[col] = 0
            else:
                df[col] = False
    
    # ==========================================
    # 步驟 1: 計算 GBS22C 指標
    # ==========================================
    
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    # 22週期移動平均
    df['gbs_ma'] = _sma(close, params.PERIOD)
    
    # Gann 角度線 (基於價格*時間關係)
    # 1x1 Gann角度 = 45度
    angle_numerator = close - df['gbs_ma'].values
    angle_denominator = np.arange(len(close)) + 1
    df['gann_angle'] = np.arctan2(angle_numerator, angle_denominator) * 180 / np.pi
    
    # ATR 和波動率
    df['atr'] = _atr(high, low, close, params.ATR_PERIOD)
    df['volatility'] = df['atr'] / close * 100
    
    # 支撐/阻力位 (Gann 扇形)
    df['gann_support'] = df['gbs_ma'] - params.SUPPORT_MULTIPLIER * df['atr']
    df['gann_resistance'] = df['gbs_ma'] + params.RESISTANCE_MULTIPLIER * df['atr']
    
    # Trend direction: 1=多頭, -1=空頭
    df['trend'] = np.where(close > df['gbs_ma'].values, 1, -1)
    
    # 22-period high/low
    df['period_high'] = _highest(high, params.PERIOD)
    df['period_low'] = _lowest(low, params.PERIOD)
    
    # 突破信號
    df['breakout_up'] = close > df['period_high'].shift(1).values
    df['breakout_down'] = close < df['period_low'].shift(1).values
    
    # ==========================================
    # 步驟 2: 生成 GBS22C 信號
    # ==========================================
    
    # 成交量確認
    if 'Volume' in df.columns:
        df['volume_ma20'] = _sma(df['Volume'].values, 20)
        df['volume_ratio'] = df['Volume'] / df['volume_ma20']
    else:
        df['volume_ratio'] = 1.0
    volume_ratio = df['volume_ratio'].values
    
    # 向量化計算信號
    volatility = df['volatility'].values
    breakout_up = df['breakout_up'].values
    breakout_down = df['breakout_down'].values
    trend = df['trend'].values
    
    # 高波動性突破信號
    high_vol_bullish = (volatility > params.VOLATILITY_THRESHOLD) & breakout_up & (volume_ratio >= 1.5)
    high_vol_bearish = (volatility > params.VOLATILITY_THRESHOLD) & breakout_down & (volume_ratio >= 1.5)
    
    # 普通突破信號
    normal_bullish = breakout_up & ~high_vol_bullish
    normal_bearish = breakout_down & ~high_vol_bearish
    
    # 趨勢跟蹤信號
    trend_bullish = (trend == 1) & ~breakout_up & ~high_vol_bullish
    trend_bearish = (trend == -1) & ~breakout_down & ~high_vol_bearish
    
    # 設置 GBS22C 信號
    df['GBS22C'] = False
    df['GBS22C_SIGNAL'] = 'neutral'
    df['GBS22C_STRENGTH'] = 'weak'
    
    # 強信號
    df.loc[high_vol_bullish, 'GBS22C'] = True
    df.loc[high_vol_bullish, 'GBS22C_SIGNAL'] = 'bullish'
    df.loc[high_vol_bullish, 'GBS22C_STRENGTH'] = 'strong'
    
    df.loc[high_vol_bearish, 'GBS22C'] = True
    df.loc[high_vol_bearish, 'GBS22C_SIGNAL'] = 'bearish'
    df.loc[high_vol_bearish, 'GBS22C_STRENGTH'] = 'strong'
    
    # 中等信號
    df.loc[normal_bullish, 'GBS22C'] = True
    df.loc[normal_bullish, 'GBS22C_SIGNAL'] = 'bullish'
    df.loc[normal_bullish, 'GBS22C_STRENGTH'] = 'medium'
    
    df.loc[normal_bearish, 'GBS22C'] = True
    df.loc[normal_bearish, 'GBS22C_SIGNAL'] = 'bearish'
    df.loc[normal_bearish, 'GBS22C_STRENGTH'] = 'medium'
    
    # 弱信號 (趨勢跟蹤)
    df.loc[trend_bullish, 'GBS22C'] = True
    df.loc[trend_bullish, 'GBS22C_SIGNAL'] = 'bullish'
    df.loc[trend_bullish, 'GBS22C_STRENGTH'] = 'weak'
    
    df.loc[trend_bearish, 'GBS22C'] = True
    df.loc[trend_bearish, 'GBS22C_SIGNAL'] = 'bearish'
    df.loc[trend_bearish, 'GBS22C_STRENGTH'] = 'weak'
    
    return df
