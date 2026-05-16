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
# Ichimoku 策略可調整參數設定
# ==========================================
class IchimokuParams:
    """Ichimoku 策略參數集中管理"""
    
    # --- 一目均衡表參數 ---
    TENKAN_PERIOD = 9       # 轉換線週期
    KIJUN_PERIOD = 26       # 基準線週期
    SENKOU_B_PERIOD = 52    # 先行區間B週期
    DISPLACEMENT = 26       # 位移量
    
    # --- ATR止損止盈參數（ICHIMOKU + ATR方案）---
    ATR_PERIOD = 14            # ATR週期
    ATR_MULTIPLIER_SL = 2.0    # 止損：2倍ATR
    ATR_MULTIPLIER_TP = 3.0    # 止盈：3倍ATR
    USE_ATR_STOP = True       # 使用ATR動態止損（替代固定百分比）


# 預設參數實例
_DEFAULT_PARAMS = IchimokuParams()


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


def checkIchimoku(df, sno, stype, params=None):
    """
    Ichimoku Cloud 策略掃描器
    
    Parameters:
        df: 股票 OHLCV 價格資料
        sno: 股票代號
        stype: 股票類型
        params: IchimokuParams 參數實例，預設使用 _DEFAULT_PARAMS
    """
    if params is None:
        params = _DEFAULT_PARAMS
    
    if df.empty:
        return df

    # 確保索引格式正確
    df.index = pd.to_datetime(df.index)
    
    # 初始化必要的欄位
    cols_to_init = [
        'ICHIMOKU', 'ICHIMOKU_SIGNAL', 'ICHIMOKU_STRENGTH',
        'tenkan_sen', 'kijun_sen', 'senkou_a', 'senkou_b',
        'cloud_top', 'cloud_bottom', 'tk_cross', 'RSI'
    ]
    for col in cols_to_init:
        if col not in df.columns:
            df[col] = np.nan if 'price' in col or 'sen' in col or 'cloud' in col else False
    
    # ==========================================
    # 步驟 1: 計算 Ichimoku 指標
    # ==========================================
    
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    # Tenkan-sen (轉換線): (9日最高 + 9日最低) / 2
    df['tenkan_sen'] = (_highest(high, params.TENKAN_PERIOD) + 
                        _lowest(low, params.TENKAN_PERIOD)) / 2
    
    # Kijun-sen (基準線): (26日最高 + 26日最低) / 2
    df['kijun_sen'] = (_highest(high, params.KIJUN_PERIOD) + 
                       _lowest(low, params.KIJUN_PERIOD)) / 2
    
    # Senkou Span A (先行區間A): (轉換線 + 基準線) / 2，往後位移
    df['senkou_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(params.DISPLACEMENT)
    
    # Senkou Span B (先行區間B): (52日最高 + 52日最低) / 2，往後位移
    senkou_b_raw = (_highest(high, params.SENKOU_B_PERIOD) + 
                     _lowest(low, params.SENKOU_B_PERIOD)) / 2
    df['senkou_b'] = pd.Series(senkou_b_raw, index=df.index).shift(params.DISPLACEMENT)
    
    # 雲圖
    df['cloud_top'] = df[['senkou_a', 'senkou_b']].max(axis=1)
    df['cloud_bottom'] = df[['senkou_a', 'senkou_b']].min(axis=1)
    
    # TK 交叉信號: 1=多頭, -1=空頭
    df['tk_cross'] = np.where(df['tenkan_sen'] > df['kijun_sen'], 1, -1)
    
    # ==========================================
    # 步驟 2: 生成 Ichimoku 信號
    # ==========================================
    
    # 計算 TK 交叉點
    tk_cross_prev = df['tk_cross'].shift(1)
    df['tk_golden'] = (tk_cross_prev <= 0) & (df['tk_cross'] > 0)  # 金叉
    df['tk_death'] = (tk_cross_prev >= 0) & (df['tk_cross'] < 0)   # 死叉
    
    # 成交量確認
    if 'Volume' in df.columns:
        df['volume_ma20'] = _sma(df['Volume'].values, 20)
        df['volume_ratio'] = df['Volume'] / df['volume_ma20']
    else:
        df['volume_ratio'] = 1.0
    
    # RSI 計算
    if params.RSI_FILTER:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/params.RSI_PERIOD, min_periods=params.RSI_PERIOD).mean()
        avg_loss = loss.ewm(alpha=1/params.RSI_PERIOD, min_periods=params.RSI_PERIOD).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
    else:
        df['RSI'] = 50  # 預設穿過中性值
    
    # ATR 計算
    if params.USE_ATR_STOP:
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        # 處理第一根K線（沒有前一天收盤價）
        tr = np.zeros(len(df))
        tr[0] = tr1[0]
        for i in range(1, len(df)):
            tr[i] = max(tr1[i], max(tr2[i], tr3[i]))
        
        # ATR (14週期平滑)
        atr = np.zeros(len(df))
        atr[params.ATR_PERIOD-1] = np.mean(tr[0:params.ATR_PERIOD])
        for i in range(params.ATR_PERIOD, len(df)):
            atr[i] = (atr[i-1] * (params.ATR_PERIOD - 1) + tr[i]) / params.ATR_PERIOD
        
        df['ATR'] = atr
        df['ATR_SL'] = atr * params.ATR_MULTIPLIER_SL  # 動態止損
        df['ATR_TP'] = atr * params.ATR_MULTIPLIER_TP  # 動態止盈
    
    # ==========================================
    # 步驟 3: 識別進場信號
    # ==========================================
    
    # 向量化計算信號
    price = df['Close'].values
    cloud_top = df['cloud_top'].values
    cloud_bottom = df['cloud_bottom'].values
    volume_ratio = df['volume_ratio'].values
    
    # 價格在雲圖上方
    above_cloud = np.where(pd.isna(cloud_top), False, price > cloud_top)
    # 價格在雲圖下方
    below_cloud = np.where(pd.isna(cloud_bottom), False, price < cloud_bottom)
    
    # 價格在雲圖中
    in_cloud = ~(above_cloud | below_cloud)
    
    # RSI 過濾：多頭需要RSI>50，空頭需要RSI<50
    rsi = df['RSI'].values
    rsi_bullish_filter = rsi > 50 if params.RSI_FILTER else True
    rsi_bearish_filter = rsi < 50 if params.RSI_FILTER else True
    
    # 多頭信號: TK金叉 + 價格在雲圖上方 + 成交量確認 + RSI過濾
    bullish_signal = df['tk_golden'].values & above_cloud & (volume_ratio >= params.VOLUME_CONFIRM) & rsi_bullish_filter
    
    # 空頭信號: TK死叉 + 價格在雲圖下方 + 成交量確認 + RSI過濾
    bearish_signal = df['tk_death'].values & below_cloud & (volume_ratio >= params.VOLUME_CONFIRM) & rsi_bearish_filter
    
    # 中性信號: TK金叉/死叉但價格在雲圖中
    tk_only_bullish = df['tk_golden'].values & in_cloud
    tk_only_bearish = df['tk_death'].values & in_cloud
    
    # 強度評估（簡化邏輯，減少假信號）
    # 強多頭: 金叉 + 雲上 + 放量
    strong_bullish = bullish_signal
    # 中多頭: 金叉 + 雲上（無量）- 移除無量雲中金叉信號以減少假信號
    medium_bullish = (df['tk_golden'].values & above_cloud & ~strong_bullish) & (volume_ratio >= params.VOLUME_CONFIRM) & rsi_bullish_filter
    # 弱多頭: 僅保留金叉 + 雲中 + 放量（移除無交叉的弱信號）
    weak_bullish = tk_only_bullish & (volume_ratio >= params.VOLUME_CONFIRM) & ~(medium_bullish | strong_bullish) & rsi_bullish_filter
    
    # 強空頭: 死叉 + 雲下 + 放量
    strong_bearish = bearish_signal
    # 中空頭: 死叉 + 雲下（無量）- 移除無量雲中死叉信號以減少假信號
    medium_bearish = (df['tk_death'].values & below_cloud & ~strong_bearish) & (volume_ratio >= params.VOLUME_CONFIRM) & rsi_bearish_filter
    # 弱空頭: 僅保留死叉 + 雲中 + 放量（移除無交叉的弱信號）
    weak_bearish = tk_only_bearish & (volume_ratio >= params.VOLUME_CONFIRM) & ~(medium_bearish | strong_bearish) & rsi_bearish_filter
    
    # 設置 ICHIMOKU 信號
    df['ICHIMOKU'] = False
    df['ICHIMOKU_SIGNAL'] = 'neutral'
    df['ICHIMOKU_STRENGTH'] = 'weak'
    
    df.loc[strong_bullish, 'ICHIMOKU'] = True
    df.loc[strong_bullish, 'ICHIMOKU_SIGNAL'] = 'bullish'
    df.loc[strong_bullish, 'ICHIMOKU_STRENGTH'] = 'strong'
    
    df.loc[medium_bullish, 'ICHIMOKU'] = True
    df.loc[medium_bullish, 'ICHIMOKU_SIGNAL'] = 'bullish'
    df.loc[medium_bullish, 'ICHIMOKU_STRENGTH'] = 'medium'
    
    df.loc[weak_bullish, 'ICHIMOKU'] = True
    df.loc[weak_bullish, 'ICHIMOKU_SIGNAL'] = 'bullish'
    df.loc[weak_bullish, 'ICHIMOKU_STRENGTH'] = 'weak'
    
    df.loc[strong_bearish, 'ICHIMOKU'] = True
    df.loc[strong_bearish, 'ICHIMOKU_SIGNAL'] = 'bearish'
    df.loc[strong_bearish, 'ICHIMOKU_STRENGTH'] = 'strong'
    
    df.loc[medium_bearish, 'ICHIMOKU'] = True
    df.loc[medium_bearish, 'ICHIMOKU_SIGNAL'] = 'bearish'
    df.loc[medium_bearish, 'ICHIMOKU_STRENGTH'] = 'medium'
    
    df.loc[weak_bearish, 'ICHIMOKU'] = True
    df.loc[weak_bearish, 'ICHIMOKU_SIGNAL'] = 'bearish'
    df.loc[weak_bearish, 'ICHIMOKU_STRENGTH'] = 'weak'
    
    return df
