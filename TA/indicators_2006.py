#!/usr/bin/env python3
"""
2006 Class F Formula - MetaTrader Indicators in Python
========================================================
根據 MetaStock .mwt 檔案名稱逆向工程的技術指標庫
包含: Ichimoku, GBS22C, 200-day Breakout, Fisher Transform, Trailing Stop
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

# ============================================================
# Indicator Base Class
# ============================================================

@dataclass
class IndicatorSignal:
    """單個指標信號"""
    name: str
    value: float
    signal: str  # 'bullish', 'bearish', 'neutral'
    strength: str  # 'strong', 'medium', 'weak'
    price: float = 0

@dataclass
class TradingSignal:
    """完整交易信號"""
    date: any
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD', 'CLOSE_LONG', 'CLOSE_SHORT'
    price: float
    stop_loss: float
    take_profit: float
    trailing_stop: float
    reason: str
    indicators: dict


# ============================================================
# Utility Functions
# ============================================================

def highest(data: np.ndarray, period: int) -> np.ndarray:
    """移動最高值"""
    result = np.full_like(data, np.nan)
    for i in range(period - 1, len(data)):
        result[i] = np.max(data[i - period + 1:i + 1])
    return result

def lowest(data: np.ndarray, period: int) -> np.ndarray:
    """移動最低值"""
    result = np.full_like(data, np.nan)
    for i in range(period - 1, len(data)):
        result[i] = np.min(data[i - period + 1:i + 1])
    return result

def sma(data: np.ndarray, period: int) -> np.ndarray:
    """簡單移動平均"""
    result = np.full_like(data, np.nan)
    for i in range(period - 1, len(data)):
        result[i] = np.mean(data[i - period + 1:i + 1])
    return result

def ema(data: np.ndarray, period: int) -> np.ndarray:
    """指數移動平均"""
    result = np.full_like(data, np.nan)
    result[period - 1] = np.mean(data[:period])
    multiplier = 2 / (period + 1)
    for i in range(period, len(data)):
        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
    return result

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """平均真實波幅 (Average True Range)"""
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    tr = np.concatenate([[np.nan], tr])
    return ema(tr, period)

def atr_series(high, low, close, period=14):
    """ATR for pandas Series"""
    tr1 = high - low
    tr2 = np.abs(high - close.shift())
    tr3 = np.abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ============================================================
# Ichimoku Cloud Indicator
# ============================================================

class IchimokuIndicator:
    """
    一目均衡表 (Ichimoku Cloud)
    
    組件:
    - Tenkan-sen (轉換線): (9日最高 + 9日最低) / 2
    - Kijun-sen (基準線): (26日最高 + 26日最低) / 2
    - Senkou Span A (先行區間A): (轉換線 + 基準線) / 2
    - Senkou Span B (先行區間B): (52日最高 + 52日最低) / 2
    - Chikou Span (延遲區間): 收盤價往後移26天
    """
    
    def __init__(self, 
                 tenkan: int = 9, 
                 kijun: int = 26, 
                 senkou_b: int = 52, 
                 displacement: int = 26):
        self.tenkan_period = tenkan
        self.kijun_period = kijun
        self.senkou_b_period = senkou_b
        self.displacement = displacement
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算 Ichimoku 所有指標"""
        df = df.copy()
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Tenkan-sen (轉換線)
        df['tenkan_sen'] = (highest(high, self.tenkan_period) + 
                            lowest(low, self.tenkan_period)) / 2
        
        # Kijun-sen (基準線)
        df['kijun_sen'] = (highest(high, self.kijun_period) + 
                          lowest(low, self.kijun_period)) / 2
        
        # Senkou Span A (先行區間A) - 往後位移
        df['senkou_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(self.displacement)
        
        # Senkou Span B (先行區間B) - 往後位移
        df['senkou_b'] = (highest(high, self.senkou_b_period) + 
                         lowest(low, self.senkou_b_period)).shift(self.displacement) / 2
        
        # 雲圖
        df['cloud_top'] = df[['senkou_a', 'senkou_b']].max(axis=1)
        df['cloud_bottom'] = df[['senkou_a', 'senkou_b']].min(axis=1)
        
        # Chikou Span (延遲區間) - 往前位移（實際是往後看）
        df['chikou_span'] = close
        
        # TK 交叉信號
        df['tk_cross'] = np.where(df['tenkan_sen'] > df['kijun_sen'], 1, -1)
        
        return df
    
    def get_signal(self, row: pd.Series) -> Tuple[str, str]:
        """取得單根K線信號"""
        price = row.get('close', 0)
        tenkan = row.get('tenkan_sen', 0)
        kijun = row.get('kijun_sen', 0)
        cloud_top = row.get('cloud_top', 0)
        cloud_bottom = row.get('cloud_bottom', 0)
        
        if pd.isna(tenkan) or pd.isna(kijun):
            return 'neutral', 'weak'
        
        # TK 交叉
        tk_bullish = tenkan > kijun
        tk_bearish = tenkan < kijun
        
        # 價格與雲圖關係
        above_cloud = price > cloud_top if not pd.isna(cloud_top) else False
        below_cloud = price < cloud_bottom if not pd.isna(cloud_bottom) else False
        
        # 綜合判斷
        if tk_bullish and above_cloud:
            return 'bullish', 'strong'
        elif tk_bearish and below_cloud:
            return 'bearish', 'strong'
        elif tk_bullish:
            return 'bullish', 'medium'
        elif tk_bearish:
            return 'bearish', 'medium'
        return 'neutral', 'weak'


# ============================================================
# GBS22C - Gann-Based System with 22 Period
# ============================================================

class GBS22C:
    """
    GBS22C - Gann-Based System
    
    原理:
    - 使用 Gann 角度線和 22 週期移動平均
    - 結合波動率調整
    - 識別支撐/阻力位
    """
    
    def __init__(self, period: int = 22):
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算 GBS22C 指標"""
        df = df.copy()
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # 22週期移動平均
        df['gbs_ma'] = sma(close, self.period)
        
        # Gann 角度線 (基于价格*时间关系)
        # 1x1 Gann角度 = 45度
        df['gann_angle'] = np.arctan2(close - df['gbs_ma'], np.arange(len(close)) + 1) * 180 / np.pi
        
        # 波動率調整
        df['volatility'] = atr(high, low, close, 14) / close * 100
        
        # 支撐/阻力位 (Gann 扇形)
        df['gann_support'] = df['gbs_ma'] - 2 * atr(high, low, close, 14)
        df['gann_resistance'] = df['gbs_ma'] + 2 * atr(high, low, close, 14)
        
        # Trend direction
        df['trend'] = np.where(close > df['gbs_ma'], 1, -1)
        
        # 22-period high/low
        df['period_high'] = highest(high, self.period)
        df['period_low'] = lowest(low, self.period)
        
        # Breakout signals
        df['breakout_up'] = close > df['period_high'].shift(1)
        df['breakout_down'] = close < df['period_low'].shift(1)
        
        return df
    
    def get_signal(self, row: pd.Series, prev_row: Optional[pd.Series] = None) -> Tuple[str, str]:
        """取得 GBS22C 信號"""
        if pd.isna(row.get('gbs_ma', np.nan)):
            return 'neutral', 'weak'
        
        trend = row.get('trend', 0)
        breakout_up = row.get('breakout_up', False)
        breakout_down = row.get('breakout_down', False)
        volatility = row.get('volatility', 0)
        
        # High volatility breakout
        if volatility > 2.0:
            if breakout_up:
                return 'bullish', 'strong'
            elif breakout_down:
                return 'bearish', 'strong'
        
        # Trend following
        if trend == 1:
            return 'bullish', 'medium'
        elif trend == -1:
            return 'bearish', 'medium'
        
        return 'neutral', 'weak'


# ============================================================
# 200-Day New High Breakout Strategy
# ============================================================

class Breakout200Days:
    """
    200日新高突破策略
    
    原理:
    - 價格創200日新高時買入
    - 價格破200日新低時賣出/做空
    - 結合成交量確認
    """
    
    def __init__(self, period: int = 200):
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算 200日突破指標"""
        df = df.copy()
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df.get('volume', pd.Series([1]*len(df))).values
        
        # 200日移動窗口內的最高價和最低價
        df['highest_200'] = highest(high, self.period)
        df['lowest_200'] = lowest(low, self.period)
        
        # 200日均線
        df['ma200'] = sma(close, self.period)
        
        # 價格相對200日均線位置
        df['price_vs_ma200'] = (close - df['ma200']) / df['ma200'] * 100
        
        # 成交量移動平均
        if 'volume' in df.columns:
            df['volume_ma20'] = sma(volume, 20)
            df['volume_ratio'] = volume / df['volume_ma20']
        else:
            df['volume_ratio'] = 1.0
        
        # 新高新價
        df['new_high_200'] = close >= df['highest_200'].shift(1)
        df['new_low_200'] = close <= df['lowest_200'].shift(1)
        
        # 突破強度
        df['breakout_strength'] = close - df['highest_200'].shift(1)
        
        return df
    
    def get_signal(self, row: pd.Series, prev_row: Optional[pd.Series] = None) -> Tuple[str, str]:
        """取得突破信號"""
        if pd.isna(row.get('ma200', np.nan)):
            return 'neutral', 'weak'
        
        new_high = row.get('new_high_200', False)
        new_low = row.get('new_low_200', False)
        volume_ratio = row.get('volume_ratio', 1.0)
        price_vs_ma = row.get('price_vs_ma200', 0)
        
        # 放量突破200日新高
        if new_high and volume_ratio > 1.5:
            return 'bullish', 'strong'
        
        # 放量跌破200日新低
        if new_low and volume_ratio > 1.5:
            return 'bearish', 'strong'
        
        # 無量突破
        if new_high:
            return 'bullish', 'medium'
        
        if new_low:
            return 'bearish', 'medium'
        
        # 價格在均線上方震盪
        if price_vs_ma > 5:
            return 'bullish', 'weak'
        elif price_vs_ma < -5:
            return 'bearish', 'weak'
        
        return 'neutral', 'weak'


# ============================================================
# Fisher Transform Indicator
# ============================================================

class FisherTransform:
    """
    Fisher Transform (Fisher 轉換)
    
    原理:
    - 將價格轉換為常態分佈的信號
    - 識別價格轉折點
    - 觸發阈值: +1 (多頭) / -1 (空頭)
    """
    
    def __init__(self, period: int = 10):
        self.period = period
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算 Fisher Transform"""
        df = df.copy()
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # 計算 HL2 價格
        hl2 = (high + low) / 2
        
        # 標準化價格範圍
        max_hl = highest(hl2, self.period)
        min_hl = lowest(hl2, self.period)
        
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
            fish[i] = 0.5 * fish[i-1] + 0.5 * (2.5 * np.log((1 + v[i]) / (1 - v[i])))
        
        df['fisher'] = fish
        df['fisher_signal'] = np.where(fish > 0, 1, -1)
        
        # 交叉信號
        df['fisher_cross'] = np.where(fish > 0, 1, -1)
        df['fisher_cross_prev'] = df['fisher_cross'].shift(1)
        
        # Trigger lines
        df['fisher_trigger_bull'] = (df['fisher'] > 0) & (df['fisher_cross_prev'] <= 0)
        df['fisher_trigger_bear'] = (df['fisher'] < 0) & (df['fisher_cross_prev'] >= 0)
        
        return df
    
    def get_signal(self, row: pd.Series, prev_row: Optional[pd.Series] = None) -> Tuple[str, str]:
        """取得 Fisher Transform 信號"""
        if pd.isna(row.get('fisher', np.nan)):
            return 'neutral', 'weak'
        
        fisher = row.get('fisher', 0)
        fisher_trigger_bull = row.get('fisher_trigger_bull', False)
        fisher_trigger_bear = row.get('fisher_trigger_bear', False)
        
        if fisher_trigger_bull:
            return 'bullish', 'strong'
        if fisher_trigger_bear:
            return 'bearish', 'strong'
        
        if fisher > 0.5:
            return 'bullish', 'medium'
        elif fisher < -0.5:
            return 'bearish', 'medium'
        
        return 'neutral', 'weak'


# ============================================================
# 2006 Trailing Stop
# ============================================================

class TrailingStop2006:
    """
    2006 追踪止損策略
    
    原理:
    - 使用 ATR 計算追踪止損
    - 多頭: 價格 - N * ATR
    - 空頭: 價格 + N * ATR
    """
    
    def __init__(self, atr_period: int = 14, multiplier: float = 2.0):
        self.atr_period = atr_period
        self.multiplier = multiplier
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算追踪止損"""
        df = df.copy()
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # ATR
        df['atr'] = atr(high, low, close, self.atr_period)
        
        # 追踪止損線
        df['trail_stop_long'] = close - self.multiplier * df['atr']
        df['trail_stop_short'] = close + self.multiplier * df['atr']
        
        # 填充 NaN
        df['trail_stop_long'] = df['trail_stop_long'].fillna(method='bfill')
        df['trail_stop_short'] = df['trail_short_long'].fillna(method='bfill') if 'trail_short_long' in df.columns else df['trail_stop_short'].fillna(method='bfill')
        
        # 根據趨勢調整
        df['in_long'] = False
        df['in_short'] = False
        
        return df
    
    def get_stop(self, row: pd.Series, position: str, entry_price: float) -> float:
        """取得止損價"""
        atr = row.get('atr', 0)
        close = row.get('close', entry_price)
        
        if position == 'long':
            stop = close - self.multiplier * atr
            return max(stop, row.get('trail_stop_long', stop))
        elif position == 'short':
            stop = close + self.multiplier * atr
            return min(stop, row.get('trail_stop_short', stop))
        
        return entry_price


# ============================================================
# Combined Strategy: Ichimoku + GBS22C + 200d Breakout
# ============================================================

class CombinedStrategy2006:
    """
    2006 Class F 組合策略
    
    結合:
    1. Ichimoku (趨勢確認)
    2. GBS22C (進場時機)
    3. 200日突破 (動量過濾)
    4. Fisher Transform (轉折確認)
    """
    
    def __init__(self):
        self.ichimoku = IchimokuIndicator(9, 26, 52, 26)
        self.gbs = GBS22C(22)
        self.breakout = Breakout200Days(200)
        self.fisher = FisherTransform(10)
        self.trailing = TrailingStop2006(14, 2.0)
        
        # Position tracking
        self.position = 0  # 1=long, -1=short, 0=flat
        self.entry_price = 0
        self.entry_date = None
        self.stop_loss = 0
        self.take_profit = 0
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算所有指標"""
        df = self.ichimoku.calculate(df)
        df = self.gbs.calculate(df)
        df = self.breakout.calculate(df)
        df = self.fisher.calculate(df)
        df = self.trailing.calculate(df)
        
        return df
    
    def get_combined_signal(self, row: pd.Series, prev_row: Optional[pd.Series] = None) -> Tuple[str, str]:
        """綜合所有指標給出信號"""
        if prev_row is None:
            return 'neutral', 'weak'
        
        # 各指標信號
        ichi_signal, ichi_strength = self.ichimoku.get_signal(row)
        gbs_signal, gbs_strength = self.gbs.get_signal(row, prev_row)
        breakout_signal, breakout_strength = self.breakout.get_signal(row, prev_row)
        fisher_signal, fisher_strength = self.fisher.get_signal(row, prev_row)
        
        # 計分系統
        bullish_score = 0
        bearish_score = 0
        
        # Ichimoku (最重要)
        if ichi_signal == 'bullish':
            bullish_score += 3 if ichi_strength == 'strong' else 2
        elif ichi_signal == 'bearish':
            bearish_score += 3 if ichi_strength == 'strong' else 2
        
        # GBS22C
        if gbs_signal == 'bullish':
            bullish_score += 2 if gbs_strength == 'strong' else 1
        elif gbs_signal == 'bearish':
            bearish_score += 2 if gbs_strength == 'strong' else 1
        
        # 200日突破
        if breakout_signal == 'bullish':
            bullish_score += 2 if breakout_strength == 'strong' else 1
        elif breakout_signal == 'bearish':
            bearish_score += 2 if breakout_strength == 'strong' else 1
        
        # Fisher
        if fisher_signal == 'bullish':
            bullish_score += 1
        elif fisher_signal == 'bearish':
            bearish_score += 1
        
        # 判斷
        if bullish_score >= 5 and bullish_score > bearish_score + 2:
            return 'bullish', 'strong'
        elif bearish_score >= 5 and bearish_score > bullish_score + 2:
            return 'bearish', 'strong'
        elif bullish_score > bearish_score:
            return 'bullish', 'medium'
        elif bearish_score > bullish_score:
            return 'bearish', 'medium'
        
        return 'neutral', 'weak'
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信號"""
        df = self.calculate_all(df)
        
        signals = []
        for i in range(len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1] if i > 0 else None
            
            signal, strength = self.get_combined_signal(row, prev_row)
            df.at[df.index[i], 'combined_signal'] = signal
            df.at[df.index[i], 'signal_strength'] = strength
        
        return df
    
    def should_entry_long(self, row: pd.Series, prev_row: pd.Series) -> bool:
        """判斷是否應該做多"""
        # TK金叉 + 價格在雲圖上方
        tk_golden = (prev_row['tenkan_sen'] <= prev_row['kijun_sen'] and 
                    row['tenkan_sen'] > row['kijun_sen'])
        above_cloud = row['close'] > row['cloud_top'] if not pd.isna(row.get('cloud_top')) else False
        
        # 200日新高確認
        new_high_200 = row.get('new_high_200', False)
        
        return tk_golden and above_cloud and new_high_200
    
    def should_entry_short(self, row: pd.Series, prev_row: pd.Series) -> bool:
        """判斷是否應該做空"""
        # TK死叉 + 價格在雲圖下方
        tk_death = (prev_row['tenkan_sen'] >= prev_row['kijun_sen'] and 
                   row['tenkan_sen'] < row['kijun_sen'])
        below_cloud = row['close'] < row['cloud_bottom'] if not pd.isna(row.get('cloud_bottom')) else False
        
        # 200日新低確認
        new_low_200 = row.get('new_low_200', False)
        
        return tk_death and below_cloud and new_low_200
    
    def should_exit(self, row: pd.Series) -> Tuple[bool, str]:
        """判斷是否應該平倉"""
        # 止損
        if self.position == 1 and row['close'] < self.stop_loss:
            return True, 'stop_loss'
        elif self.position == -1 and row['close'] > self.stop_loss:
            return True, 'stop_loss'
        
        # 止盈
        if self.position == 1 and row['close'] > self.take_profit:
            return True, 'take_profit'
        elif self.position == -1 and row['close'] < self.take_profit:
            return True, 'take_profit'
        
        # TK交叉反向
        if self.position == 1:
            if row['tenkan_sen'] < row['kijun_sen']:
                return True, 'tk_death_cross'
        elif self.position == -1:
            if row['tenkan_sen'] > row['kijun_sen']:
                return True, 'tk_golden_cross'
        
        return False, ''


# ============================================================
# Backtest Engine for Combined Strategy
# ============================================================

class StrategyBacktester:
    """策略回測引擎"""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.strategy = CombinedStrategy2006()
    
    def run(self, df: pd.DataFrame) -> dict:
        """執行回測"""
        df = df.copy()
        df = self.strategy.calculate_all(df)
        
        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = [capital]
        
        for i in range(50, len(df)):  # 需要足夠數據計算指標
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            date = df.index[i] if hasattr(df.index[i], 'strftime') else i
            
            current_price = row['close']
            
            # === 平倉檢查 ===
            if position != 0:
                exit_now, reason = self.strategy.should_exit(row)
                
                if exit_now:
                    if position == 1:  # 平多
                        pnl = (current_price - self.strategy.entry_price) * 1
                    else:  # 平空
                        pnl = (self.strategy.entry_price - current_price) * 1
                    
                    pnl -= current_price * self.commission * 2  # 來回手續費
                    capital += pnl
                    
                    trades.append({
                        'entry_date': self.strategy.entry_date,
                        'entry_price': self.strategy.entry_price,
                        'exit_date': date,
                        'exit_price': current_price,
                        'direction': 'LONG' if position == 1 else 'SHORT',
                        'pnl': pnl,
                        'pnl_pct': pnl / self.strategy.entry_price * 100,
                        'reason': reason
                    })
                    
                    position = 0
            
            # === 建倉檢查 ===
            if position == 0:
                # 嘗試做多
                if self.strategy.should_entry_long(row, prev_row):
                    position = 1
                    self.strategy.entry_price = current_price * (1 + self.commission)
                    self.strategy.entry_date = date
                    
                    atr_val = row.get('atr', current_price * 0.02)
                    self.strategy.stop_loss = current_price - 2 * atr_val
                    self.strategy.take_profit = current_price + 3 * atr_val
                
                # 嘗試做空
                elif self.strategy.should_entry_short(row, prev_row):
                    position = -1
                    self.strategy.entry_price = current_price * (1 - self.commission)
                    self.strategy.entry_date = date
                    
                    atr_val = row.get('atr', current_price * 0.02)
                    self.strategy.stop_loss = current_price + 2 * atr_val
                    self.strategy.take_profit = current_price - 3 * atr_val
            
            # 記錄equity
            if position != 0:
                if position == 1:
                    current_equity = capital + (current_price - self.strategy.entry_price)
                else:
                    current_equity = capital + (self.strategy.entry_price - current_price)
            else:
                current_equity = capital
            
            equity_curve.append(current_equity)
        
        # 統計結果
        if trades:
            winning = [t for t in trades if t['pnl'] > 0]
            losing = [t for t in trades if t['pnl'] <= 0]
            
            return {
                'total_trades': len(trades),
                'winning_trades': len(winning),
                'losing_trades': len(losing),
                'win_rate': len(winning) / len(trades) * 100,
                'total_pnl': capital - self.initial_capital,
                'total_pnl_pct': (capital - self.initial_capital) / self.initial_capital * 100,
                'avg_win': np.mean([t['pnl'] for t in winning]) if winning else 0,
                'avg_loss': np.mean([t['pnl'] for t in losing]) if losing else 0,
                'profit_factor': abs(sum(t['pnl'] for t in winning) / sum(t['pnl'] for t in losing)) if losing else float('inf'),
                'trades': trades,
                'equity_curve': equity_curve
            }
        
        return {'total_trades': 0, 'trades': [], 'equity_curve': equity_curve}


# ============================================================
# Example Usage
# ============================================================

def example_usage():
    """使用範例"""
    
    # 創建策略
    strategy = CombinedStrategy2006()
    
    # 創建回測引擎
    backtester = StrategyBacktester(initial_capital=10000)
    
    # 計算指標
    print("=" * 60)
    print("    2006 Class F Formula - Python 版本")
    print("    Ichimoku + GBS22C + 200d Breakout + Fisher")
    print("=" * 60)
    
    print("\n[1] 指標類別:")
    print("    - IchimokuIndicator: 一目均衡表")
    print("    - GBS22C: Gann-Based System (22週期)")
    print("    - Breakout200Days: 200日新高突破")
    print("    - FisherTransform: Fisher 轉換")
    print("    - TrailingStop2006: 追踪止損")
    print("    - CombinedStrategy2006: 組合策略")
    
    print("\n[2] 使用方法:")
    print("""
    # 載入數據
    import pandas as pd
    df = pd.read_csv('your_data.csv')
    
    # 計算所有指標
    strategy = CombinedStrategy2006()
    df = strategy.calculate_all(df)
    
    # 生成信號
    df = strategy.generate_signals(df)
    
    # 回測
    backtester = StrategyBacktester(initial_capital=10000)
    results = backtester.run(df)
    
    # 查看結果
    print(results['total_trades'])
    print(results['win_rate'])
    print(results['total_pnl'])
    """)
    
    print("\n[3] 檔案對應關係:")
    print("    2006 ichi + GBS22C.mwt -> CombinedStrategy2006 (Ichimoku + GBS22C)")
    print("    2006 200 days new high.mwt -> Breakout200Days")
    print("    2006 F to T.mwt -> FisherTransform")
    print("    2006 Trailing Stop -> TrailingStop2006")


if __name__ == '__main__':
    example_usage()
