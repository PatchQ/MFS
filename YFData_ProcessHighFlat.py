import pandas as pd
import numpy as np
import yfinance as yf
import concurrent.futures as cf
import os
from tqdm import tqdm
import time as t
from datetime import datetime
import matplotlib.pyplot as plt
import mplfinance as mpf


PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/" 

"""
參數:
DAYS (list): 多個時間範圍列表
RESISTANCE_RATE (float): 阻力位識別閾值
BREAKOUT_RATE (float): 突破檢查閾值
MIN_PEAKS (int): 峰值之間的最小天數
"""
DAYS=100
RESISTANCE_RATE=0.004
BREAKOUT_RATE=0.004
MIN_PEAKS=10
WINDOW=10

SLIDING_WINDOW = False
WINDOW_DAYS = 60
STEP_DAYS = 20


# 設定中文字體
def set_chinese_font():
    """
    設定 matplotlib 使用支援中文的字體
    """
    try:
        # 嘗試使用系統中的中文字體
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']        
        plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

        print("已設定中文字體")
    except:
        print("無法設定中文字體，將使用預設字體")



def convertData(df):

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=numeric_cols, how='all', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def CheckEMA(df):

    try:
        # Volatility indicators
        df['EMA10'] = df['Close'].ewm(span=10, min_periods=5, adjust=False).mean()
        df['EMA22'] = df['Close'].ewm(span=22, min_periods=11, adjust=False).mean()     
        df['EMA50'] = df['Close'].ewm(span=50, min_periods=25, adjust=False).mean()     
        df['EMA100'] = df['Close'].ewm(span=100, min_periods=50, adjust=False).mean()             

        df['EMA'] = ((df["EMA10"] > df["EMA22"]) & (df["EMA22"] > df["EMA50"]) & (df["EMA50"] > df["EMA100"]))        

        return df['EMA'].iloc[-1]        

    except Exception as e:
        print(f"Indicator error: {str(e)}")
        return False


def find_bullish_then_consolidation_stocks(ticker_list, min_bullish_days=5, consolidation_days=5, 
                                         consolidation_threshold=0.03, lookback_period="3mo"):
    """
    找出至少以5支阳烛上升后横行的股票
    
    参数:
    ticker_list: 股票代码列表
    min_bullish_days: 最少连续阳烛天数
    consolidation_days: 横盘整理天数
    consolidation_threshold: 横盘波动阈值 (百分比)
    lookback_period: 数据回顾周期
    """
    
    matched_stocks = []
    
    for ticker in ticker_list:
        try:
            print(f"分析 {ticker}...")
            
            # 下载股票数据
            df = yf.Ticker(ticker)
            stock_data = df.history(period=lookback_period,auto_adjust=True)

            
            if stock_data.empty or len(stock_data) < (min_bullish_days + consolidation_days):
                continue
            
            # 数据清洗
            stock_data = stock_data.dropna()
            
            # 分析模式
            result = analyze_bullish_consolidation_pattern(
                stock_data, 
                min_bullish_days, 
                consolidation_days, 
                consolidation_threshold
            )
            
            if result['found']:
                matched_stocks.append({
                    'ticker': ticker,
                    'bullish_start': result['bullish_start'],
                    'bullish_end': result['bullish_end'],
                    'consolidation_end': result['consolidation_end'],
                    'bullish_return': result['bullish_return'],
                    'consolidation_range': result['consolidation_range'],
                    'current_price': stock_data['Close'].iloc[-1],
                    'pattern_strength': result['pattern_strength']
                })
                print(f"  ✓ 找到匹配模式: {ticker}")
                
        except Exception as e:
            print(f"  处理 {ticker} 时出错: {str(e)}")
            continue
    
    return matched_stocks

def analyze_bullish_consolidation_pattern(stock_data, min_bullish_days, consolidation_days, threshold):
    """
    分析阳烛上升后横行的模式
    """
    # 计算技术指标
    stock_data = calculate_technical_indicators(stock_data)
    
    # 寻找连续阳烛模式
    bullish_periods = find_consecutive_bullish_candles(stock_data, min_bullish_days)
    
    for bullish_period in bullish_periods:
        bullish_end = bullish_period['end_index']
        
        # 检查是否有足够的后续数据进行横盘分析
        if bullish_end + consolidation_days >= len(stock_data):
            continue
        
        # 分析横盘整理
        consolidation_data = stock_data.iloc[bullish_end:bullish_end + consolidation_days]
        consolidation_result = analyze_consolidation(consolidation_data, threshold)
        
        if consolidation_result['is_consolidating']:
            # 计算模式强度
            pattern_strength = calculate_pattern_strength(
                bullish_period, consolidation_result, stock_data
            )
            
            return {
                'found': True,
                'bullish_start': stock_data.index[bullish_period['start_index']],
                'bullish_end': stock_data.index[bullish_period['end_index']],
                'consolidation_end': stock_data.index[bullish_end + consolidation_days - 1],
                'bullish_return': bullish_period['total_return'],
                'consolidation_range': consolidation_result['price_range_pct'],
                'pattern_strength': pattern_strength
            }
    
    return {'found': False}

def calculate_technical_indicators(df):
    """
    计算技术指标
    """
    df = df.copy()
    
    # 标记阳烛 (收盘价 > 开盘价)
    df['Bullish'] = df['Close'] > df['Open']
    
    # 计算价格变化
    df['Price_Change'] = df['Close'].pct_change()
    
    # 计算移动平均线
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    
    # 计算价格波动率
    df['Volatility'] = df['High'].rolling(window=5).std() / df['Close'].rolling(window=5).mean()
    
    return df

def find_consecutive_bullish_candles(df, min_days):
    """
    寻找连续阳烛
    """
    bullish_periods = []
    current_streak = 0
    streak_start = None
    
    for i in range(len(df)):
        if df['Bullish'].iloc[i]:
            if current_streak == 0:
                streak_start = i
            current_streak += 1
        else:
            if current_streak >= min_days:
                # 计算这个连续阳烛期间的总回报
                start_price = df['Close'].iloc[streak_start]
                end_price = df['Close'].iloc[i-1]
                total_return = (end_price - start_price) / start_price
                
                bullish_periods.append({
                    'start_index': streak_start,
                    'end_index': i-1,
                    'length': current_streak,
                    'total_return': total_return
                })
            
            current_streak = 0
            streak_start = None
    
    # 检查最后一个序列
    if current_streak >= min_days:
        start_price = df['Close'].iloc[streak_start]
        end_price = df['Close'].iloc[-1]
        total_return = (end_price - start_price) / start_price
        
        bullish_periods.append({
            'start_index': streak_start,
            'end_index': len(df) - 1,
            'length': current_streak,
            'total_return': total_return
        })
    
    return bullish_periods

def analyze_consolidation(df, threshold):
    """
    分析横盘整理
    """
    if len(df) == 0:
        return {'is_consolidating': False}
    
    # 计算价格范围
    highest_high = df['High'].max()
    lowest_low = df['Low'].min()
    price_range = highest_high - lowest_low
    avg_price = df['Close'].mean()
    price_range_pct = price_range / avg_price
    
    # 计算价格波动率
    volatility = df['Close'].std() / df['Close'].mean()
    
    # 检查是否横盘 (价格范围小)
    is_consolidating = price_range_pct <= threshold
    
    return {
        'is_consolidating': is_consolidating,
        'price_range_pct': price_range_pct,
        'volatility': volatility,
        'highest_high': highest_high,
        'lowest_low': lowest_low
    }

def calculate_pattern_strength(bullish_period, consolidation_result, stock_data):
    """
    计算模式强度 (0-10分)
    """
    score = 0
    
    # 1. 阳烛序列长度 (最高3分)
    bullish_length_score = min(bullish_period['length'] / 5, 3)
    score += bullish_length_score
    
    # 2. 阳烛期间涨幅 (最高3分)
    return_score = min(bullish_period['total_return'] * 100, 3)  # 每1%得1分，最高3分
    score += return_score
    
    # 3. 横盘整理质量 (最高4分)
    consolidation_quality = max(0, 4 - (consolidation_result['price_range_pct'] * 400))
    score += consolidation_quality
    
    # 确保分数在0-10之间
    return min(score, 10)

def visualize_pattern(ticker, stock_data, pattern_info):
    """
    可视化阳烛上升后横行的模式
    """
    # 确定显示范围
    start_idx = max(0, pattern_info['bullish_start_index'] - 5)
    end_idx = min(len(stock_data) - 1, pattern_info['consolidation_end_index'] + 5)
    
    display_data = stock_data.iloc[start_idx:end_idx+1].copy()
    
    # 准备额外图表元素
    add_plots = []
    
    # 标记阳烛上升期间
    bullish_start_idx = pattern_info['bullish_start_index'] - start_idx
    bullish_end_idx = pattern_info['bullish_end_index'] - start_idx
    
    # 创建标记序列
    bullish_marker = pd.Series(index=display_data.index, dtype=float)
    for i in range(bullish_start_idx, bullish_end_idx + 1):
        if i < len(display_data):
            bullish_marker.iloc[i] = display_data['Low'].iloc[i] * 0.98
    
    bullish_plot = mpf.make_addplot(bullish_marker, type='scatter', markersize=50,
                                   marker='^', color='green', label='阳烛上升')
    add_plots.append(bullish_plot)
    
    # 标记横盘期间
    consolidation_start_idx = pattern_info['bullish_end_index'] + 1 - start_idx
    consolidation_end_idx = pattern_info['consolidation_end_index'] - start_idx
    
    consolidation_marker = pd.Series(index=display_data.index, dtype=float)
    for i in range(consolidation_start_idx, consolidation_end_idx + 1):
        if i < len(display_data):
            consolidation_marker.iloc[i] = display_data['Low'].iloc[i] * 0.96
    
    consolidation_plot = mpf.make_addplot(consolidation_marker, type='scatter', markersize=50,
                                         marker='s', color='blue', label='横盘整理')
    add_plots.append(consolidation_plot)
    
    # 绘制支撑阻力线
    resistance_level = pattern_info['consolidation_high']
    support_level = pattern_info['consolidation_low']
    
    # 设置图表样式
    mc = mpf.make_marketcolors(
        up='red',
        down='green',
        edge='black',
        wick='black',
        volume='in'
    )
    
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', gridcolor='lightgray',
                                rc={
                                    'font.family': 'Microsoft YaHei',
                                    'axes.unicode_minus': False
                                    }
                                )

    # 绘制图表
    title = f'{ticker} 陽燭上升後橫盤模式 (强度: {pattern_info["pattern_strength"]:.1f}/10)'
    
    fig, axes = mpf.plot(
        display_data[['Open', 'High', 'Low', 'Close', 'Volume']],
        type='candle',
        style=style,
        addplot=add_plots,
        title=title,
        ylabel='价格',
        volume=True,
        figsize=(15, 10),
        returnfig=True,
        hlines=dict(
            hlines=[resistance_level, support_level],
            colors=['red', 'green'],
            linestyle=['--', '--'],
            alpha=[0.7, 0.7]
        )
    )
    
    # 添加图例
    axes[0].legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

def scan_and_visualize_top_patterns(ticker_list, top_n=5, **kwargs):
    """
    扫描并可视化前N个最佳模式
    """
    # 扫描股票
    results = find_bullish_then_consolidation_stocks(ticker_list, **kwargs)
    
    if not results:
        print("未找到符合条件的模式")
        return
    
    # 按模式强度排序
    results.sort(key=lambda x: x['pattern_strength'], reverse=True)
    
    print(f"\n找到 {len(results)} 个符合条件的模式:")
    for i, result in enumerate(results[:top_n]):
        print(f"{i+1}. {result['ticker']}: 强度 {result['pattern_strength']:.1f}/10, "
              f"阳烛涨幅 {result['bullish_return']:.2%}, 横盘幅度 {result['consolidation_range']:.2%}")
    
    # 可视化前N个
    for i, result in enumerate(results[:top_n]):
        print(f"\n可视化 {result['ticker']}...")
        
        # 重新下载数据以获取详细信息
        stock_data = yf.download(result['ticker'], period=kwargs.get('lookback_period', '3mo'))

        df = yf.Ticker(result['ticker'])
        stock_data = df.history(period="3mo",auto_adjust=True)

        
        # 重新分析以获取详细模式信息
        pattern_result = analyze_bullish_consolidation_pattern(
            stock_data,
            kwargs.get('min_bullish_days', 5),
            kwargs.get('consolidation_days', 5),
            kwargs.get('consolidation_threshold', 0.03)
        )
        
        if pattern_result['found']:
            # 添加索引信息用于可视化
            pattern_result.update({
                'bullish_start_index': stock_data.index.get_loc(pattern_result['bullish_start']),
                'bullish_end_index': stock_data.index.get_loc(pattern_result['bullish_end']),
                'consolidation_end_index': stock_data.index.get_loc(pattern_result['consolidation_end']),
                'consolidation_high': stock_data['High'].loc[pattern_result['bullish_end']:pattern_result['consolidation_end']].max(),
                'consolidation_low': stock_data['Low'].loc[pattern_result['bullish_end']:pattern_result['consolidation_end']].min()
            })
            
            visualize_pattern(result['ticker'], stock_data, pattern_result)

# 使用示例
if __name__ == "__main__":
    # 示例股票列表
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'META', 'NVDA', 'AMD', 'AMZN']
    
    # 扫描参数
    scan_params = {
        'min_bullish_days': 5,           # 最少5支阳烛
        'consolidation_days': 5,         # 横盘5天
        'consolidation_threshold': 0.03, # 3%的横盘幅度
        'lookback_period': "3mo"         # 查看3个月数据
    }
    
    # 扫描并可视化前3个最佳模式
    scan_and_visualize_top_patterns(tickers, top_n=3, **scan_params)