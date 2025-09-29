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
DAYS=150
RESISTANCE_RATE=0.004
BREAKOUT_RATE=0.004
MIN_PEAKS=10

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

def find_swing_points(high_series, low_series, window=3):
    """
    找出摆动高点和摆动低点
    """
    # 使用滚动窗口找到局部高点和低点
    highs = high_series.rolling(window=window, center=True).max()
    lows = low_series.rolling(window=window, center=True).min()
    
    # 找出摆动高点 (当前高点等于滚动窗口内的最大值)
    swing_high_mask = high_series == highs
    swing_highs = high_series[swing_high_mask]
    
    # 找出摆动低点 (当前低点等于滚动窗口内的最小值)
    swing_low_mask = low_series == lows
    swing_lows = low_series[swing_low_mask]
    
    # 过滤掉太接近的摆动点
    swing_highs = filter_close_points(swing_highs, window)
    swing_lows = filter_close_points(swing_lows, window)
    
    return swing_highs, swing_lows

def filter_close_points(series, min_distance):
    """
    过滤掉距离太近的摆动点
    """
    filtered = pd.Series(index=series.index, dtype=float)
    last_index = None
    
    for date, value in series.items():
        if last_index is None:
            filtered[date] = value
            last_index = date
        else:
            # 计算与前一个点的距离 (天数)
            days_diff = (date - last_index).days
            if days_diff >= min_distance:
                filtered[date] = value
                last_index = date
    
    return filtered.dropna()

def classify_all_swing_points(swing_highs, swing_lows, tolerance=0.001):
    """
    分类所有摆动点为 HH, HL, LH, LL, -H, -L
    
    参数:
    tolerance: 价格相等的容忍度 (0.1%)
    
    返回:
    swing_analysis: 包含所有摆动点及其分类的DataFrame
    """
    # 合并所有摆动点并标记类型
    highs_df = pd.DataFrame({
        'date': swing_highs.index,
        'price': swing_highs.values,
        'type': 'high'
    })
    
    lows_df = pd.DataFrame({
        'date': swing_lows.index,
        'price': swing_lows.values,
        'type': 'low'
    })
    
    # 合并并排序
    all_swings = pd.concat([highs_df, lows_df]).sort_values('date')
    all_swings = all_swings.reset_index(drop=True)
    
    # 初始化分类列
    all_swings['classification'] = None
    
    # 分离高点和低点序列
    high_points = all_swings[all_swings['type'] == 'high'].copy().reset_index(drop=True)
    low_points = all_swings[all_swings['type'] == 'low'].copy().reset_index(drop=True)
    
    # 分类高点序列
    for i in range(len(high_points)):
        if i == 0:
            # 第一个高点标记为起始点
            high_points.loc[i, 'classification'] = 'Start_H'
            continue
            
        current_price = high_points.loc[i, 'price']
        prev_price = high_points.loc[i-1, 'price']
        
        # 计算价格变化百分比
        price_diff_pct = abs((current_price - prev_price) / prev_price)
        
        if price_diff_pct <= tolerance:
            high_points.loc[i, 'classification'] = '-H'  # 相同高位
        elif current_price > prev_price:
            high_points.loc[i, 'classification'] = 'HH'  # 更高高点
        else:
            high_points.loc[i, 'classification'] = 'LH'  # 更低高点
    
    # 分类低点序列
    for i in range(len(low_points)):
        if i == 0:
            # 第一个低点标记为起始点
            low_points.loc[i, 'classification'] = 'Start_L'
            continue
            
        current_price = low_points.loc[i, 'price']
        prev_price = low_points.loc[i-1, 'price']
        
        # 计算价格变化百分比
        price_diff_pct = abs((current_price - prev_price) / prev_price)
        
        if price_diff_pct <= tolerance:
            low_points.loc[i, 'classification'] = '-L'  # 相同低位
        elif current_price > prev_price:
            low_points.loc[i, 'classification'] = 'HL'  # 更高低点
        else:
            low_points.loc[i, 'classification'] = 'LL'  # 更低低点
    
    # 合并分类结果
    classified_swings = pd.concat([high_points, low_points]).sort_values('date')
    
    return classified_swings

def analyze_trend_structure(ticker, period="6mo", window=3, tolerance=0.001):
    """
    分析股票的趋势结构
    
    返回:
    trend_analysis: 趋势分析结果
    swing_points: 所有摆动点及其分类
    """
    # 下载数据
    s = yf.Ticker(ticker)
    stock = s.history(period=period,auto_adjust=True)      
    
    if stock.empty:
        print(f"无法获取 {ticker} 的数据")
        return None, None
    
    # 找出摆动点
    swing_highs, swing_lows = find_swing_points(stock['High'], stock['Low'], window)
    
    # 分类所有摆动点
    swing_analysis = classify_all_swing_points(swing_highs, swing_lows, tolerance)
    
    # 分析整体趋势
    trend_direction = "震荡"
    hh_count = len(swing_analysis[swing_analysis['classification'] == 'HH'])
    hl_count = len(swing_analysis[swing_analysis['classification'] == 'HL'])
    lh_count = len(swing_analysis[swing_analysis['classification'] == 'LH'])
    ll_count = len(swing_analysis[swing_analysis['classification'] == 'LL'])
    same_h_count = len(swing_analysis[swing_analysis['classification'] == '-H'])
    same_l_count = len(swing_analysis[swing_analysis['classification'] == '-L'])
    
    # 计算趋势强度
    bullish_signals = hh_count + hl_count
    bearish_signals = lh_count + ll_count
    
    if bullish_signals > bearish_signals:
        trend_direction = "上升"
    elif bearish_signals > bullish_signals:
        trend_direction = "下降"
    
    trend_analysis = {
        'ticker': ticker,
        'period': period,
        'trend_direction': trend_direction,
        'hh_count': hh_count,
        'hl_count': hl_count,
        'lh_count': lh_count,
        'll_count': ll_count,
        'same_h_count': same_h_count,
        'same_l_count': same_l_count,
        'total_swings': len(swing_analysis),
        'bullish_signals': bullish_signals,
        'bearish_signals': bearish_signals,
        'trend_strength': abs(bullish_signals - bearish_signals)
    }
    
    return trend_analysis, swing_analysis

def visualize_trend_analysis(ticker, stock_data, swing_analysis, trend_analysis):
    """
    可视化趋势分析结果
    """
    plt.figure(figsize=(15, 10))
    
    # 绘制股价
    plt.plot(stock_data.index, stock_data['Close'], label='收盘价', color='black', linewidth=1, alpha=0.7)
    
    # 提取分类后的摆动点
    hh_points = swing_analysis[swing_analysis['classification'] == 'HH']
    hl_points = swing_analysis[swing_analysis['classification'] == 'HL']
    lh_points = swing_analysis[swing_analysis['classification'] == 'LH']
    ll_points = swing_analysis[swing_analysis['classification'] == 'LL']
    same_h_points = swing_analysis[swing_analysis['classification'] == '-H']
    same_l_points = swing_analysis[swing_analysis['classification'] == '-L']
    start_h_points = swing_analysis[swing_analysis['classification'] == 'Start_H']
    start_l_points = swing_analysis[swing_analysis['classification'] == 'Start_L']
    
    # 绘制摆动点
    # plt.scatter(hh_points['date'], hh_points['price'], color='darkgreen', marker='^', s=120, label='HH', zorder=5)
    # plt.scatter(hl_points['date'], hl_points['price'], color='blue', marker='^', s=120, label='HL', zorder=5)
    # plt.scatter(lh_points['date'], lh_points['price'], color='orange', marker='v', s=120, label='LH', zorder=5)
    # plt.scatter(ll_points['date'], ll_points['price'], color='red', marker='v', s=120, label='LL', zorder=5)
    # plt.scatter(same_h_points['date'], same_h_points['price'], color='yellow', marker='s', s=100, label='-H', zorder=5)
    # plt.scatter(same_l_points['date'], same_l_points['price'], color='cyan', marker='s', s=100, label='-L', zorder=5)
    # plt.scatter(start_h_points['date'], start_h_points['price'], color='gray', marker='*', s=80, label='起始高点', zorder=5)
    # plt.scatter(start_l_points['date'], start_l_points['price'], color='gray', marker='*', s=80, label='起始低点', zorder=5)
    
    # 添加标注
    for _, point in hh_points.iterrows():
        plt.annotate('HH', (point['date'], point['price']), 
                    xytext=(5, 10), textcoords='offset points', fontweight='bold', color='darkgreen')
    
    for _, point in hl_points.iterrows():
        plt.annotate('HL', (point['date'], point['price']), 
                    xytext=(5, 10), textcoords='offset points', fontweight='bold', color='blue')
    
    for _, point in lh_points.iterrows():
        plt.annotate('LH', (point['date'], point['price']), 
                    xytext=(5, -15), textcoords='offset points', fontweight='bold', color='orange')
    
    for _, point in ll_points.iterrows():
        plt.annotate('LL', (point['date'], point['price']), 
                    xytext=(5, -15), textcoords='offset points', fontweight='bold', color='red')
    
    for _, point in same_h_points.iterrows():
        plt.annotate('-H', (point['date'], point['price']), 
                    xytext=(5, 10), textcoords='offset points', fontweight='bold', color='yellow')
    
    for _, point in same_l_points.iterrows():
        plt.annotate('-L', (point['date'], point['price']), 
                    xytext=(5, -15), textcoords='offset points', fontweight='bold', color='cyan')
    
    # 连接摆动点以显示趋势
    highs = swing_analysis[swing_analysis['type'] == 'high'].sort_values('date')
    lows = swing_analysis[swing_analysis['type'] == 'low'].sort_values('date')
    
    plt.plot(highs['date'], highs['price'], 'r--', alpha=0.5, label='阻力线')
    plt.plot(lows['date'], lows['price'], 'g--', alpha=0.5, label='支撑线')
    
    plt.title(f'{ticker} 趋势结构分析 - 主要趋势: {trend_analysis["trend_direction"]} '
              f'(强度: {trend_analysis["trend_strength"]})')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def find_zigzag_swings_with_classification(high_series, low_series, min_percent_change=2.0, tolerance=0.001):
    """
    使用 ZigZag 指标找出更精确的摆动点，并进行分类
    
    返回:
    classified_swings: 包含所有摆动点及其分类的DataFrame
    """
    # 合并高低点
    prices = pd.concat([high_series, low_series]).sort_index()
    
    # ZigZag 算法实现
    swing_points = []
    last_swing_index = prices.index[0]
    last_swing_price = prices.iloc[0]
    last_swing_type = 'high' if prices.index[0] in high_series.index else 'low'
    
    for i in range(1, len(prices)):
        current_index = prices.index[i]
        current_price = prices.iloc[i]
        
        # 计算价格变化百分比
        percent_change = abs((current_price - last_swing_price) / last_swing_price * 100)
        
        # 检查是否达到最小变化阈值
        if percent_change >= min_percent_change:
            swing_points.append({
                'date': last_swing_index,
                'price': last_swing_price,
                'type': last_swing_type
            })
            
            last_swing_index = current_index
            last_swing_price = current_price
            last_swing_type = 'high' if current_index in high_series.index else 'low'
    
    # 添加最后一个摆动点
    swing_points.append({
        'date': last_swing_index,
        'price': last_swing_price,
        'type': last_swing_type
    })
    
    # 转换为DataFrame
    swings_df = pd.DataFrame(swing_points)
    
    # 分类摆动点
    return classify_all_swing_points(
        swings_df[swings_df['type'] == 'high']['price'],
        swings_df[swings_df['type'] == 'low']['price'],
        tolerance
    )

# 使用示例
if __name__ == "__main__":

    set_chinese_font()
    # 分析单个股票    
    ticker = "TSLA"
    period = "1y"
    tolerance = 0.001  # 0.1% 的容忍度
    
    print(f"分析 {ticker} 的趋势结构...")
    
    # 方法1: 使用滚动窗口方法
    trend_analysis, swing_points = analyze_trend_structure(ticker, period, tolerance=tolerance)
    
    if trend_analysis is not None:
        print(f"\n=== {ticker} 趋势分析结果 ===")
        print(f"趋势方向: {trend_analysis['trend_direction']}")
        print(f"HH数量: {trend_analysis['hh_count']}")
        print(f"HL数量: {trend_analysis['hl_count']}")
        print(f"LH数量: {trend_analysis['lh_count']}")
        print(f"LL数量: {trend_analysis['ll_count']}")
        print(f"-H数量: {trend_analysis['same_h_count']}")
        print(f"-L数量: {trend_analysis['same_l_count']}")
        print(f"总摆动点: {trend_analysis['total_swings']}")
        print(f"趋势强度: {trend_analysis['trend_strength']}")
        
        print(f"\n=== 详细摆动点分类 ===")
        print(swing_points)
        
        # 检查是否有未分类的点
        unclassified = swing_points[swing_points['classification'].isna()]
        if len(unclassified) == 0:
            print("✓ 所有摆动点都已正确分类")
        else:
            print(f"⚠ 仍有 {len(unclassified)} 个未分类的摆动点")
        
        # 可视化结果        
        s = yf.Ticker(ticker)
        stock_data = s.history(period=period,auto_adjust=True)      

        
        visualize_trend_analysis(ticker, stock_data, swing_points, trend_analysis)
    
    # 方法2: 使用 ZigZag 指标 (更精确)
    print(f"\n=== 使用 ZigZag 指标分析 ===")    
    s = yf.Ticker(ticker)
    stock_data = s.history(period=period,auto_adjust=True)      


    zigzag_classified = find_zigzag_swings_with_classification(
        stock_data['High'], 
        stock_data['Low'], 
        min_percent_change=2.0,
        tolerance=tolerance
    )
    
    # 分析趋势
    hh_count = len(zigzag_classified[zigzag_classified['classification'] == 'HH'])
    hl_count = len(zigzag_classified[zigzag_classified['classification'] == 'HL'])
    lh_count = len(zigzag_classified[zigzag_classified['classification'] == 'LH'])
    ll_count = len(zigzag_classified[zigzag_classified['classification'] == 'LL'])
    same_h_count = len(zigzag_classified[zigzag_classified['classification'] == '-H'])
    same_l_count = len(zigzag_classified[zigzag_classified['classification'] == '-L'])
    
    bullish_signals = hh_count + hl_count
    bearish_signals = lh_count + ll_count
    
    if bullish_signals > bearish_signals:
        zigzag_trend = "上升"
    elif bearish_signals > bullish_signals:
        zigzag_trend = "下降"
    else:
        zigzag_trend = "震荡"
    
    print("ZigZag 摆动点:")
    print(zigzag_classified)
    print(f"ZigZag 趋势方向: {zigzag_trend}")
    print(f"趋势强度: {abs(bullish_signals - bearish_signals)}")
