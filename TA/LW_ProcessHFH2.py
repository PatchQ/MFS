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

def find_swing_points(high_series, low_series):
    """
    找出摆动高点和摆动低点
    """
    # 使用滚动窗口找到局部高点和低点
    highs = high_series.rolling(window=WINDOW, center=True).max()
    lows = low_series.rolling(window=WINDOW, center=True).min()
    
    # 找出摆动高点 (当前高点等于滚动窗口内的最大值)
    swing_high_mask = high_series == highs
    swing_highs = high_series[swing_high_mask]
    
    # 找出摆动低点 (当前低点等于滚动窗口内的最小值)
    swing_low_mask = low_series == lows
    swing_lows = low_series[swing_low_mask]
    
    # 过滤掉太接近的摆动点
    swing_highs = filter_close_points(swing_highs, WINDOW)
    swing_lows = filter_close_points(swing_lows, WINDOW)
    
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

def analyze_trend_structure(ticker, period="6mo", tolerance=0.001):
    """
    分析股票的趋势结构
    
    返回:
    trend_analysis: 趋势分析结果
    swing_points: 所有摆动点及其分类
    """
    # 下载数据
    s = yf.Ticker(ticker)
    stock = s.history(period=period,auto_adjust=False)      
    
    if stock.empty:
        print(f"无法获取 {ticker} 的数据")
        return None, None
    
    # 找出摆动点
    swing_highs, swing_lows = find_swing_points(stock['High'], stock['Low'])
    
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
    可視化趨勢分析結果 - 同時顯示陰陽燭和線圖
    """
    # 確保數據格式正確
    if not all(col in stock_data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        print("缺少陰陽燭圖所需的數據欄位")
        return
    
    # 準備陰陽燭數據
    ohlc_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # 準備額外圖層
    add_plots = []
    
    # 1. 添加收盤價線
    close_line = mpf.make_addplot(stock_data['Close'], color='blue', width=1, alpha=0.7, label='收盤價')
    add_plots.append(close_line)
    
    # 2. 準備擺動點標記
    # 將擺動點轉換為與主數據相同索引的序列
    hh_points = swing_analysis[swing_analysis['classification'] == 'HH']
    hl_points = swing_analysis[swing_analysis['classification'] == 'HL']
    lh_points = swing_analysis[swing_analysis['classification'] == 'LH']
    ll_points = swing_analysis[swing_analysis['classification'] == 'LL']
    same_h_points = swing_analysis[swing_analysis['classification'] == '-H']
    same_l_points = swing_analysis[swing_analysis['classification'] == '-L']
    
    # 創建標記序列
    def create_marker_series(points_df, ohlc_index):
        """創建標記點序列，與主數據索引對齊"""
        marker_series = pd.Series(index=ohlc_index, dtype=float)
        for _, point in points_df.iterrows():
            if point['date'] in ohlc_index:
                marker_series[point['date']] = point['price']
        return marker_series
    
    # HH 標記 (綠色三角形)
    if not hh_points.empty:
        hh_series = create_marker_series(hh_points, ohlc_data.index)
        hh_plot = mpf.make_addplot(hh_series, type='scatter', markersize=80, 
                                  marker='^', color='darkgreen', label='HH')
        add_plots.append(hh_plot)
    
    # HL 標記 (藍色三角形)
    if not hl_points.empty:
        hl_series = create_marker_series(hl_points, ohlc_data.index)
        hl_plot = mpf.make_addplot(hl_series, type='scatter', markersize=80, 
                                  marker='^', color='blue', label='HL')
        add_plots.append(hl_plot)
    
    # LH 標記 (橙色倒三角形)
    if not lh_points.empty:
        lh_series = create_marker_series(lh_points, ohlc_data.index)
        lh_plot = mpf.make_addplot(lh_series, type='scatter', markersize=80, 
                                  marker='v', color='orange', label='LH')
        add_plots.append(lh_plot)
    
    # LL 標記 (紅色倒三角形)
    if not ll_points.empty:
        ll_series = create_marker_series(ll_points, ohlc_data.index)
        ll_plot = mpf.make_addplot(ll_series, type='scatter', markersize=80, 
                                  marker='v', color='purple', label='LL')
        add_plots.append(ll_plot)
    
    # -H 標記 (黃色三角形)
    if not same_h_points.empty:
        same_h_series = create_marker_series(same_h_points, ohlc_data.index)
        same_h_plot = mpf.make_addplot(same_h_series, type='scatter', markersize=60, 
                                      marker='s', color='red', label='-H')
        add_plots.append(same_h_plot)
    
    # -L 標記 (青色倒三角形)
    if not same_l_points.empty:
        same_l_series = create_marker_series(same_l_points, ohlc_data.index)
        same_l_plot = mpf.make_addplot(same_l_series, type='scatter', markersize=60, 
                                      marker='s', color='cyan', label='-L')
        add_plots.append(same_l_plot)
    
    # 3. 準備趨勢線
    # 高點趨勢線
    high_points = swing_analysis[swing_analysis['type'] == 'high'].sort_values('date')
    if len(high_points) >= 2:
        # 創建高點趨勢線序列
        high_trend_series = pd.Series(index=ohlc_data.index, dtype=float)
        for i in range(len(high_points)-1):
            start_date = high_points.iloc[i]['date']
            end_date = high_points.iloc[i+1]['date']
            start_price = high_points.iloc[i]['price']
            end_price = high_points.iloc[i+1]['price']
            
            # 計算線性插值
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            for date in date_range:
                if date in ohlc_data.index:
                    # 線性插值計算價格
                    days_total = (end_date - start_date).days
                    if days_total > 0:
                        days_passed = (date - start_date).days
                        ratio = days_passed / days_total
                        price = start_price + (end_price - start_price) * ratio
                        high_trend_series[date] = price
        
        high_trend_plot = mpf.make_addplot(high_trend_series, type='line', 
                                          color='red', linestyle='--', alpha=0.5, label='阻力線')
        add_plots.append(high_trend_plot)
    
    # 低點趨勢線
    low_points = swing_analysis[swing_analysis['type'] == 'low'].sort_values('date')
    if len(low_points) >= 2:
        # 創建低點趨勢線序列
        low_trend_series = pd.Series(index=ohlc_data.index, dtype=float)
        for i in range(len(low_points)-1):
            start_date = low_points.iloc[i]['date']
            end_date = low_points.iloc[i+1]['date']
            start_price = low_points.iloc[i]['price']
            end_price = low_points.iloc[i+1]['price']
            
            # 計算線性插值
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            for date in date_range:
                if date in ohlc_data.index:
                    # 線性插值計算價格
                    days_total = (end_date - start_date).days
                    if days_total > 0:
                        days_passed = (date - start_date).days
                        ratio = days_passed / days_total
                        price = start_price + (end_price - start_price) * ratio
                        low_trend_series[date] = price
        
        low_trend_plot = mpf.make_addplot(low_trend_series, type='line', 
                                         color='green', linestyle='--', alpha=0.5, label='支撐線')
        add_plots.append(low_trend_plot)
    
    # 4. 設置圖表樣式
    mc = mpf.make_marketcolors(
        up='yellow', down='black',
        wick={'up':'green', 'down':'red'},
        volume={'up':'green', 'down':'red'}
    )
    
    custom_style = mpf.make_mpf_style(marketcolors=mc, 
                                    gridstyle='--', 
                                    gridcolor='lightgray',
                                    base_mpf_style='charles', 
                                    rc={
                                        'font.family': 'Microsoft YaHei',
                                        'axes.unicode_minus': False
                                        }
                                    )
    
    
    # 5. 繪製圖表
    title = f'{ticker} 趨勢結構分析 - 主要趨勢: {trend_analysis["trend_direction"]} (強度: {trend_analysis["trend_strength"]})'
    
    # 使用 mplfinance 繪製陰陽燭圖
    fig, axes = mpf.plot(
        ohlc_data,
        type='candle',
        style=custom_style,
        addplot=add_plots,
        title=title,
        ylabel='價格',
        volume=True,
        figsize=(15, 10),
        returnfig=True,
        show_nontrading=False
    )
    
    # 6. 添加圖例和標注
    # 在主圖上添加圖例
    axes[0].legend(loc='upper left')
    
    # 為擺動點添加文字標注
    for _, point in swing_analysis.iterrows():
        if point['date'] in ohlc_data.index:
            # 根據分類設置偏移量
            if point['type'] == 'high':
                y_offset = 10
            else:
                y_offset = -15
            
            # 添加文字標注
            axes[0].annotate(
                point['classification'], 
                (point['date'], point['price']),
                xytext=(5, y_offset),
                textcoords='offset points',
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
            )
    
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

def find_all_patterns(df, pattern_length=3):
    all_patterns = []
    for i in range(len(df) - pattern_length + 1):
        sequence = list(df['pattern'].iloc[i:i+pattern_length])
        all_patterns.append({
            'start_index': i,
            'sequence': sequence,
            'as_string': '->'.join(sequence)
        })
    return all_patterns

# 使用示例
if __name__ == "__main__":

    set_chinese_font()
    # 分析单个股票    
    ticker = "0011.HK"
    period = "6y"
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

        swing_points['PATTERN'] = ""

        for i in range(len(swing_points) - 2):
            templist = list(swing_points['classification'].iloc[i:i+3])            
            swing_points['PATTERN'].iloc[i] = ''.join(templist)

        swing_points["BOSS"] = (swing_points['PATTERN']=="LHLLHH")

        swing_points.to_csv("Data/0011_HHLL.csv", index=False)

        
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
    
    # # 方法2: 使用 ZigZag 指标 (更精确)
    # print(f"\n=== 使用 ZigZag 指标分析 ===")    
    # s = yf.Ticker(ticker)
    # stock_data = s.history(period=period,auto_adjust=True)      


    # zigzag_classified = find_zigzag_swings_with_classification(
    #     stock_data['High'], 
    #     stock_data['Low'], 
    #     min_percent_change=2.0,
    #     tolerance=tolerance
    # )
    
    # # 分析趋势
    # hh_count = len(zigzag_classified[zigzag_classified['classification'] == 'HH'])
    # hl_count = len(zigzag_classified[zigzag_classified['classification'] == 'HL'])
    # lh_count = len(zigzag_classified[zigzag_classified['classification'] == 'LH'])
    # ll_count = len(zigzag_classified[zigzag_classified['classification'] == 'LL'])
    # same_h_count = len(zigzag_classified[zigzag_classified['classification'] == '-H'])
    # same_l_count = len(zigzag_classified[zigzag_classified['classification'] == '-L'])
    
    # bullish_signals = hh_count + hl_count
    # bearish_signals = lh_count + ll_count
    
    # if bullish_signals > bearish_signals:
    #     zigzag_trend = "上升"
    # elif bearish_signals > bullish_signals:
    #     zigzag_trend = "下降"
    # else:
    #     zigzag_trend = "震荡"
    
    # print("ZigZag 摆动点:")
    # print(zigzag_classified)
    # print(f"ZigZag 趋势方向: {zigzag_trend}")
    # print(f"趋势强度: {abs(bullish_signals - bearish_signals)}")
