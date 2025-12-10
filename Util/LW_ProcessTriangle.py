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
PERIOD (list): 多個時間範圍列表
RESISTANCE_RATE (float): 阻力位識別閾值
BREAKOUT_RATE (float): 突破檢查閾值
MIN_PEAKS (int): 峰值之間的最小天數
"""

PERIOD="1y"
RESISTANCE_RATE=0.004
BREAKOUT_RATE=0.004
MIN_PEAKS=10
ADJ=False

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
    
        
def find_resistance_test_pattern(sno, stype):
    """
    找出股票三次測試同一高位的模式，並檢查中間是否有突破
    
    參數:
    symbol (str): 股票代號
    period (str): 資料期間
    resistance_threshold (float): 阻力位識別閾值 (百分比)
    min_days_between_peaks (int): 峰值之間的最小天數
    breakout_threshold (float): 突破閾值 (百分比)
    
    返回:
    dict: 包含模式資訊的字典
    """
    
    # 下載股票資料
    ticker = yf.Ticker(sno)
    stock = ticker.history(period=PERIOD,auto_adjust=ADJ)
    stock = stock[stock['Volume'] > 0]
    stock.index = pd.to_datetime(stock.index,utc=True).tz_convert('Asia/Shanghai')   
    
    if stock.empty:
        print("無法下載資料，請檢查股票代號和期間設定")
        return None
    
    # 確保索引是DatetimeIndex
    stock.index = pd.to_datetime(stock.index,utc=True).tz_convert('Asia/Shanghai')   
    
    # 計算價格變動
    #stock['Price_Change'] = stock['Close'].pct_change()
    stock = stock.assign(Price_Change=stock['Close'].pct_change())
    
    # 找出局部高點 (峰值)
    peaks = find_peaks(stock, window=5)  # 使用5天窗口找峰值
    
    if len(peaks) < 3:
        print("資料中不足3個峰值，無法識別模式")
        return None
    
    # 找出可能的阻力位 (相似的高點群組)
    resistance_levels = group_similar_highs(peaks)
    
    # 尋找符合條件的模式
    patterns = []
    
    for resistance_level, peaks_in_level in resistance_levels.items():
        if len(peaks_in_level) >= 3:  # 至少3次測試同一阻力位
            # 按時間排序峰值
            sorted_peaks = sorted(peaks_in_level, key=lambda x: x['date'])
            
            # 檢查是否符合模式條件，包括突破檢查
            valid_pattern = check_pattern_conditions(
                stock, sorted_peaks, MIN_PEAKS, resistance_level, BREAKOUT_RATE
            )
            
            if valid_pattern:
                patterns.append({
                    'resistance_level': resistance_level,
                    'peaks': sorted_peaks,
                    'pullbacks': valid_pattern['pullbacks'],
                    'pattern_strength': calculate_pattern_strength(valid_pattern),
                    'breakout_checked': True
                })
    
    return {
        'symbol': sno,
        'period': PERIOD,
        'patterns': patterns,
        'all_peaks': peaks,
        'resistance_levels': resistance_levels
    }

def find_peaks(stock_data, window=5):
    """
    找出局部高點 (峰值)
    """
    peaks = []
    
    for i in range(window, len(stock_data) - window):
        current_price = stock_data['Close'].iloc[i]
        # 檢查是否為窗口期內的最高點
        if current_price == stock_data['Close'].iloc[i-window:i+window+1].max():
            peaks.append({
                'date': stock_data.index[i],  # 這裡已經是datetime對象
                'price': current_price,
                'index': i
            })
    
    return peaks

def group_similar_highs(peaks):
    """
    將相似的高點分組，識別可能的阻力位
    """
    if not peaks:
        return {}
    
    # 按價格排序峰值
    sorted_peaks = sorted(peaks, key=lambda x: x['price'])
    
    groups = {}
    current_group = [sorted_peaks[0]]
    current_avg = sorted_peaks[0]['price']
    
    for i in range(1, len(sorted_peaks)):
        price = sorted_peaks[i]['price']
        # 檢查是否屬於當前群組 (價格在閾值範圍內)
        if abs(price - current_avg) / current_avg <= RESISTANCE_RATE:
            current_group.append(sorted_peaks[i])
            current_avg = sum(p['price'] for p in current_group) / len(current_group)
        else:
            # 開始新群組
            if len(current_group) >= 2:  # 只保留有至少2個高點的群組
                groups[current_avg] = current_group
            
            current_group = [sorted_peaks[i]]
            current_avg = price
    
    # 處理最後一個群組
    if len(current_group) >= 2:
        groups[current_avg] = current_group
    
    return groups

def check_pattern_conditions(stock_data, peaks, min_days_between_peaks, resistance_level, breakout_threshold):
    """
    檢查是否符合三次測試同一高位的模式條件，並檢查中間是否有突破
    """
    if len(peaks) < 3:
        return None
    
    # 檢查峰值之間的時間間隔
    for i in range(1, len(peaks)):
        # 確保日期是datetime對象
        if isinstance(peaks[i]['date'], (pd.Timestamp, datetime)):
            days_between = (peaks[i]['date'] - peaks[i-1]['date']).days
        else:
            # 如果日期是索引位置，轉換為實際日期
            date1 = stock_data.index[peaks[i]['date']] if isinstance(peaks[i]['date'], int) else peaks[i]['date']
            date2 = stock_data.index[peaks[i-1]['date']] if isinstance(peaks[i-1]['date'], int) else peaks[i-1]['date']
            days_between = (date1 - date2).days
            
        if days_between < min_days_between_peaks:
            return None
    
    # 檢查在峰值之間是否有突破阻力位的情況
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]['index']
        end_idx = peaks[i+1]['index']
        
        # 檢查兩個峰值之間的價格是否曾突破阻力位
        between_prices = stock_data.iloc[start_idx:end_idx]['Close']
        breakout_level = resistance_level * (1 + breakout_threshold)
        
        if any(price > breakout_level for price in between_prices):
            return None  # 中間有突破，不符合模式
    
    # 找出每次測試後的回落低點
    pullbacks = []
    
    for i in range(len(peaks)):
        # 確保使用正確的日期
        if isinstance(peaks[i]['date'], (pd.Timestamp, datetime)):
            peak_date = peaks[i]['date']
        else:
            peak_date = stock_data.index[peaks[i]['date']] if isinstance(peaks[i]['date'], int) else peaks[i]['date']
            
        peak_idx = stock_data.index.get_loc(peak_date)
        
        # 找出這次峰值之後的低點 (下一次峰值之前)
        if i < len(peaks) - 1:
            if isinstance(peaks[i+1]['date'], (pd.Timestamp, datetime)):
                next_peak_date = peaks[i+1]['date']
            else:
                next_peak_date = stock_data.index[peaks[i+1]['date']] if isinstance(peaks[i+1]['date'], int) else peaks[i+1]['date']
                
            next_peak_idx = stock_data.index.get_loc(next_peak_date)
            # 在兩次峰值之間找最低點
            pullback_data = stock_data.iloc[peak_idx:next_peak_idx]
        else:
            # 最後一次峰值，找之後的低點 (直到資料結束)
            pullback_data = stock_data.iloc[peak_idx:]
        
        if len(pullback_data) > 0:
            min_idx = pullback_data['Close'].idxmin()
            min_price = pullback_data['Close'].min()
            pullbacks.append({
                'date': min_idx,
                'price': min_price,
                'after_peak': peaks[i]['date']
            })
    
    # 檢查回調低點是否依次抬高 (C > B)
    if len(pullbacks) >= 2:
        for i in range(1, len(pullbacks)):
            if pullbacks[i]['price'] <= pullbacks[i-1]['price']:
                return None  # 回調低點沒有依次抬高
    
    return {
        'peaks': peaks,
        'pullbacks': pullbacks,
        'resistance_level': np.mean([p['price'] for p in peaks])
    }

def calculate_pattern_strength(pattern):
    """
    計算模式強度 (基於峰值相似度和回調幅度)
    """
    peaks = pattern['peaks']
    pullbacks = pattern['pullbacks']
    resistance_level = pattern['resistance_level']
    
    # 峰值相似度 (變異係數越小越好)
    peak_prices = [p['price'] for p in peaks]
    peak_cv = np.std(peak_prices) / np.mean(peak_prices)
    
    # 回調深度 (回調幅度越小，模式越強)
    pullback_depths = []
    for i, peak in enumerate(peaks):
        if i < len(pullbacks):
            pullback_depth = (peak['price'] - pullbacks[i]['price']) / peak['price']
            pullback_depths.append(pullback_depth)
    
    avg_pullback_depth = np.mean(pullback_depths) if pullback_depths else 0
    
    # 綜合強度評分 (0-10分，越高越強)
    similarity_score = max(0, 10 - (peak_cv * 1000))  # 峰值相似度貢獻
    pullback_score = max(0, 10 - (avg_pullback_depth * 100))  # 回調深度貢獻
    
    return min(10, (similarity_score + pullback_score) / 2)

def CheckTriangle(sno, stype):
    """
    分析並顯示阻力位測試模式
    """
    result = find_resistance_test_pattern(sno, stype)
    
    if result is None or not result['patterns']:
        print(f"在 {sno} 的 {PERIOD} 資料中未找到符合條件的模式")
        return
    
    print(f"股票 {sno} 的阻力位測試模式分析（期間: {PERIOD}）")
    print(f"突破檢查閾值: {BREAKOUT_RATE*100}%")
    print("=" * 80)
    
    for i, pattern in enumerate(result['patterns']):
        print(f"\n模式 #{i+1} (強度: {pattern['pattern_strength']:.1f}/10)")
        print(f"阻力位: ${pattern['resistance_level']:.2f}")
        print(f"突破檢查: {'已通過' if pattern.get('breakout_checked', False) else '未檢查'}")
        print("-" * 50)
        
        peaks = pattern['peaks']
        pullbacks = pattern['pullbacks']
        
        for j, peak in enumerate(peaks):
            # 確保日期格式正確
            if isinstance(peak['date'], (pd.Timestamp, datetime)):
                peak_date_str = peak['date'].strftime('%Y-%m-%d')
            else:
                peak_date_str = str(peak['date'])
                
            print(f"第{j+1}次測試高位A:")
            print(f"  日期: {peak_date_str}")
            print(f"  價格: ${peak['price']:.2f}")
            
            if j < len(pullbacks):
                pullback = pullbacks[j]
                
                if isinstance(pullback['date'], (pd.Timestamp, datetime)):
                    pullback_date_str = pullback['date'].strftime('%Y-%m-%d')
                else:
                    pullback_date_str = str(pullback['date'])
                    
                print(f"  回調低點: ${pullback['price']:.2f} (日期: {pullback_date_str})")
                
                if j > 0:
                    prev_pullback = pullbacks[j-1]
                    improvement = ((pullback['price'] - prev_pullback['price']) / prev_pullback['price']) * 100
                    print(f"  較前一次回調低點上漲: {improvement:.2f}%")
            
            # 檢查下一個峰值之前是否有突破
            if j < len(peaks) - 1:
                next_peak = peaks[j+1]
                if isinstance(next_peak['date'], (pd.Timestamp, datetime)):
                    next_peak_date_str = next_peak['date'].strftime('%Y-%m-%d')
                else:
                    next_peak_date_str = str(next_peak['date'])
                    
                print(f"  下一次測試日期: {next_peak_date_str}")
            
            print()
    
    if result and result['patterns']:
        set_chinese_font()
        visualize_resistance_pattern(sno, stype, pattern_index=0)    
    
    return result

def visualize_resistance_pattern(sno, stype, pattern_index=0):
    """
    可視化阻力位測試模式 - 同時顯示陰陽燭和線圖
    """
    result = find_resistance_test_pattern(sno, stype)
    
    if result is None or not result['patterns']:
        print("沒有可視化的模式")
        return
    
    if pattern_index >= len(result['patterns']):
        print(f"模式索引 {pattern_index} 超出範圍")
        return
    
    pattern = result['patterns'][pattern_index]

    ticker = yf.Ticker(sno)
    stock = ticker.history(period=PERIOD,auto_adjust=ADJ)
    stock = stock[stock['Volume'] > 0]
    stock.index = pd.to_datetime(stock.index,utc=True).tz_convert('Asia/Shanghai')   

    #stock = df.tail(DAYS).copy()  # 使用 copy() 避免 SettingWithCopyWarning
        
    # 準備陰陽燭圖所需的數據
    if not all(col in stock.columns for col in ['Open', 'High', 'Low', 'Close']):
        print("缺少陰陽燭圖所需的數據欄位 (Open, High, Low, Close)")
        return
    
    # 準備 mplfinance 所需的數據格式
    ohlc_data = stock[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # 準備額外圖形元素
    add_plots = []
    
    # 1. 添加收盤價線 - 確保與主數據相同的索引
    close_line = mpf.make_addplot(stock['Close'], color='blue', width=1, alpha=0.7, label='收盤價')
    add_plots.append(close_line)
    
    # 2. 準備峰值標記 - 確保日期在股票數據索引中
    peak_prices = []
    peak_dates = []
    for i, peak in enumerate(pattern['peaks']):
        # 處理日期格式
        if not isinstance(peak['date'], (pd.Timestamp, datetime)):
            if isinstance(peak['date'], int):
                # 如果是索引數字，確保在範圍內
                if peak['date'] < len(stock.index):
                    peak_date = stock.index[peak['date']]
                else:
                    continue
            else:
                peak_date = pd.to_datetime(peak['date'])
        else:
            peak_date = peak['date']
        
        # 確保日期在股票數據的索引中
        if peak_date in stock.index:
            peak_prices.append(peak['price'])
            peak_dates.append(peak_date)
        else:
            # 如果日期不在索引中，找到最接近的日期
            try:
                closest_date = min(stock.index, key=lambda x: abs(x - peak_date))
                if abs((closest_date - peak_date).days) <= 5:  # 允許5天內的誤差
                    peak_prices.append(peak['price'])
                    peak_dates.append(closest_date)
                    print(f"調整峰值日期: {peak_date} -> {closest_date}")
            except:
                continue
    
    # 創建峰值標記序列
    if peak_prices and peak_dates:
        # 確保日期和價格數量一致
        if len(peak_dates) == len(peak_prices):
            peaks_series = pd.Series(peak_prices, index=peak_dates)
            # 重新索引以確保與主數據對齊
            peaks_series = peaks_series.reindex(stock.index, method=None)
            peak_markers = mpf.make_addplot(peaks_series, type='scatter', markersize=80, 
                                           marker='v', color='red', label='峰值')
            add_plots.append(peak_markers)
    
    # 3. 準備回調低點標記
    pullback_prices = []
    pullback_dates = []
    for i, pullback in enumerate(pattern['pullbacks']):
        # 處理日期格式
        if not isinstance(pullback['date'], (pd.Timestamp, datetime)):
            if isinstance(pullback['date'], int):
                if pullback['date'] < len(stock.index):
                    pullback_date = stock.index[pullback['date']]
                else:
                    continue
            else:
                pullback_date = pd.to_datetime(pullback['date'])
        else:
            pullback_date = pullback['date']
        
        # 確保日期在股票數據的索引中
        if pullback_date in stock.index:
            pullback_prices.append(pullback['price'])
            pullback_dates.append(pullback_date)
        else:
            # 如果日期不在索引中，找到最接近的日期
            try:
                closest_date = min(stock.index, key=lambda x: abs(x - pullback_date))
                if abs((closest_date - pullback_date).days) <= 5:  # 允許5天內的誤差
                    pullback_prices.append(pullback['price'])
                    pullback_dates.append(closest_date)
                    print(f"調整回調日期: {pullback_date} -> {closest_date}")
            except:
                continue
    
    # 創建回調低點標記序列
    if pullback_prices and pullback_dates:
        if len(pullback_dates) == len(pullback_prices):
            pullbacks_series = pd.Series(pullback_prices, index=pullback_dates)
            # 重新索引以確保與主數據對齊
            pullbacks_series = pullbacks_series.reindex(stock.index, method=None)
            pullback_markers = mpf.make_addplot(pullbacks_series, type='scatter', markersize=80, 
                                               marker='^', color='green', label='回調低點')
            add_plots.append(pullback_markers)
    
    # 4. 準備趨勢線（連接回調低點）
    if len(pattern['pullbacks']) >= 2:
        trend_dates = []
        trend_prices = []
        
        for pullback in pattern['pullbacks']:
            # 處理日期格式
            if not isinstance(pullback['date'], (pd.Timestamp, datetime)):
                if isinstance(pullback['date'], int):
                    if pullback['date'] < len(stock.index):
                        pullback_date = stock.index[pullback['date']]
                    else:
                        continue
                else:
                    pullback_date = pd.to_datetime(pullback['date'])
            else:
                pullback_date = pullback['date']
            
            # 確保日期在股票數據的索引中
            if pullback_date in stock.index:
                trend_dates.append(pullback_date)
                trend_prices.append(pullback['price'])
            else:
                # 如果日期不在索引中，找到最接近的日期
                try:
                    closest_date = min(stock.index, key=lambda x: abs(x - pullback_date))
                    if abs((closest_date - pullback_date).days) <= 5:
                        trend_dates.append(closest_date)
                        trend_prices.append(pullback['price'])
                        print(f"調整趨勢線日期: {pullback_date} -> {closest_date}")
                except:
                    continue
        
        # 創建趨勢線數據
        if len(trend_dates) >= 2 and len(trend_dates) == len(trend_prices):
            # 按日期排序
            sorted_indices = np.argsort(trend_dates)
            trend_dates_sorted = [trend_dates[i] for i in sorted_indices]
            trend_prices_sorted = [trend_prices[i] for i in sorted_indices]
            
            # 創建趨勢線序列
            trend_series = pd.Series(trend_prices_sorted, index=trend_dates_sorted)
            
            # 創建連續的趨勢線數據
            # 找到趨勢線的開始和結束日期
            start_date = min(trend_dates_sorted)
            end_date = max(trend_dates_sorted)
            
            # 計算趨勢線的斜率和截距
            x_values = [(date - start_date).days for date in trend_dates_sorted]
            if len(x_values) > 1 and x_values[-1] > x_values[0]:
                slope, intercept = np.polyfit(x_values, trend_prices_sorted, 1)
                
                # 創建連續的日期範圍
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                x_range = [(date - start_date).days for date in date_range]
                y_range = [slope * x + intercept for x in x_range]
                
                # 創建趨勢線序列
                trend_line_series = pd.Series(y_range, index=date_range)
                
                # 重新索引以確保與主數據對齊
                trend_line_series = trend_line_series.reindex(stock.index, method=None)
                
                trend_line = mpf.make_addplot(trend_line_series, type='line', color='green', 
                                             linestyle='--', alpha=0.7, label='趨勢線')
                add_plots.append(trend_line)
    
    # 5. 準備水平線
    resistance_level = pattern['resistance_level']
    breakout_level = resistance_level * (1 + BREAKOUT_RATE)
    
    # 使用 mplfinance 繪製陰陽燭圖
    title = f'{sno} 三次測試高位模式 (強度: {pattern["pattern_strength"]:.1f}/10)'
    
    # 設置圖形樣式
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

    
    try:
        # 繪製圖形
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
            hlines=dict(
                hlines=[resistance_level, breakout_level],
                colors=['red', 'orange'],
                linestyle=['--', ':'],
                alpha=[0.7, 0.5]
            )
        )
        
        # 添加圖例
        axes[0].legend(loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
    except ValueError as e:
        print(f"繪圖錯誤: {e}")
        print("嘗試使用簡化版本...")
        # 如果出現錯誤，使用簡化版本
        visualize_resistance_pattern_simple(sno, stype, pattern_index)

def visualize_resistance_pattern_simple(sno, stype, pattern_index=0):
    """
    簡化版本 - 只繪製陰陽燭和基本標記
    """
    result = find_resistance_test_pattern(sno, stype)
    
    if result is None or not result['patterns']:
        print("沒有可視化的模式")
        return
    
    if pattern_index >= len(result['patterns']):
        print(f"模式索引 {pattern_index} 超出範圍")
        return
    
    pattern = result['patterns'][pattern_index]

    ticker = yf.Ticker(sno)
    stock = ticker.history(period=PERIOD,auto_adjust=ADJ)
    stock = stock[stock['Volume'] > 0]
    stock.index = pd.to_datetime(stock.index,utc=True).tz_convert('Asia/Shanghai')   

    # 準備陰陽燭圖數據
    ohlc_data = stock[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # 準備水平線
    resistance_level = pattern['resistance_level']
    breakout_level = resistance_level * (1 + BREAKOUT_RATE)
    
    # 設置圖形樣式
     # 創建自定義樣式
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

    
    # 只繪製陰陽燭圖和水平線
    mpf.plot(
        ohlc_data,
        type='candle',
        style=custom_style,
        title=f'{sno} 三次測試高位模式 (強度: {pattern["pattern_strength"]:.1f}/10)',
        ylabel='價格',
        volume=True,
        figsize=(15, 10),
        hlines=dict(
            hlines=[resistance_level, breakout_level],
            colors=['red', 'orange'],
            linestyle=['--', ':'],
            alpha=[0.7, 0.5],
            label=['阻力位', '突破檢查線']
        )
    )


def main(stype):

    snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(PATH+"/"+stype)))
    SLIST = pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST[:]

    with cf.ProcessPoolExecutor(max_workers=12) as executor:
        list(tqdm(executor.map(CheckTriangle,SLIST["sno"],SLIST["stype"],chunksize=1),total=len(SLIST)))


if __name__ == '__main__':
    start = t.perf_counter()

    main("L")
    main("M")
    main("S")

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')