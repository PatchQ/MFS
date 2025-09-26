import pandas as pd
import numpy as np
import yfinance as yf
import concurrent.futures as cf
import os
from tqdm import tqdm
import time as t
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/" 

# 設定中文字體
def set_chinese_font():
    """
    設定 matplotlib 使用支援中文的字體
    """
    try:
        # 嘗試使用系統中的中文字體
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
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
        df['EMA250'] = df['Close'].ewm(span=250, min_periods=125, adjust=False).mean()

        df['EMA'] = ((df["EMA10"] > df["EMA22"]) & (df["EMA22"] > df["EMA50"]) & (df["EMA50"] > df["EMA100"]) & (df["EMA100"] > df["EMA250"]))        
        
        return df        

    except Exception as e:
        print(f"Indicator error: {str(e)}")
        df['EMA'] = False
        return df
    

def adaptive_resistance_pattern(symbol, target_period="6mo", base_period="4mo", 
                               resistance_threshold=0.03, breakout_threshold=0.01):
    """
    自適應阻力模式識別：先使用基礎期間識別模式，然後在目標期間驗證
    
    參數:
    symbol (str): 股票代號
    target_period (str): 目標時間範圍 (如 "6mo")
    base_period (str): 基礎時間範圍 (如 "4mo")
    resistance_threshold (float): 阻力位識別閾值
    breakout_threshold (float): 突破檢查閾值
    """
    
    # 下載目標期間的完整數據      
    ticker = yf.Ticker(symbol)
    target_data = ticker.history(period=target_period,auto_adjust=True)    
    #stock = convertData(stock)    
    target_data.index = pd.to_datetime(target_data.index)
    
    # 下載基礎期間的數據
    ticker = yf.Ticker(symbol)
    base_data = ticker.history(period=base_period,auto_adjust=True)    
    #stock = convertData(stock)    
    base_data.index = pd.to_datetime(base_data.index)
    
    # 在基礎期間內識別模式
    base_patterns = find_patterns_in_data(base_data, resistance_threshold, breakout_threshold)
    
    if not base_patterns:
        print(f"在基礎期間 {base_period} 中未找到模式")
        return None
    
    # 在目標期間內驗證這些模式
    validated_patterns = []
    for pattern in base_patterns:
        # 檢查模式中的關鍵點是否也在目標期間內
        if validate_pattern_in_target(pattern, target_data, base_data.index[0]):
            validated_patterns.append(pattern)
    
    if validated_patterns:
        print(f"在目標期間 {target_period} 中驗證了 {len(validated_patterns)} 個模式")
        return validated_patterns
    else:
        print(f"在目標期間 {target_period} 中未驗證任何模式")
        return None

def find_patterns_in_data(stock_data, resistance_threshold=0.03, breakout_threshold=0.01):
    """
    在給定的數據中識別三次測試高位模式
    """
    # 找出所有高點
    highs = find_peaks_adaptive(stock_data)
    
    if len(highs) < 3:
        return []
    
    # 按價格分組高點
    groups = group_similar_highs_adaptive(highs, threshold=resistance_threshold)
    
    # 找出至少有3個高點的群組
    valid_groups = {level: sorted(peaks, key=lambda x: x['date']) 
                   for level, peaks in groups.items() if len(peaks) >= 3}
    
    # 檢查每個群組是否符合模式條件
    patterns = []
    for level, peaks in valid_groups.items():
        # 只考慮最近的三個峰值
        recent_peaks = peaks[-3:] if len(peaks) > 3 else peaks
        
        # 檢查峰值之間是否有突破
        valid_pattern = True
        for i in range(len(recent_peaks)-1):
            start_idx = recent_peaks[i]['index']
            end_idx = recent_peaks[i+1]['index']
            between_prices = stock_data.iloc[start_idx:end_idx]['Close']
            
            if any(price > level * (1 + breakout_threshold) for price in between_prices):
                valid_pattern = False
                break
        
        if valid_pattern:
            # 找出回調低點
            pullbacks = []
            for i in range(len(recent_peaks)-1):
                start_idx = recent_peaks[i]['index']
                end_idx = recent_peaks[i+1]['index']
                pullback_data = stock_data.iloc[start_idx:end_idx]
                
                if len(pullback_data) > 0:
                    min_idx = pullback_data['Close'].idxmin()
                    min_price = pullback_data['Close'].min()
                    pullbacks.append({
                        'date': min_idx,
                        'price': min_price
                    })
            
            # 檢查回調低點是否依次抬高
            if len(pullbacks) >= 2:
                ascending = True
                for i in range(1, len(pullbacks)):
                    if pullbacks[i]['price'] <= pullbacks[i-1]['price']:
                        ascending = False
                        break
                
                if ascending:
                    patterns.append({
                        'resistance_level': level,
                        'peaks': recent_peaks,
                        'pullbacks': pullbacks,
                        'data_points': len(stock_data)
                    })
    
    return patterns

def find_peaks_adaptive(stock_data, window_ratio=0.05):
    """
    自適應峰值檢測：根據數據點數量調整窗口大小
    """
    # 根據數據點數量計算窗口大小
    min_window = 3  # 最小窗口
    max_window = 10  # 最大窗口
    window_size = max(min_window, min(max_window, int(len(stock_data) * window_ratio)))
    
    highs = []
    for i in range(window_size, len(stock_data) - window_size):
        if stock_data['Close'].iloc[i] == stock_data['Close'].iloc[i-window_size:i+window_size+1].max():
            highs.append({
                'date': stock_data.index[i],
                'price': stock_data['Close'].iloc[i],
                'index': i
            })
    
    return highs

def group_similar_highs_adaptive(highs, threshold=0.03, min_group_size=2):
    """
    自適應高點分組：根據數據特徵調整分組策略
    """
    if not highs:
        return {}
    
    # 按價格排序高點
    sorted_highs = sorted(highs, key=lambda x: x['price'])
    
    groups = {}
    
    # 如果高點數量較少，使用較寬的閾值
    if len(sorted_highs) < 10:
        threshold = max(threshold, 0.05)  # 至少5%的閾值
    
    current_group = [sorted_highs[0]]
    current_avg = sorted_highs[0]['price']
    
    for i in range(1, len(sorted_highs)):
        price = sorted_highs[i]['price']
        
        # 檢查是否屬於當前群組
        if abs(price - current_avg) / current_avg <= threshold:
            current_group.append(sorted_highs[i])
            current_avg = sum(p['price'] for p in current_group) / len(current_group)
        else:
            # 開始新群組
            if len(current_group) >= min_group_size:
                groups[current_avg] = current_group
            
            current_group = [sorted_highs[i]]
            current_avg = price
    
    # 處理最後一個群組
    if len(current_group) >= min_group_size:
        groups[current_avg] = current_group
    
    return groups

def validate_pattern_in_target(pattern, target_data, base_start_date):
    """
    在目標數據中驗證模式
    """
    # 檢查模式中的關鍵點是否在目標數據中
    resistance_level = pattern['resistance_level']
    
    # 在目標數據中找出接近阻力位的點
    target_highs = []
    for i in range(len(target_data)):
        if abs(target_data['Close'].iloc[i] - resistance_level) / resistance_level <= 0.05:  # 5%閾值
            target_highs.append({
                'date': target_data.index[i],
                'price': target_data['Close'].iloc[i],
                'index': i
            })
    
    # 如果目標數據中有足夠的高點，則認為模式有效
    return len(target_highs) >= 2  # 至少兩個高點

def multi_timeframe_analysis(symbol, periods=["4mo", "6mo", "1y"]):
    """
    多時間範圍分析：在多個時間範圍內尋找一致的模式
    """
    all_patterns = {}
    
    for period in periods:
        print(f"\n分析時間範圍: {period}")
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period,auto_adjust=True)                    
        data.index = pd.to_datetime(data.index)
        
        patterns = find_patterns_in_data(data)
        all_patterns[period] = patterns
        
        if patterns:
            print(f"找到 {len(patterns)} 個模式")
            for i, pattern in enumerate(patterns):
                print(f"  模式 {i+1}: 阻力位 ${pattern['resistance_level']:.2f}")
        else:
            print("未找到模式")
    
    # 找出在多個時間範圍內都出現的阻力位
    common_resistance_levels = find_common_resistance_levels(all_patterns)
    
    if common_resistance_levels:
        print(f"\n共同阻力位: {common_resistance_levels}")
        
        # 使用最長的時間範圍可視化這些阻力位
        longest_period = periods[-1]
        visualize_common_resistance_levels(symbol, longest_period, common_resistance_levels)
        
        return common_resistance_levels
    else:
        print("\n未找到共同的阻力位")
        return None

def find_common_resistance_levels(all_patterns, threshold=0.03):
    """
    找出在多個時間範圍內都出現的阻力位
    """
    # 收集所有阻力位
    all_levels = []
    for period, patterns in all_patterns.items():
        for pattern in patterns:
            all_levels.append({
                'level': pattern['resistance_level'],
                'period': period
            })
    
    if not all_levels:
        return []
    
    # 按阻力位分組
    level_groups = {}
    for item in all_levels:
        level = item['level']
        period = item['period']
        
        found_group = False
        for group_level in level_groups:
            if abs(level - group_level) / group_level <= threshold:
                level_groups[group_level].add(period)
                found_group = True
                break
        
        if not found_group:
            level_groups[level] = {period}
    
    # 找出在多個時間範圍內出現的阻力位
    common_levels = []
    for level, periods in level_groups.items():
        if len(periods) > 1:  # 在至少兩個時間範圍內出現
            common_levels.append(level)
    
    return common_levels

def visualize_common_resistance_levels(symbol, period, resistance_levels, patterns=None):
    """
    可視化共同阻力位，並標記A1,A2,A3,B,C等關鍵點
    
    參數:
    symbol (str): 股票代號
    period (str): 時間範圍
    resistance_levels (list): 阻力位列表
    patterns (list): 模式列表 (可選)
    """
    
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period,auto_adjust=True)                    
    data.index = pd.to_datetime(data.index)
    
    # 創建圖表
    plt.figure(figsize=(15, 10))
    
    # 繪製股價走勢
    plt.plot(data.index, data['Close'], label='收盤價', linewidth=1, color='black')
    
    # 繪製阻力位水平線
    for i, level in enumerate(resistance_levels):
        plt.axhline(y=level, color=f'C{i}', linestyle='--', alpha=0.7, 
                   label=f'阻力位 {i+1}: ${level:.2f}')
    
    # 如果有模式信息，標記關鍵點
    if patterns:
        for pattern_idx, pattern in enumerate(patterns):
            resistance_level = pattern['resistance_level']
            peaks = pattern['peaks']
            pullbacks = pattern['pullbacks']
            
            # 標記峰值 (A1, A2, A3)
            for i, peak in enumerate(peaks):
                if i < 3:  # 只標記前三個峰值
                    plt.scatter(peak['date'], peak['price'], 
                               color='red', marker='v', s=100, zorder=5)
                    plt.annotate(f'A{i+1}', (peak['date'], peak['price']), 
                                xytext=(5, 15), textcoords='offset points', 
                                fontweight='bold', color='red')
            
            # 標記回調低點 (B, C)
            for i, pullback in enumerate(pullbacks):
                if i < 2:  # 只標記前兩個回調低點
                    label = 'B' if i == 0 else 'C'
                    plt.scatter(pullback['date'], pullback['price'], 
                               color='green', marker='^', s=100, zorder=5)
                    plt.annotate(label, (pullback['date'], pullback['price']), 
                                xytext=(5, -15), textcoords='offset points', 
                                fontweight='bold', color='green')
            
            # 連接回調低點形成上升趨勢線
            if len(pullbacks) >= 2:
                pullback_dates = [p['date'] for p in pullbacks[:2]]
                pullback_prices = [p['price'] for p in pullbacks[:2]]
                plt.plot(pullback_dates, pullback_prices, 'g--', alpha=0.7, 
                        label=f'模式 {pattern_idx+1} 上升趨勢線')
    
    plt.title(f'{symbol} 共同阻力位與三次測試高位模式 ({period})')
    plt.xlabel('日期')
    plt.ylabel('價格 (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def enhanced_multi_timeframe_analysis(symbol, periods=["4mo", "6mo", "1y"]):
    """
    增強版多時間範圍分析：找出共同阻力位並可視化模式
    """
    all_patterns = {}
    
    for period in periods:
        print(f"\n分析時間範圍: {period}")
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period,auto_adjust=True)                    
        data.index = pd.to_datetime(data.index)
        
        patterns = find_patterns_in_data(data)
        all_patterns[period] = patterns
        
        if patterns:
            print(f"找到 {len(patterns)} 個模式")
            for i, pattern in enumerate(patterns):
                print(f"  模式 {i+1}: 阻力位 ${pattern['resistance_level']:.2f}")
        else:
            print("未找到模式")
    
    # 找出在多個時間範圍內都出現的阻力位
    common_resistance_levels = find_common_resistance_levels(all_patterns)
    
    if common_resistance_levels:
        print(f"\n共同阻力位: {common_resistance_levels}")
        
        # 找出與共同阻力位相關的模式
        common_patterns = []
        for period, patterns in all_patterns.items():
            for pattern in patterns:
                # 檢查模式阻力位是否接近任何共同阻力位
                for common_level in common_resistance_levels:
                    if abs(pattern['resistance_level'] - common_level) / common_level <= 0.03:
                        common_patterns.append(pattern)
                        break
        
        # 使用最長的時間範圍可視化這些阻力位和模式
        longest_period = periods[-1]
        visualize_common_resistance_levels(symbol, longest_period, 
                                          common_resistance_levels, common_patterns)
        
        # 詳細顯示每個模式
        print("\n模式詳細信息:")
        for i, pattern in enumerate(common_patterns):
            print(f"\n模式 #{i+1}:")
            print(f"阻力位: ${pattern['resistance_level']:.2f}")
            
            for j, peak in enumerate(pattern['peaks']):
                if j < 3:  # 只顯示前三個峰值
                    print(f"A{j+1}: {peak['date'].strftime('%Y-%m-%d')} - ${peak['price']:.2f}")
            
            for j, pullback in enumerate(pattern['pullbacks']):
                if j < 2:  # 只顯示前兩個回調
                    label = 'B' if j == 0 else 'C'
                    print(f"{label}: {pullback['date'].strftime('%Y-%m-%d')} - ${pullback['price']:.2f}")
        
        return common_resistance_levels, common_patterns
    else:
        print("\n未找到共同的阻力位")
        return None, None

def visualize_specific_pattern(symbol, period, pattern):
    """
    可視化特定模式，標記A1,A2,A3,B,C等關鍵點
    """

    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period,auto_adjust=True)                    
    data.index = pd.to_datetime(data.index)
    
    # 創建圖表
    plt.figure(figsize=(15, 10))
    
    # 繪製股價走勢
    plt.plot(data.index, data['Close'], label='收盤價', linewidth=1, color='black')
    
    # 標記峰值 (A1, A2, A3)
    peaks = pattern['peaks']
    for i, peak in enumerate(peaks):
        if i < 3:  # 只標記前三個峰值
            plt.scatter(peak['date'], peak['price'], 
                       color='red', marker='v', s=100, zorder=5)
            plt.annotate(f'A{i+1}', (peak['date'], peak['price']), 
                        xytext=(5, 15), textcoords='offset points', 
                        fontweight='bold', color='red', fontsize=12)
    
    # 標記回調低點 (B, C)
    pullbacks = pattern['pullbacks']
    for i, pullback in enumerate(pullbacks):
        if i < 2:  # 只標記前兩個回調低點
            label = 'B' if i == 0 else 'C'
            plt.scatter(pullback['date'], pullback['price'], 
                       color='green', marker='^', s=100, zorder=5)
            plt.annotate(label, (pullback['date'], pullback['price']), 
                        xytext=(5, -15), textcoords='offset points', 
                        fontweight='bold', color='green', fontsize=12)
    
    # 繪製阻力位水平線
    resistance_level = pattern['resistance_level']
    plt.axhline(y=resistance_level, color='red', linestyle='--', alpha=0.7, 
                label=f'阻力位: ${resistance_level:.2f}')
    
    # 連接回調低點形成上升趨勢線
    if len(pullbacks) >= 2:
        pullback_dates = [p['date'] for p in pullbacks[:2]]
        pullback_prices = [p['price'] for p in pullbacks[:2]]
        plt.plot(pullback_dates, pullback_prices, 'g--', alpha=0.7, 
                label='上升趨勢線')
        
        # 計算趨勢線斜率
        slope = (pullback_prices[1] - pullback_prices[0]) / (
            (pullback_dates[1] - pullback_dates[0]).days
        )
        print(f"上升趨勢線斜率: {slope:.4f} 元/天")
    
    # 標記模式持續時間
    first_peak_date = peaks[0]['date']
    last_peak_date = peaks[-1]['date']
    duration_days = (last_peak_date - first_peak_date).days
    plt.annotate(f'模式持續時間: {duration_days}天', 
                xy=(first_peak_date, resistance_level * 0.95),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5),
                fontsize=10)
    
    plt.title(f'{symbol} 三次測試高位模式 ({period})')
    plt.xlabel('日期')
    plt.ylabel('價格 (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 輸出模式統計信息
    print(f"\n模式統計信息:")
    print(f"阻力位: ${resistance_level:.2f}")
    print(f"模式持續時間: {duration_days} 天")
    print(f"測試次數: {len(peaks)}")
    
    for i, peak in enumerate(peaks):
        if i < 3:
            print(f"A{i+1}: {peak['date'].strftime('%Y-%m-%d')} - ${peak['price']:.2f}")
    
    for i, pullback in enumerate(pullbacks):
        if i < 2:
            label = 'B' if i == 0 else 'C'
            print(f"{label}: {pullback['date'].strftime('%Y-%m-%d')} - ${pullback['price']:.2f}")
            
            if i > 0:
                improvement = ((pullback['price'] - pullbacks[i-1]['price']) / 
                              pullbacks[i-1]['price'] * 100)
                print(f"  {label}較前一次回調上漲: {improvement:.2f}%")


def sliding_window_analysis(symbol, window_days=120, step_days=30):
    """
    滑動窗口分析：使用固定天數的窗口進行分析
    """
    # 下載足夠的數據（1年）
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1y",auto_adjust=True)    
    data.index = pd.to_datetime(data.index)
    
    # 使用滑動窗口分析
    patterns_by_window = {}
    
    for start_idx in range(0, len(data) - window_days, step_days):
        end_idx = start_idx + window_days
        window_data = data.iloc[start_idx:end_idx]
        
        window_start = window_data.index[0].strftime('%Y-%m-%d')
        window_end = window_data.index[-1].strftime('%Y-%m-%d')
        window_key = f"{window_start} 至 {window_end}"
        
        patterns = find_patterns_in_data(window_data)
        patterns_by_window[window_key] = patterns
    
    # 找出在多次窗口分析中都出現的阻力位
    common_levels = find_common_resistance_levels(patterns_by_window)
    
    if common_levels:
        print(f"在滑動窗口分析中找到 {len(common_levels)} 個共同阻力位")
        for i, level in enumerate(common_levels):
            print(f"  阻力位 {i+1}: ${level:.2f}")
        
        # 可視化
        visualize_common_resistance_levels(symbol, "1y", common_levels)
        
        return common_levels
    else:
        print("在滑動窗口分析中未找到共同阻力位")
        return None

# 使用範例
# if __name__ == "__main__":
#     # 在程式開始時呼叫
#     set_chinese_font()
#     # 設定股票代號
#     symbol = "0700.HK"  # 更改為你想要分析的股票
#     target_period = "6mo"
#     base_period = "4mo"
#     breakout_threshold = 0.01
    
#     print("=== 自適應阻力模式分析 ===")
    
#     # 方法1: 自適應分析
#     print(f"\n1. 自適應分析 (使用{base_period}識別，{target_period}驗證)")
#     patterns = adaptive_resistance_pattern(symbol, target_period=target_period, base_period=base_period)
    
#     if patterns:
#         for i, pattern in enumerate(patterns):
#             print(f"模式 {i+1}: 阻力位 ${pattern['resistance_level']:.2f}")
#             for j, peak in enumerate(pattern['peaks']):
#                 print(f"  峰值 {j+1}: {peak['date'].strftime('%Y-%m-%d')} - ${peak['price']:.2f}")
    
#     # 方法2: 多時間範圍分析
#     print("\n2. 多時間範圍分析")
#     common_levels = multi_timeframe_analysis(symbol, periods=["3mo", "4mo", "6mo", "1y"])
    
#     # 方法3: 滑動窗口分析
#     print("\n3. 滑動窗口分析")
#     sliding_levels = sliding_window_analysis(symbol, window_days=120, step_days=30)


# 使用範例
if __name__ == "__main__":

    set_chinese_font()
    # 設定股票代號
    symbol = "TSLA"  # 更改為你想要分析的股票
    #target_period = "6mo"
    #base_period = "4mo"
    breakout_threshold = 0.01
    
    print("=== 增強版多時間範圍分析 ===")
    
    # 使用增強版多時間範圍分析
    common_levels, common_patterns = enhanced_multi_timeframe_analysis(
        symbol, periods=["3mo", "4mo", "6mo"]
    )
    
    # 如果找到模式，可視化第一個模式
    if common_patterns:
        print("\n=== 詳細可視化第一個模式 ===")
        visualize_specific_pattern(symbol, "1y", common_patterns[0])
    
    # 也可以直接分析特定時間範圍的模式
    # print("\n=== 直接分析6個月數據 ===")

    # ticker = yf.Ticker(symbol)
    # data_6mo = ticker.history(period="6mo",auto_adjust=True)                    
    # data_6mo.index = pd.to_datetime(data_6mo.index)
    
    # patterns_6mo = find_patterns_in_data(data_6mo)
    # if patterns_6mo:
    #     print(f"在6個月數據中找到 {len(patterns_6mo)} 個模式")
    #     for i, pattern in enumerate(patterns_6mo):
    #         print(f"\n模式 {i+1}:")
    #         visualize_specific_pattern(symbol, "6mo", pattern)
    # else:
    #     print("在6個月數據中未找到模式")
        
        # 嘗試使用更寬鬆的參數
        # print("\n嘗試使用更寬鬆的參數...")
        # patterns_relaxed = find_patterns_in_data(
        #     data_6mo, 
        #     resistance_threshold=0.05,  # 放寬阻力位閾值
        #     breakout_threshold=0.02     # 放寬突破閾值
        # )
        
        # if patterns_relaxed:
        #     print(f"使用寬鬆參數找到 {len(patterns_relaxed)} 個模式")
        #     for i, pattern in enumerate(patterns_relaxed):
        #         visualize_specific_pattern(symbol, "6mo", pattern)