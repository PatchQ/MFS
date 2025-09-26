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
import mplfinance as mpf
from matplotlib.patches import Rectangle


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
        
def enhanced_resistance_pattern(symbol, periods=["3mo", "6mo", "1y"], 
                               resistance_threshold=0.03, breakout_threshold=0.01,
                               use_sliding_window=True, window_days=90, step_days=30):
    """
    增強版阻力模式識別：整合多時間範圍分析和滑動窗口分析
    
    參數:
    symbol (str): 股票代號
    periods (list): 多個時間範圍列表
    resistance_threshold (float): 阻力位識別閾值
    breakout_threshold (float): 突破檢查閾值
    use_sliding_window (bool): 是否使用滑動窗口分析
    window_days (int): 滑動窗口天數
    step_days (int): 滑動步長天數
    
    返回:
    dict: 包含分析結果的字典
    """
    
    print(f"=== {symbol} 增強版阻力模式分析 ===")
    print(f"時間範圍: {periods}")
    print(f"阻力位閾值: {resistance_threshold*100}%")
    print(f"突破閾值: {breakout_threshold*100}%")
    print("=" * 60)
    
    # 多時間範圍分析
    multi_timeframe_results = multi_timeframe_analysis_optimized(
        symbol, periods, resistance_threshold, breakout_threshold
    )
    
    # 滑動窗口分析（可選）
    sliding_window_results = None
    if use_sliding_window:
        sliding_window_results = sliding_window_analysis_optimized(
            symbol, window_days, step_days, resistance_threshold, breakout_threshold
        )
    
    # 整合結果
    all_patterns = []
    
    # 從多時間範圍分析中提取模式
    for period_result in multi_timeframe_results.values():
        all_patterns.extend(period_result.get('patterns', []))
    
    # 從滑動窗口分析中提取模式
    if sliding_window_results:
        all_patterns.extend(sliding_window_results.get('patterns', []))
    
    # 找出共同的阻力位
    common_levels = find_common_resistance_levels_optimized(all_patterns)
    
    # 過濾出最強的模式（基於共同阻力位）
    strongest_patterns = []
    if common_levels:
        for pattern in all_patterns:
            for level in common_levels:
                if abs(pattern['resistance_level'] - level) / level <= 0.02:
                    pattern['confidence'] = calculate_pattern_confidence(pattern, common_levels)
                    strongest_patterns.append(pattern)
                    break
    
    # 按置信度排序
    strongest_patterns.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    # 準備結果
    result = {
        'symbol': symbol,
        'periods_analyzed': periods,
        'multi_timeframe_results': multi_timeframe_results,
        'sliding_window_results': sliding_window_results,
        'common_resistance_levels': common_levels,
        'strongest_patterns': strongest_patterns,
        'total_patterns_found': len(all_patterns),
        'strong_patterns_count': len(strongest_patterns)
    }
    
    # 輸出結果摘要
    print_results_summary(result)
    
    # 可視化最強模式
    # if strongest_patterns:
    #     visualize_strongest_patterns(result)
    
    return result

def multi_timeframe_analysis_optimized(symbol, periods, resistance_threshold, breakout_threshold):
    """
    優化的多時間範圍分析
    """
    results = {}
    
    for period in periods:
        print(f"\n分析時間範圍: {period}")
        
        # 下載數據
        try:

            stock = yf.Ticker(symbol)
            data = stock.history(period=period,auto_adjust=True)    

            if data.empty:
                print(f"  ⚠️ 無法下載 {period} 數據")
                continue
                
            data.index = pd.to_datetime(data.index)
            
            # 識別模式
            patterns = find_patterns_in_data_optimized(
                data, resistance_threshold, breakout_threshold
            )
            
            results[period] = {
                'data_points': len(data),
                'patterns': patterns,
                'patterns_count': len(patterns)
            }
            
            print(f"  ✅ 找到 {len(patterns)} 個模式")
            
        except Exception as e:
            print(f"  ❌ 分析 {period} 時出錯: {e}")
            results[period] = {'error': str(e), 'patterns': []}
    
    return results

def sliding_window_analysis_optimized(symbol, window_days, step_days, resistance_threshold, breakout_threshold):
    """
    優化的滑動窗口分析
    """
    print(f"\n進行滑動窗口分析 (窗口: {window_days}天, 步長: {step_days}天)")
    
    # 下載足夠的數據（2年）
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="2y",auto_adjust=True)    

        if data.empty:
            print("  ⚠️ 無法下載數據")
            return None
            
        data.index = pd.to_datetime(data.index)
        
        # 滑動窗口分析
        window_patterns = []
        window_info = []
        
        for start_idx in range(0, len(data) - window_days, step_days):
            end_idx = start_idx + window_days
            window_data = data.iloc[start_idx:end_idx]
            
            window_start = window_data.index[0].strftime('%Y-%m-%d')
            window_end = window_data.index[-1].strftime('%Y-%m-%d')
            
            # 識別模式
            patterns = find_patterns_in_data_optimized(
                window_data, resistance_threshold, breakout_threshold
            )
            
            if patterns:
                window_patterns.extend(patterns)
                window_info.append({
                    'window': f"{window_start} 至 {window_end}",
                    'patterns_count': len(patterns)
                })
        
        print(f"  ✅ 在 {len(window_info)} 個窗口中找到 {len(window_patterns)} 個模式")
        
        return {
            'total_windows': len(range(0, len(data) - window_days, step_days)),
            'windows_with_patterns': len(window_info),
            'total_patterns': len(window_patterns),
            'patterns': window_patterns,
            'window_info': window_info
        }
        
    except Exception as e:
        print(f"  ❌ 滑動窗口分析出錯: {e}")
        return None

def find_patterns_in_data_optimized(data, resistance_threshold, breakout_threshold):
    """
    優化的模式識別函數
    """
    # 自適應參數調整
    n_points = len(data)
    
    # 根據數據點數量調整參數
    if n_points < 60:  # 數據點較少
        min_peaks = 2
        peak_window = max(3, int(n_points * 0.08))
    elif n_points < 120:  # 中等數據量
        min_peaks = 3
        peak_window = max(5, int(n_points * 0.06))
    else:  # 數據點較多
        min_peaks = 3
        peak_window = max(5, int(n_points * 0.04))
    
    # 找出所有高點
    highs = find_peaks_optimized(data, window=peak_window)
    
    if len(highs) < min_peaks:
        return []
    
    # 按價格分組高點
    groups = group_similar_highs_optimized(highs, threshold=resistance_threshold)
    
    # 找出至少有3個高點的群組
    valid_groups = {level: sorted(peaks, key=lambda x: x['date']) 
                   for level, peaks in groups.items() if len(peaks) >= min_peaks}
    
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
            between_prices = data.iloc[start_idx:end_idx]['Close']
            
            if any(price > level * (1 + breakout_threshold) for price in between_prices):
                valid_pattern = False
                break
        
        if valid_pattern:
            # 找出回調低點
            pullbacks = []
            for i in range(len(recent_peaks)-1):
                start_idx = recent_peaks[i]['index']
                end_idx = recent_peaks[i+1]['index']
                pullback_data = data.iloc[start_idx:end_idx]
                
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
                    # 計算模式強度
                    strength = calculate_pattern_strength_optimized(recent_peaks, pullbacks, level)
                    
                    patterns.append({
                        'resistance_level': level,
                        'peaks': recent_peaks,
                        'pullbacks': pullbacks,
                        'strength': strength,
                        'data_points': n_points,
                        'peak_window': peak_window
                    })
    
    return patterns

def find_peaks_optimized(data, window=None):
    """
    優化的峰值檢測
    """
    if window is None:
        # 根據數據長度自動確定窗口大小
        window = max(3, min(10, int(len(data) * 0.05)))
    
    highs = []
    for i in range(window, len(data) - window):
        if data['Close'].iloc[i] == data['Close'].iloc[i-window:i+window+1].max():
            highs.append({
                'date': data.index[i],
                'price': data['Close'].iloc[i],
                'index': i
            })
    
    return highs

def group_similar_highs_optimized(highs, threshold=0.03):
    """
    優化的高點分組
    """
    if not highs:
        return {}
    
    # 按價格排序高點
    sorted_highs = sorted(highs, key=lambda x: x['price'])
    
    groups = {}
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
            if len(current_group) >= 2:
                groups[current_avg] = current_group
            
            current_group = [sorted_highs[i]]
            current_avg = price
    
    # 處理最後一個群組
    if len(current_group) >= 2:
        groups[current_avg] = current_group
    
    return groups

def calculate_pattern_strength_optimized(peaks, pullbacks, resistance_level):
    """
    優化的模式強度計算
    """
    # 峰值相似度
    peak_prices = [p['price'] for p in peaks]
    peak_cv = np.std(peak_prices) / np.mean(peak_prices)
    
    # 回調深度
    pullback_depths = []
    for i, peak in enumerate(peaks):
        if i < len(pullbacks):
            depth = (peak['price'] - pullbacks[i]['price']) / peak['price']
            pullback_depths.append(depth)
    
    avg_depth = np.mean(pullback_depths) if pullback_depths else 0
    
    # 回調低點上升幅度
    pullback_rise = 0
    if len(pullbacks) >= 2:
        rise = (pullbacks[1]['price'] - pullbacks[0]['price']) / pullbacks[0]['price']
        pullback_rise = max(0, rise)
    
    # 時間跨度（較長的模式通常更可靠）
    time_span = (peaks[-1]['date'] - peaks[0]['date']).days
    time_score = min(1.0, time_span / 90)  # 以90天為基準
    
    # 綜合評分
    similarity_score = max(0, 1 - peak_cv * 10)
    depth_score = max(0, 1 - avg_depth * 5)
    rise_score = min(1.0, pullback_rise * 10)
    
    strength = (similarity_score * 0.4 + depth_score * 0.3 + 
                rise_score * 0.2 + time_score * 0.1)
    
    return min(1.0, strength) * 10  # 轉換為0-10分

def find_common_resistance_levels_optimized(patterns, threshold=0.02):
    """
    優化的共同阻力位識別
    """
    if not patterns:
        return []
    
    # 收集所有阻力位
    levels = [p['resistance_level'] for p in patterns]
    
    # 分組相似的阻力位
    level_groups = {}
    for level in levels:
        found = False
        for group_level in level_groups:
            if abs(level - group_level) / group_level <= threshold:
                level_groups[group_level].append(level)
                found = True
                break
        
        if not found:
            level_groups[level] = [level]
    
    # 計算每組的平均值和出現次數
    common_levels = []
    for group_level, level_list in level_groups.items():
        avg_level = np.mean(level_list)
        frequency = len(level_list)
        
        # 只保留出現多次的阻力位
        if frequency >= 2:
            common_levels.append(avg_level)
    
    # 按出現頻率排序
    common_levels.sort()
    return common_levels

def calculate_pattern_confidence(pattern, common_levels):
    """
    計算模式置信度（基於與共同阻力位的接近程度和模式強度）
    """
    # 檢查與共同阻力位的接近程度
    level_match = 0
    for common_level in common_levels:
        deviation = abs(pattern['resistance_level'] - common_level) / common_level
        if deviation <= 0.02:
            level_match = 1 - deviation * 10  # 偏差越小，匹配度越高
            break
    
    # 結合模式強度
    strength = pattern.get('strength', 0) / 10  # 轉換為0-1
    
    confidence = (level_match * 0.6 + strength * 0.4) * 10  # 轉換為0-10分
    
    return min(10.0, confidence)

def print_results_summary(result):
    """
    輸出結果摘要
    """
    print("\n" + "=" * 60)
    print("分析結果摘要")
    print("=" * 60)
    
    symbol = result['symbol']
    total_patterns = result['total_patterns_found']
    strong_patterns = result['strong_patterns_count']
    common_levels = result['common_resistance_levels']
    
    print(f"股票: {symbol}")
    print(f"總共找到模式: {total_patterns} 個")
    print(f"強模式數量: {strong_patterns} 個")
    print(f"共同阻力位: {len(common_levels)} 個")
    
    if common_levels:
        print("共同阻力位水平:")
        for i, level in enumerate(common_levels):
            print(f"  {i+1}. ${level:.2f}")
    
    # 多時間範圍分析結果
    print("\n多時間範圍分析:")
    for period, period_result in result['multi_timeframe_results'].items():
        patterns_count = period_result.get('patterns_count', 0)
        data_points = period_result.get('data_points', 0)
        print(f"  {period}: {patterns_count} 個模式 ({data_points} 個數據點)")
    
    # 滑動窗口分析結果
    if result['sliding_window_results']:
        sw_result = result['sliding_window_results']
        print(f"滑動窗口分析: {sw_result['windows_with_patterns']}/{sw_result['total_windows']} 個窗口找到模式")
    
    # 最強模式
    if result['strongest_patterns']:
        print(f"\n最強模式 (按置信度排序):")
        for i, pattern in enumerate(result['strongest_patterns'][:3]):  # 只顯示前3個
            confidence = pattern.get('confidence', 0)
            strength = pattern.get('strength', 0)
            level = pattern['resistance_level']
            print(f"  {i+1}. 阻力位: ${level:.2f}, 強度: {strength:.1f}/10, 置信度: {confidence:.1f}/10")

def visualize_strongest_patterns(result):
    """
    可視化最強模式
    """
    symbol = result['symbol']
    strongest_patterns = result['strongest_patterns']
    
    if not strongest_patterns:
        return
    
    # 使用最長的時間範圍進行可視化
    longest_period = result['periods_analyzed'][-1]
    
    stock = yf.Ticker(symbol)
    data = stock.history(period=longest_period,auto_adjust=True)    
    data.index = pd.to_datetime(data.index)
    
    # 創建圖表
    plt.figure(figsize=(15, 10))
    plt.plot(data.index, data['Close'], label='收盤價', linewidth=1, color='black')
    
    # 繪製最強模式
    for i, pattern in enumerate(strongest_patterns[:2]):  # 只繪製前2個最強模式
        resistance_level = pattern['resistance_level']
        peaks = pattern['peaks']
        pullbacks = pattern['pullbacks']
        confidence = pattern.get('confidence', 0)
        
        # 繪製阻力位
        plt.axhline(y=resistance_level, color=f'C{i}', linestyle='--', alpha=0.7,
                   label=f'模式{i+1}: ${resistance_level:.2f} (置信度: {confidence:.1f}/10)')
        
        # 標記峰值
        for j, peak in enumerate(peaks):
            if j < 3:
                plt.scatter(peak['date'], peak['price'], 
                           color=f'C{i}', marker='v', s=80, zorder=5)
                plt.annotate(f'A{j+1}', (peak['date'], peak['price']),
                            xytext=(5, 10+j*5), textcoords='offset points',
                            fontweight='bold', color=f'C{i}', fontsize=10)
        
        # 標記回調低點
        for j, pullback in enumerate(pullbacks):
            if j < 2:
                label = 'B' if j == 0 else 'C'
                plt.scatter(pullback['date'], pullback['price'],
                           color=f'C{i}', marker='^', s=80, zorder=5)
                plt.annotate(label, (pullback['date'], pullback['price']),
                            xytext=(5, -15-j*5), textcoords='offset points',
                            fontweight='bold', color=f'C{i}', fontsize=10)
    
    plt.title(f'{symbol} 最強阻力模式分析 ({longest_period})')
    plt.xlabel('日期')
    plt.ylabel('價格')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def enhanced_resistance_pattern_with_candlestick(symbol, periods=["3mo", "6mo", "1y"], 
                                               resistance_threshold=0.03, breakout_threshold=0.01,
                                               use_sliding_window=True, window_days=90, step_days=30):
    """
    增強版阻力模式識別：使用陰陽燭顯示，並識別D低點
    """
    # 調用原有的分析函數
    result = enhanced_resistance_pattern(
        symbol, periods, resistance_threshold, breakout_threshold,
        use_sliding_window, window_days, step_days
    )
    
    # 使用陰陽燭可視化最強模式
    if result and result['strongest_patterns']:
        visualize_strongest_patterns_with_candlestick(result)
    
    return result

def visualize_strongest_patterns_with_candlestick(result):
    """
    使用陰陽燭可視化最強模式，並標記A1,A2,A3,B,C,D等關鍵點
    """
    symbol = result['symbol']
    strongest_patterns = result['strongest_patterns']
    
    if not strongest_patterns:
        return
    
    # 使用最長的時間範圍進行可視化
    longest_period = result['periods_analyzed'][-1]
    stock = yf.Ticker(symbol)
    data = stock.history(period=longest_period,auto_adjust=True)        
    data.index = pd.to_datetime(data.index)
    
    # 準備陰陽燭數據
    ohlc_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    ohlc_data.columns = ['open', 'high', 'low', 'close', 'volume']  # 符合mplfinance的命名約定
    
    # 創建自定義樣式
    mc = mpf.make_marketcolors(
        up='green', down='red',
        wick={'up':'green', 'down':'red'},
        volume={'up':'green', 'down':'red'}
    )
    
    custom_style = mpf.make_mpf_style(marketcolors=mc, 
                                    gridstyle='--', 
                                    gridcolor='gray',
                                    base_mpf_style='charles', 
                                    rc={
                                        'font.family': 'Microsoft YaHei',
                                        'axes.unicode_minus': False
                                        }
                                    )
    
    # 創建自定義繪圖函數來標記關鍵點
    apds = []
    
    # 為每個模式添加標記
    for i, pattern in enumerate(strongest_patterns[:2]):  # 只繪製前2個最強模式
        resistance_level = pattern['resistance_level']
        peaks = pattern['peaks']
        pullbacks = pattern['pullbacks']
        confidence = pattern.get('confidence', 0)
        
        # 識別D低點（如果存在）
        d_point = identify_d_point(data, peaks, pullbacks)
        
        # 添加阻力位線 - 修復：確保長度匹配
        resistance_series = pd.Series([resistance_level] * len(data), index=data.index)
        apds.append(mpf.make_addplot(
            resistance_series,
            color=f'C{i}', linestyle='--', alpha=0.7
        ))
        
        # 添加標記點 - 修復：創建與主數據相同索引的序列
        peak_prices = []
        peak_dates = []
        for p in peaks:
            if p['date'] in data.index:  # 確保日期在數據索引中
                peak_prices.append(p['price'])
                peak_dates.append(p['date'])
        
        if peak_prices:
            # 創建一個全為NaN的序列，只在峰值點有值
            peak_series = pd.Series([np.nan] * len(data), index=data.index)
            for date, price in zip(peak_dates, peak_prices):
                peak_series[date] = price
                
            mark_peaks = mpf.make_addplot(
                peak_series,
                type='scatter', marker='v', markersize=80, color=f'C{i}'
            )
            apds.append(mark_peaks)
        
        # 添加回調點標記
        pullback_prices = []
        pullback_dates = []
        for p in pullbacks:
            if p['date'] in data.index:  # 確保日期在數據索引中
                pullback_prices.append(p['price'])
                pullback_dates.append(p['date'])
        
        if pullback_prices:
            # 創建一個全為NaN的序列，只在回調點有值
            pullback_series = pd.Series([np.nan] * len(data), index=data.index)
            for date, price in zip(pullback_dates, pullback_prices):
                pullback_series[date] = price
                
            mark_pullbacks = mpf.make_addplot(
                pullback_series,
                type='scatter', marker='^', markersize=80, color=f'C{i}'
            )
            apds.append(mark_pullbacks)
        
        # 添加D點標記（如果存在）
        if d_point and d_point['date'] in data.index:
            # 創建一個全為NaN的序列，只在D點有值
            d_series = pd.Series([np.nan] * len(data), index=data.index)
            d_series[d_point['date']] = d_point['price']
            
            mark_d = mpf.make_addplot(
                d_series,
                type='scatter', marker='s', markersize=100, color='purple'
            )
            apds.append(mark_d)
    
    try:
        # 創建圖表
        fig, axes = mpf.plot(
            ohlc_data,
            type='candle',
            style=custom_style,
            addplot=apds,
            volume=True,
            figsize=(15, 10),
            returnfig=True,
            title=f'\n{symbol} 最強阻力模式分析 - 陰陽燭圖 ({longest_period})',
            ylabel='價格'
        )
        
        # 添加文字標註
        ax_main = axes[0]
        
        # 為每個模式添加文字標註
        for i, pattern in enumerate(strongest_patterns[:2]):
            resistance_level = pattern['resistance_level']
            peaks = pattern['peaks']
            pullbacks = pattern['pullbacks']
            confidence = pattern.get('confidence', 0)
            
            # 識別D低點（如果存在）
            d_point = identify_d_point(data, peaks, pullbacks)
            
            # 標註峰值 (A1, A2, A3)
            for j, peak in enumerate(peaks):
                if j < 3 and peak['date'] in data.index:
                    ax_main.annotate(f'A{j+1}', 
                                   xy=(peak['date'], peak['price']),
                                   xytext=(5, 10+j*5), textcoords='offset points',
                                   fontweight='bold', color=f'C{i}', fontsize=12)
            
            # 標註回調低點 (B, C)
            for j, pullback in enumerate(pullbacks):
                if j < 2 and pullback['date'] in data.index:
                    label = 'B' if j == 0 else 'C'
                    ax_main.annotate(label, 
                                   xy=(pullback['date'], pullback['price']),
                                   xytext=(5, -15-j*5), textcoords='offset points',
                                   fontweight='bold', color=f'C{i}', fontsize=12)
            
            # 標註D點（如果存在）
            if d_point and d_point['date'] in data.index:
                ax_main.annotate('D', 
                               xy=(d_point['date'], d_point['price']),
                               xytext=(5, -25), textcoords='offset points',
                               fontweight='bold', color='purple', fontsize=12)
            
            # 添加阻力位標籤
            ax_main.annotate(f'阻力位 {i+1}: ${resistance_level:.2f} (置信度: {confidence:.1f}/10)', 
                           xy=(data.index[-10], resistance_level),
                           xytext=(10, 0), textcoords='offset points',
                           fontweight='bold', color=f'C{i}', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        
        # 添加圖例
        legend_elements = [
            plt.Line2D([0], [0], color='C0', marker='v', linestyle='--', label='模式1: A點(峰值)'),
            plt.Line2D([0], [0], color='C0', marker='^', linestyle='None', label='模式1: B/C點(回調)'),
            plt.Line2D([0], [0], color='C1', marker='v', linestyle='--', label='模式2: A點(峰值)'),
            plt.Line2D([0], [0], color='C1', marker='^', linestyle='None', label='模式2: B/C點(回調)'),
            plt.Line2D([0], [0], color='purple', marker='s', linestyle='None', label='D點(更高低點)')
        ]
        
        ax_main.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"可視化過程中出錯: {e}")
        # 如果可視化失敗，使用簡單的線圖
        visualize_strongest_patterns_simple(result)
    
    # 輸出模式詳細信息
    print("\n模式詳細信息:")
    for i, pattern in enumerate(strongest_patterns[:2]):
        print(f"\n模式 #{i+1}:")
        print(f"阻力位: ${pattern['resistance_level']:.2f}")
        print(f"置信度: {pattern.get('confidence', 0):.1f}/10")
        
        for j, peak in enumerate(pattern['peaks']):
            if j < 3:
                print(f"A{j+1}: {peak['date'].strftime('%Y-%m-%d')} - ${peak['price']:.2f}")
        
        for j, pullback in enumerate(pattern['pullbacks']):
            if j < 2:
                label = 'B' if j == 0 else 'C'
                print(f"{label}: {pullback['date'].strftime('%Y-%m-%d')} - ${pullback['price']:.2f}")
                
                if j > 0:
                    improvement = ((pullback['price'] - pattern['pullbacks'][j-1]['price']) / 
                                  pattern['pullbacks'][j-1]['price'] * 100)
                    print(f"  {label}較前一次回調上漲: {improvement:.2f}%")
        
        # 顯示D點信息（如果存在）
        d_point = identify_d_point(data, pattern['peaks'], pattern['pullbacks'])
        if d_point:
            print(f"D: {d_point['date'].strftime('%Y-%m-%d')} - ${d_point['price']:.2f}")
            # 計算D點相對於C點的變化
            if len(pattern['pullbacks']) >= 2:
                c_point = pattern['pullbacks'][1]
                d_change = ((d_point['price'] - c_point['price']) / c_point['price'] * 100)
                print(f"  D點較C點變化: {d_change:.2f}%")
                
                # 判斷D點的意義
                if d_point['price'] > c_point['price']:
                    print("  → D點高於C點，形成更高低點，是看漲信號")
                else:
                    print("  → D點低於C點，可能破壞上升趨勢")

def visualize_strongest_patterns_simple(result):
    """
    簡單版本的可視化（使用線圖而不是陰陽燭）
    """
    symbol = result['symbol']
    strongest_patterns = result['strongest_patterns']
    
    if not strongest_patterns:
        return
    
    # 使用最長的時間範圍進行可視化
    longest_period = result['periods_analyzed'][-1]
    stock = yf.Ticker(symbol)
    data = stock.history(period=longest_period,auto_adjust=True)        
    data.index = pd.to_datetime(data.index)
    
    # 創建圖表
    plt.figure(figsize=(15, 10))
    plt.plot(data.index, data['Close'], label='收盤價', linewidth=1, color='black')
    
    # 繪製最強模式
    for i, pattern in enumerate(strongest_patterns[:2]):  # 只繪製前2個最強模式
        resistance_level = pattern['resistance_level']
        peaks = pattern['peaks']
        pullbacks = pattern['pullbacks']
        confidence = pattern.get('confidence', 0)
        
        # 繪製阻力位
        plt.axhline(y=resistance_level, color=f'C{i}', linestyle='--', alpha=0.7,
                   label=f'模式{i+1}: ${resistance_level:.2f} (置信度: {confidence:.1f}/10)')
        
        # 標記峰值
        for j, peak in enumerate(peaks):
            if j < 3 and peak['date'] in data.index:
                plt.scatter(peak['date'], peak['price'], 
                           color=f'C{i}', marker='v', s=80, zorder=5)
                plt.annotate(f'A{j+1}', (peak['date'], peak['price']),
                            xytext=(5, 10+j*5), textcoords='offset points',
                            fontweight='bold', color=f'C{i}', fontsize=10)
        
        # 標記回調低點
        for j, pullback in enumerate(pullbacks):
            if j < 2 and pullback['date'] in data.index:
                label = 'B' if j == 0 else 'C'
                plt.scatter(pullback['date'], pullback['price'],
                           color=f'C{i}', marker='^', s=80, zorder=5)
                plt.annotate(label, (pullback['date'], pullback['price']),
                            xytext=(5, -15-j*5), textcoords='offset points',
                            fontweight='bold', color=f'C{i}', fontsize=10)
        
        # 識別並標記D低點（如果存在）
        d_point = identify_d_point(data, peaks, pullbacks)
        if d_point and d_point['date'] in data.index:
            plt.scatter(d_point['date'], d_point['price'],
                       color='purple', marker='s', s=100, zorder=5)
            plt.annotate('D', (d_point['date'], d_point['price']),
                        xytext=(5, -25), textcoords='offset points',
                        fontweight='bold', color='purple', fontsize=12)
            
            # 繪製上升趨勢線（連接B、C、D點）
            trend_points = []
            trend_dates = []
            
            # 添加B點
            if len(pullbacks) > 0:
                trend_points.append(pullbacks[0]['price'])
                trend_dates.append(pullbacks[0]['date'])
            
            # 添加C點
            if len(pullbacks) > 1:
                trend_points.append(pullbacks[1]['price'])
                trend_dates.append(pullbacks[1]['date'])
            
            # 添加D點
            trend_points.append(d_point['price'])
            trend_dates.append(d_point['date'])
            
            # 繪製趨勢線
            plt.plot(trend_dates, trend_points, 'g--', alpha=0.7, 
                    label=f'模式{i+1} 上升趨勢線')
    
    plt.title(f'{symbol} 最強阻力模式分析 ({longest_period})')
    plt.xlabel('日期')
    plt.ylabel('價格 (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def identify_d_point(data, peaks, pullbacks, lookback_days=30):
    """
    識別D低點：在第三次測試高位後出現的新低點，且D點必須高於C點
    
    參數:
    data: 股票數據
    peaks: 峰值列表 (A1, A2, A3)
    pullbacks: 回調低點列表 (B, C)
    lookback_days: 在最後一個峰值後多少天內尋找D點
    
    返回:
    dict: D點信息 (如果存在)
    """
    if len(peaks) < 3 or len(pullbacks) < 2:
        return None
    
    # 獲取最後一個峰值 (A3) 的日期
    last_peak_date = peaks[-1]['date']
    
    # 找到A3在數據中的位置
    if last_peak_date not in data.index:
        return None
    
    last_peak_idx = data.index.get_loc(last_peak_date)
    
    # 計算搜索範圍的結束索引
    end_idx = min(last_peak_idx + lookback_days, len(data) - 1)
    
    # 在A3之後的數據中尋找局部低點
    search_data = data.iloc[last_peak_idx:end_idx]
    
    if len(search_data) < 5:  # 數據點太少
        return None
    
    # 使用滑動窗口尋找局部低點
    window_size = min(5, len(search_data) // 3)
    potential_d_points = []
    
    for i in range(window_size, len(search_data) - window_size):
        current_low = search_data['Low'].iloc[i]
        
        # 檢查是否為窗口內的最低點
        if current_low == search_data['Low'].iloc[i-window_size:i+window_size+1].min():
            potential_d_points.append({
                'date': search_data.index[i],
                'price': current_low,
                'index': i + last_peak_idx
            })
    
    if not potential_d_points:
        return None
    
    # 選擇最低的點作為D點候選
    d_candidate = min(potential_d_points, key=lambda x: x['price'])
    
    # 檢查D點是否符合條件（D點必須高於C點，且高於B點）
    if len(pullbacks) >= 2:
        b_point = pullbacks[0]['price']
        c_point = pullbacks[1]['price']
        
        # 修改條件：D點應該高於C點，且高於B點（形成更高的低點）
        if d_candidate['price'] > c_point and d_candidate['price'] > b_point:
            return d_candidate
    
    return None


# 使用範例
if __name__ == "__main__":

    # 高級用法 - 自定義參數
    result = enhanced_resistance_pattern_with_candlestick(
        "9879.HK",
        periods=["2mo", "4mo", "6mo", "8mo", "1y"],
        resistance_threshold=0.02,
        breakout_threshold=0.015,
        use_sliding_window=False,
        window_days=60,
        step_days=20
    )

    # # 如果找到模式，可視化第一個模式
    # if result and result['strongest_patterns']:
    #     print("\n=== 詳細可視化第一個模式 ===")
    #     # 使用簡單版本可視化
    #     visualize_strongest_patterns_simple(result)