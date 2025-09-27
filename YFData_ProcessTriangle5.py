import pandas as pd
import numpy as np
import yfinance as yf
import concurrent.futures as cf
import os
from tqdm import tqdm
import time as t
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


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
RESISTANCE_RATE=0.005
BREAKOUT_RATE=0.015
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
    
        
def find_resistance_test_pattern(sno,stype):
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
    df = pd.read_excel(PATH+"/"+stype+"/"+sno+".xlsx",index_col="Date")   
    stock = df.tail(DAYS)            
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
        'period': DAYS,
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
        print(f"在 {sno} 的 {DAYS} 資料中未找到符合條件的模式")
        return
    
    print(f"股票 {sno} 的阻力位測試模式分析（期間: {DAYS}）")
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
    可視化阻力位測試模式
    """

    result = find_resistance_test_pattern(sno, stype)
    
    if result is None or not result['patterns']:
        print("沒有可視化的模式")
        return
    
    if pattern_index >= len(result['patterns']):
        print(f"模式索引 {pattern_index} 超出範圍")
        return
    
    pattern = result['patterns'][pattern_index]

    df = pd.read_excel(PATH+"/"+stype+"/"+sno+".xlsx",index_col="Date")   
    stock = df.tail(DAYS)            
    stock.index = pd.to_datetime(stock.index,utc=True).tz_convert('Asia/Shanghai')       

    
    # 創建圖表
    plt.figure(figsize=(15, 10))
    
    # 繪製股價走勢
    plt.plot(stock.index, stock['Close'], label='收盤價', linewidth=1, color='black')
    
    # 標記峰值
    for i, peak in enumerate(pattern['peaks']):
        # 確保日期是正確的格式
        if not isinstance(peak['date'], (pd.Timestamp, datetime)):
            peak_date = stock.index[peak['date']] if isinstance(peak['date'], int) else pd.to_datetime(peak['date'])
        else:
            peak_date = peak['date']
            
        plt.scatter(peak_date, peak['price'], color='red', marker='v', s=100, zorder=5)
        plt.annotate(f'A{i+1}', (peak_date, peak['price']), 
                    xytext=(5, 15), textcoords='offset points', fontweight='bold')
    
    # 標記回調低點
    for i, pullback in enumerate(pattern['pullbacks']):
        if not isinstance(pullback['date'], (pd.Timestamp, datetime)):
            pullback_date = stock.index[pullback['date']] if isinstance(pullback['date'], int) else pd.to_datetime(pullback['date'])
        else:
            pullback_date = pullback['date']
            
        plt.scatter(pullback_date, pullback['price'], color='green', marker='^', s=100, zorder=5)
        plt.annotate(f'{"B" if i == 0 else "C"}', (pullback_date, pullback['price']), 
                    xytext=(5, -15), textcoords='offset points', fontweight='bold')
    
    # 繪製阻力位水平線
    resistance_level = pattern['resistance_level']
    plt.axhline(y=resistance_level, color='red', linestyle='--', alpha=0.7, 
                label=f'阻力位: ${resistance_level:.2f}')
    
    # 繪製突破檢查線
    breakout_level = resistance_level * (1 + BREAKOUT_RATE)
    plt.axhline(y=breakout_level, color='orange', linestyle=':', alpha=0.5, 
                label=f'突破檢查線: ${breakout_level:.2f} (+{BREAKOUT_RATE*100}%)')
    
    # 添加趨勢線 (連接回調低點)
    if len(pattern['pullbacks']) >= 2:
        pullback_dates = []
        pullback_prices = []
        
        for pullback in pattern['pullbacks']:
            if not isinstance(pullback['date'], (pd.Timestamp, datetime)):
                pullback_date = stock.index[pullback['date']] if isinstance(pullback['date'], int) else pd.to_datetime(pullback['date'])
            else:
                pullback_date = pullback['date']
                
            pullback_dates.append(pullback_date)
            pullback_prices.append(pullback['price'])
            
        plt.plot(pullback_dates, pullback_prices, 'g--', alpha=0.7, label='上升趨勢線')
    
    plt.title(f'{sno} 三次測試高位模式 (強度: {pattern["pattern_strength"]:.1f}/10)')
    plt.xlabel('日期')
    plt.ylabel('價格 (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main(stype):

    snolist = list(map(lambda s: s.replace(".xlsx", ""), os.listdir(PATH+"/"+stype)))
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