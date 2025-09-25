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
    

def CheckTriangle(sno, stype, days, resistance_threshold=0.05, min_days_between_peaks=10):
    """
    找出股票三次測試同一高位的模式
    參數:
    sno (str): 股票代號
    period (str): 資料期間
    resistance_threshold (float): 阻力位識別閾值 (百分比)
    min_days_between_peaks (int): 峰值之間的最小天數
    
    返回:
    dict: 包含模式資訊的字典
    """
    
    stock = pd.read_excel(PATH+"/"+stype+"/"+sno+".xlsx",index_col="Date")   
    stock = convertData(stock.tail(days))
    stock.index = pd.to_datetime(stock.index,utc=True).tz_convert('Asia/Shanghai')

    print(stock)

    if stock.empty:
        print("無法下載資料，請檢查股票代號和期間設定")
        return None
    
    # 確保索引是DatetimeIndex
    stock.index = pd.to_datetime(stock.index,utc=True).tz_convert('Asia/Shanghai')
    
    # 計算價格變動
    stock['Price_Change'] = stock['Close'].pct_change()
    
    # 找出局部高點 (峰值)
    peaks = find_peaks(stock, window=5)  # 使用5天窗口找峰值
    
    if len(peaks) < 3:
        print("資料中不足3個峰值，無法識別模式")
        return None
    
    # 找出可能的阻力位 (相似的高點群組)
    resistance_levels = group_similar_highs(peaks, threshold=resistance_threshold)
    
    # 尋找符合條件的模式
    patterns = []
    
    for resistance_level, peaks_in_level in resistance_levels.items():
        if len(peaks_in_level) >= 3:  # 至少3次測試同一阻力位
            # 按時間排序峰值
            sorted_peaks = sorted(peaks_in_level, key=lambda x: x['date'])
            
            # 檢查是否符合模式條件
            valid_pattern = check_pattern_conditions(stock, sorted_peaks, min_days_between_peaks)
            
            if valid_pattern:
                patterns.append({
                    'resistance_level': resistance_level,
                    'peaks': sorted_peaks,
                    'pullbacks': valid_pattern['pullbacks'],
                    'pattern_strength': calculate_pattern_strength(valid_pattern)
                })
    
    return {
        'sno': sno,
        'days': days,
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

def group_similar_highs(peaks, threshold=0.02):
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
        if abs(price - current_avg) / current_avg <= threshold:
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

def check_pattern_conditions(stock_data, peaks, min_days_between_peaks):
    """
    檢查是否符合三次測試同一高位的模式條件
    """
    if len(peaks) < 3:
        return None
    
    # 檢查峰值之間的時間間隔 - 修復這裡的錯誤
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

def analyze_resistance_patterns(sno, stype, days ,resistance_threshold):
    """
    分析並顯示阻力位測試模式
    """
    #result = find_resistance_test_pattern(days,period,resistance_threshold=resistance_threshold)
    result = CheckTriangle(sno, stype, days, resistance_threshold, min_days_between_peaks=10)
    
    if result is None or not result['patterns']:
        print(f"在 {sno} 的 {days} 資料中未找到符合條件的模式")
        return
    
    print(f"股票 {sno} 的阻力位測試模式分析（期間: {days}）")
    print("=" * 80)
    
    for i, pattern in enumerate(result['patterns']):
        print(f"\n模式 #{i+1} (強度: {pattern['pattern_strength']:.1f}/10)")
        print(f"阻力位: ${pattern['resistance_level']:.2f}")
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
            
            print()
    
    return result


# 簡化版本 - 更穩定的實現
def simple_resistance_pattern(symbol, period="1y", lookback_days=30, resistance_threshold=0.05):
    """
    簡化版本的三次測試高位模式識別
    """
    # 下載股票資料
    ticker = yf.Ticker(symbol)
    stock = ticker.history(period=period,auto_adjust=True)    
    stock = convertData(stock)
    stock.index = pd.to_datetime(stock.index)
    
    # 找出最近的高點作為潛在阻力位
    recent_high = stock['Close'].tail(lookback_days).max()
    resistance_level = recent_high
    
    # 找出所有接近這個阻力位的點
    resistance_points = []
    for i in range(len(stock)):
        if abs(stock['Close'].iloc[i] - resistance_level) / resistance_level <= resistance_threshold:
            resistance_points.append({
                'date': stock.index[i],
                'price': stock['Close'].iloc[i],
                'index': i
            })
    
    # 如果至少有3個點，嘗試識別模式
    if len(resistance_points) >= 3:
        # 按時間排序
        resistance_points.sort(key=lambda x: x['date'])
        
        # 找出回調低點
        pullbacks = []
        for i in range(len(resistance_points)-1):
            start_idx = resistance_points[i]['index']
            end_idx = resistance_points[i+1]['index']
            
            # 在兩個阻力點之間找最低點
            if end_idx > start_idx:
                pullback_data = stock.iloc[start_idx:end_idx]
                min_idx = pullback_data['Close'].idxmin()
                min_price = pullback_data['Close'].min()
                
                pullbacks.append({
                    'date': min_idx,
                    'price': min_price
                })
        
        # 檢查回調低點是否依次抬高
        valid_pattern = True
        if len(pullbacks) >= 2:
            for i in range(1, len(pullbacks)):
                if pullbacks[i]['price'] <= pullbacks[i-1]['price']:
                    valid_pattern = False
                    break
        
        if valid_pattern and len(pullbacks) >= 2:
            print(f"找到三次測試高位模式!")
            print(f"阻力位: ${resistance_level:.2f}")
            
            for i, point in enumerate(resistance_points[:3]):  # 只顯示前三次測試
                print(f"第{i+1}次測試: {point['date'].strftime('%Y-%m-%d')} - ${point['price']:.2f}")
                if i < len(pullbacks):
                    print(f"  回調低點: {pullbacks[i]['date'].strftime('%Y-%m-%d')} - ${pullbacks[i]['price']:.2f}")
            
            return {
                'resistance_level': resistance_level,
                'resistance_points': resistance_points[:3],
                'pullbacks': pullbacks
            }
    
    print("未找到符合條件的模式")
    return None

# 使用範例
if __name__ == "__main__":
    # 在程式開始時呼叫
    set_chinese_font()
    # 設定股票代號
    sno = "1810.HK"  # 更改為你想要分析的股票
    stype = "L"
    days = 243
    lookback_days=30
    resistance_threshold=0.02
    
    print("=== 三次測試高位模式分析 ===")
    
    # 使用簡化版本（更穩定）
    #result = simple_resistance_pattern(sno,period=period,lookback_days=lookback_days,resistance_threshold=resistance_threshold)
    result = None


    # 如果簡化版本找不到，嘗試完整版本
    if result is None:
        print("\n嘗試完整版本分析...")        
        result = analyze_resistance_patterns(sno, stype, days ,resistance_threshold)        
