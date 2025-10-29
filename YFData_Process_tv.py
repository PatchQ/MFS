import pandas as pd
import numpy as np
import yfinance as yf
import concurrent.futures as cf
import os
from tqdm import tqdm
import time as t
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import warnings

from YFData_Calindicator import *



PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/" 
DAYS=0
TOLERANCE=0.001
#WINDOW=10

def pivot_high(series, left_bars, right_bars):
    """
    识别枢轴高点
    """
    highs = []
    for i in range(left_bars, len(series) - right_bars):
        left_window = series[i-left_bars:i]
        right_window = series[i+1:i+right_bars+1]
        current = series.iloc[i]
        
        if (current > max(left_window) and 
            current > max(right_window)):
            highs.append((i, current))
    
    return highs

def pivot_low(series, left_bars, right_bars):
    """
    识别枢轴低点
    """
    lows = []
    for i in range(left_bars, len(series) - right_bars):
        left_window = series[i-left_bars:i]
        right_window = series[i+1:i+right_bars+1]
        current = series.iloc[i]
        
        if (current < min(left_window) and 
            current < min(right_window)):
            lows.append((i, current))
    
    return lows

def find_previous_points(current_index, pivot_points, current_type):
    """
    找到前四个相关点 (b, c, d, e)
    """
    # 过滤出当前索引之前的点
    previous_points = [p for p in pivot_points if p[0] < current_index]
    
    if len(previous_points) < 4:
        return None, None, None, None
    
    # 根据当前类型找到前四个点
    ehl = -1 if current_type == 1 else 1  # 相反类型
    
    # 找到第一个点 (b)
    b_point = None
    for i in range(len(previous_points)-1, -1, -1):
        if previous_points[i][2] == ehl:
            b_point = previous_points[i]
            break
    
    if b_point is None:
        return None, None, None, None
    
    # 找到第二个点 (c) - 与当前点同类型
    c_point = None
    for i in range(len(previous_points)-1, -1, -1):
        if (previous_points[i][2] == current_type and 
            previous_points[i][0] < b_point[0]):
            c_point = previous_points[i]
            break
    
    if c_point is None:
        return None, None, None, None
    
    # 找到第三个点 (d) - 与第一个点同类型
    d_point = None
    for i in range(len(previous_points)-1, -1, -1):
        if (previous_points[i][2] == ehl and 
            previous_points[i][0] < c_point[0]):
            d_point = previous_points[i]
            break
    
    if d_point is None:
        return b_point, c_point, None, None
    
    # 找到第四个点 (e) - 与当前点同类型
    e_point = None
    for i in range(len(previous_points)-1, -1, -1):
        if (previous_points[i][2] == current_type and 
            previous_points[i][0] < d_point[0]):
            e_point = previous_points[i]
            break
    
    return b_point, c_point, d_point, e_point

def identify_hh_ll_patterns(high_series, low_series, close_series, left_bars, right_bars):
    """
    识别HH, LL, HL, LH模式
    
    参数:
    high_series: 最高价序列
    low_series: 最低价序列
    close_series: 收盘价序列
    left_bars: 左侧柱数
    right_bars: 右侧柱数
    
    返回:
    patterns_df: 包含所有识别模式的DataFrame
    """
    # 识别枢轴高点和低点
    pivot_highs = pivot_high(high_series, left_bars, right_bars)
    pivot_lows = pivot_low(low_series, left_bars, right_bars)
    
    # 合并所有枢轴点并按索引排序
    all_pivots = []
    for idx, price in pivot_highs:
        all_pivots.append((idx, price, 1))  # 1 表示高点
    for idx, price in pivot_lows:
        all_pivots.append((idx, price, -1))  # -1 表示低点
    
    # 按索引排序
    all_pivots.sort(key=lambda x: x[0])
    
    # 过滤枢轴点 - 类似TradingView的过滤逻辑
    filtered_pivots = []
    for i, (idx, price, p_type) in enumerate(all_pivots):
        if i == 0:
            filtered_pivots.append((idx, price, p_type))
            continue
        
        prev_idx, prev_price, prev_type = filtered_pivots[-1]
        
        # 过滤条件1: 连续同类型点
        if (p_type == -1 and prev_type == -1 and 
            price > prev_price):
            continue  # 跳过这个点
        
        if (p_type == 1 and prev_type == 1 and 
            price < prev_price):
            continue  # 跳过这个点
        
        # 过滤条件2: 类型转换时的价格关系
        if (p_type == -1 and prev_type == 1 and 
            price > prev_price):
            continue  # 跳过这个点
        
        if (p_type == 1 and prev_type == -1 and 
            price < prev_price):
            continue  # 跳过这个点
        
        filtered_pivots.append((idx, price, p_type))
    
    # 识别模式
    patterns = []
    for i, (idx, price, p_type) in enumerate(filtered_pivots):
        if i < 4:  # 需要至少4个点才能判断模式
            continue
        
        # 找到前四个点
        b_point, c_point, d_point, e_point = find_previous_points(
            idx, filtered_pivots[:i], p_type
        )
        
        if None in [b_point, c_point, d_point, e_point]:
            continue
        
        b_idx, b_price, b_type = b_point
        c_idx, c_price, c_type = c_point
        d_idx, d_price, d_type = d_point
        e_idx, e_price, e_type = e_point
        
        # 判断模式
        is_hh = False
        is_ll = False
        is_hl = False
        is_lh = False
        
        if p_type == 1:  # 当前是高点
            # HH条件: a > b and a > c and c > b and c > d
            if (price > b_price and price > c_price and 
                c_price > b_price and c_price > d_price):
                is_hh = True
            
            # LH条件1: a <= c and (b < c and b < d and d < c and d < e)
            elif (price <= c_price and 
                  b_price < c_price and b_price < d_price and 
                  d_price < c_price and d_price < e_price):
                is_lh = True
            
            # LH条件2: a > b and a < c and b > d
            elif (price > b_price and price < c_price and 
                  b_price > d_price):
                is_lh = True
        
        else:  # 当前是低点
            # LL条件: a < b and a < c and c < b and c < d
            if (price < b_price and price < c_price and 
                c_price < b_price and c_price < d_price):
                is_ll = True
            
            # HL条件1: a >= c and (b > c and b > d and d > c and d > e)
            elif (price >= c_price and 
                  b_price > c_price and b_price > d_price and 
                  d_price > c_price and d_price > e_price):
                is_hl = True
            
            # HL条件2: a < b and a > c and b < d
            elif (price < b_price and price > c_price and 
                  b_price < d_price):
                is_hl = True
        
        # 记录模式
        pattern_type = None
        if is_hh:
            pattern_type = 'HH'
        elif is_ll:
            pattern_type = 'LL'
        elif is_hl:
            pattern_type = 'HL'
        elif is_lh:
            pattern_type = 'LH'
        
        if pattern_type:
            patterns.append({
                'date': high_series.index[idx],
                #'index': idx,
                'price': price,
                'type': 'high' if p_type == 1 else 'low',
                'classification': pattern_type,
                'close': close_series.iloc[idx] if idx < len(close_series) else None
            })
    
    # 转换为DataFrame
    if patterns:        
        patterns_df = pd.DataFrame(patterns)
    else:
        patterns.append({
                'date': datetime.now(),
                #'index': idx,
                'price': 0,
                'type': '',
                'classification': '',
                'close': 0
            })
        patterns_df = pd.DataFrame(patterns)
        
    return patterns_df

def calHHLL(df, left_bars, right_bars):
    # 下载数据
    stock_data = df.copy()
    stock_data.index = pd.to_datetime(stock_data.index,utc=True).tz_convert('Asia/Shanghai') 
    stock_data = extendData(stock_data)

    # 识别模式
    patterns_df = identify_hh_ll_patterns(
        stock_data['High'],
        stock_data['Low'],
        stock_data['Close'],
        left_bars,
        right_bars
    )

    return patterns_df

def AnalyzeData(sno,stype):
       
    df = pd.read_csv(PATH+"/"+stype+"/"+sno+".csv",index_col=0)    
    df = convertData(df)
    
    #df = calEMA(df)


    tempdf = calHHLL(df, left_bars=3, right_bars=3)
    df = checkLHHHLL(df, sno, stype, tempdf)

    #df = calT1(df,22)
    #df = calT1(df,50)
        
    df = df.reset_index()
    df.to_csv(OUTPATH+"/"+stype+"/P_"+sno+".csv",index=False)


def YFprocessData(stype):

    snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(PATH+"/"+stype)))
    SLIST = pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST[:]

    with cf.ProcessPoolExecutor(max_workers=5) as executor:
        list(tqdm(executor.map(AnalyzeData,SLIST["sno"],SLIST["stype"],chunksize=1),total=len(SLIST)))


if __name__ == '__main__':
    start = t.perf_counter()

    #YFprocessData("L")
    YFprocessData("M")
    #YFprocessData("S")


    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    