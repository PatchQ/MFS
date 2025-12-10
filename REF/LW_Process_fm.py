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


def calHHLL(df, window, min_swing_change, merge_threshold):
        
    stock = df.copy()
    stock.index = pd.to_datetime(stock.index,utc=True).tz_convert('Asia/Shanghai') 
    stock = extendData(stock)
   

    swing_highs, swing_lows = find_swing_points(stock['High'], stock['Low'], stock["Close"], window, min_swing_change)
    swing_analysis = classify_all_swing_points(swing_highs, swing_lows)

    final_swings = merge_near_hl_lh(swing_analysis, merge_threshold)

    return final_swings

def find_swing_points(high_series, low_series, close_series, window=10, min_swing_change=0.02):
    """
    找出摆动高点和摆动低点，忽略小回调和噪音
    
    参数:
    high_series: 最高价序列
    low_series: 最低价序列
    close_series: 收盘价序列
    window: 用于识别摆动点的窗口大小
    min_swing_change: 最小摆动变化百分比，用于过滤小回调
    
    返回:
    swing_highs: DataFrame 包含摆动高点的日期、最高价和对应收盘价
    swing_lows: DataFrame 包含摆动低点的日期、最低价和对应收盘价
    """
    # 使用滚动窗口找到局部高点和低点
    highs = high_series.rolling(window=window, center=True).max()
    lows = low_series.rolling(window=window, center=True).min()
    
    # 找出摆动高点 (当前高点等于滚动窗口内的最大值)
    swing_high_mask = high_series == highs
    swing_high_dates = high_series[swing_high_mask].index
    swing_high_prices = high_series[swing_high_mask].values
    
    # 找出摆动低点 (当前低点等于滚动窗口内的最小值)
    swing_low_mask = low_series == lows
    swing_low_dates = low_series[swing_low_mask].index
    swing_low_prices = low_series[swing_low_mask].values
    
    # 获取对应的收盘价
    swing_high_closes = []
    for date in swing_high_dates:
        if date in close_series.index:
            swing_high_closes.append(close_series[date])
        else:
            swing_high_closes.append(None)
    
    swing_low_closes = []
    for date in swing_low_dates:
        if date in close_series.index:
            swing_low_closes.append(close_series[date])
        else:
            swing_low_closes.append(None)
    
    # 创建包含所有信息的DataFrame
    swing_highs = pd.DataFrame({
        'date': swing_high_dates,
        'high': swing_high_prices,
        'close': swing_high_closes
    })
    
    swing_lows = pd.DataFrame({
        'date': swing_low_dates,
        'low': swing_low_prices,
        'close': swing_low_closes
    })
    
    # 过滤掉太接近的摆动点
    swing_highs = filter_close_points_df(swing_highs, window, price_col='high')
    swing_lows = filter_close_points_df(swing_lows, window, price_col='low')
    
    # 进一步过滤小回调
    swing_highs = filter_minor_swings(swing_highs, min_swing_change, is_high=True)
    swing_lows = filter_minor_swings(swing_lows, min_swing_change, is_high=False)
    
    return swing_highs, swing_lows

def filter_minor_swings(swing_df, min_change, is_high=True):
    """
    过滤掉小回调，保留主要摆动点，并将接近的HL或LH合并为LH
    
    参数:
    swing_df: 摆动点DataFrame
    min_change: 最小变化百分比
    is_high: 是否是高点（True为高点，False为低点）
    """
    if len(swing_df) < 3:
        return swing_df
    
    # 按日期排序
    swing_df = swing_df.sort_values('date').reset_index(drop=True)
    
    # 计算价格变化
    price_col = 'high' if is_high else 'low'
    prices = swing_df[price_col].values
    
    # 找出主要趋势变化点
    significant_swings = []
    
    # 总是保留第一个点
    significant_swings.append(0)
    
    for i in range(1, len(prices)-1):
        prev_idx = significant_swings[-1]
        prev_price = prices[prev_idx]
        current_price = prices[i]
        next_price = prices[i+1]
        
        # 计算价格变化百分比
        if is_high:
            # 对于高点，检查是否显著高于前一个主要高点
            change_from_prev = (current_price - prev_price) / prev_price
            # 检查是否显著高于下一个点
            change_to_next = (current_price - next_price) / current_price
            
            # 只有当价格显著高于前一个高点，并且显著高于下一个点时，才认为是主要高点
            if change_from_prev >= min_change and change_to_next >= min_change:
                significant_swings.append(i)
            # 如果变化不大，但当前点接近下一个点，考虑合并
            elif (change_from_prev >= min_change * 0.5 and 
                  change_to_next < min_change * 0.3):
                # 这种情况可能是LH，保留它
                significant_swings.append(i)
        else:
            # 对于低点，检查是否显著低于前一个主要低点
            change_from_prev = (prev_price - current_price) / prev_price
            # 检查是否显著低于下一个点
            change_to_next = (next_price - current_price) / current_price
            
            # 只有当价格显著低于前一个低点，并且显著低于下一个点时，才认为是主要低点
            if change_from_prev >= min_change and change_to_next >= min_change:
                significant_swings.append(i)
            # 如果变化不大，但当前点接近下一个点，考虑合并
            elif (change_from_prev >= min_change * 0.5 and 
                  change_to_next < min_change * 0.3):
                # 这种情况可能是HL，保留它
                significant_swings.append(i)
    
    # 总是保留最后一个点
    if len(significant_swings) == 0 or significant_swings[-1] != len(prices)-1:
        significant_swings.append(len(prices)-1)
    
    return swing_df.iloc[significant_swings].reset_index(drop=True)

# 新增函数：专门处理HL和LH的合并
def merge_near_hl_lh(swing_analysis, merge_threshold=0.01):
    """
    将接近的HL或LH合并为LH
    
    参数:
    swing_analysis: 已分类的摆动点DataFrame
    merge_threshold: 合并阈值（价格变化百分比）
    
    返回:
    合并后的摆动点DataFrame
    """
    if len(swing_analysis) < 2:
        return swing_analysis
    
    # 按日期排序
    swing_analysis = swing_analysis.sort_values('date').reset_index(drop=True)
    
    merged_swings = []
    
    i = 0
    while i < len(swing_analysis):
        current_swing = swing_analysis.iloc[i]
        
        # 如果是最后一个点，直接添加
        if i == len(swing_analysis) - 1:
            merged_swings.append(current_swing)
            break
        
        next_swing = swing_analysis.iloc[i+1]
        
        # 检查是否需要合并
        should_merge = False
        
        # 情况1: HL后面跟着LH，且价格接近
        if (current_swing['classification'] == 'HL' and 
            next_swing['classification'] == 'LH'):
            price_diff = abs(current_swing['price'] - next_swing['price']) / current_swing['price']
            if price_diff <= merge_threshold:
                should_merge = True
        
        # 情况2: LH后面跟着HL，且价格接近
        elif (current_swing['classification'] == 'LH' and 
              next_swing['classification'] == 'HL'):
            price_diff = abs(current_swing['price'] - next_swing['price']) / current_swing['price']
            if price_diff <= merge_threshold:
                should_merge = True
        
        # 情况3: 两个连续的LH，且价格接近
        elif (current_swing['classification'] == 'LH' and 
              next_swing['classification'] == 'LH'):
            price_diff = abs(current_swing['price'] - next_swing['price']) / current_swing['price']
            if price_diff <= merge_threshold:
                should_merge = True
        
        # 情况4: 两个连续的HL，且价格接近
        elif (current_swing['classification'] == 'HL' and 
              next_swing['classification'] == 'HL'):
            price_diff = abs(current_swing['price'] - next_swing['price']) / current_swing['price']
            if price_diff <= merge_threshold:
                should_merge = True
        
        if should_merge:
            # 合并为LH（选择价格较高的点作为LH）
            merged_price = max(current_swing['price'], next_swing['price'])
            merged_date = next_swing['date']  # 使用较晚的日期
            
            # 创建合并后的摆动点
            merged_swing = current_swing.copy()
            merged_swing['date'] = merged_date
            merged_swing['price'] = merged_price
            merged_swing['classification'] = 'LH'
            
            merged_swings.append(merged_swing)
            i += 2  # 跳过下一个点
        else:
            # 不需要合并，直接添加当前点
            merged_swings.append(current_swing)
            i += 1
    
    # 转换为DataFrame
    merged_df = pd.DataFrame(merged_swings)
    return merged_df


def filter_close_points_df(df, min_distance, price_col='price'):
    """
    过滤掉距离太近的摆动点 - DataFrame版本
    """
    if len(df) == 0:
        return df
    
    # 按日期排序
    df = df.sort_values('date').reset_index(drop=True)
    
    filtered_df = pd.DataFrame(columns=df.columns)
    last_date = None
    
    for i, row in df.iterrows():
        current_date = row['date']
        
        if last_date is None:
            filtered_df = pd.concat([filtered_df, pd.DataFrame([row])], ignore_index=True)
            last_date = current_date
        else:
            # 计算与前一个点的距离 (天数)
            days_diff = (current_date - last_date).days
            if days_diff >= min_distance:
                filtered_df = pd.concat([filtered_df, pd.DataFrame([row])], ignore_index=True)
                last_date = current_date
    
    return filtered_df


def classify_all_swing_points(swing_highs_df, swing_lows_df, tolerance=0.001):
    """
    分类所有摆动点为 HH, HL, LH, LL, -H, -L - 高级版本
    
    参数:
    swing_highs_df: 摆动高点DataFrame (来自find_swing_points_advanced)
    swing_lows_df: 摆动低点DataFrame (来自find_swing_points_advanced)
    tolerance: 价格相等的容忍度 (0.1%)
    
    返回:
    swing_analysis: 包含所有摆动点及其分类的DataFrame
    """
    # 合并所有摆动点并标记类型
    highs_df = swing_highs_df.copy()
    highs_df['type'] = 'high'
    highs_df.rename(columns={'high': 'price'}, inplace=True)
    
    lows_df = swing_lows_df.copy()
    lows_df['type'] = 'low'
    lows_df.rename(columns={'low': 'price'}, inplace=True)
    
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


def AnalyzeData(sno,stype):
       
    df = pd.read_csv(PATH+"/"+stype+"/"+sno+".csv",index_col=0)    
    df = convertData(df)
    
    #df = calEMA(df)

    window = 10
    min_swing_change = 0.02
    merge_threshold = 0.015

    tempdf = calHHLL(df, window, min_swing_change, merge_threshold)    
    df = checkLHHHLL(df, sno, stype, tempdf)

    #df = calT1(df,22)
    #df = calT1(df,50)
        
    df = df.reset_index()
    df.to_csv(OUTPATH+"/"+stype+"/P_"+sno+".csv",index=False)


def YFprocessData(stype):

    snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(PATH+"/"+stype)))
    SLIST = pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST[7:8]

    with cf.ProcessPoolExecutor(max_workers=5) as executor:
        list(tqdm(executor.map(AnalyzeData,SLIST["sno"],SLIST["stype"],chunksize=1),total=len(SLIST)))


if __name__ == '__main__':
    start = t.perf_counter()

    YFprocessData("L")
    #YFprocessData("M")
    #YFprocessData("S")

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    