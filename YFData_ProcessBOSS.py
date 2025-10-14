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
import warnings


PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/" 

#PERIOD="1y"
DAYS=300
TOLERANCE=0.001
WINDOW=10


def convertData(df):

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=numeric_cols, how='all', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def find_swing_points(high_series, low_series, close_series):
    """
    找出摆动高点和摆动低点
    """
    # 使用滚动窗口找到局部高点和低点
    highs = high_series.rolling(window=WINDOW, center=True).max()
    lows = low_series.rolling(window=WINDOW, center=True).min()  
    
    # 找出摆动高点 (当前高点等于滚动窗口内的最大值)
    swing_high_mask = high_series == highs
    swing_high_dates = high_series[swing_high_mask].index
    swing_high_prices = high_series[swing_high_mask].values    

    # 找出摆动低点 (当前低点等于滚动窗口内的最小值)
    swing_low_mask = low_series == lows
    swing_low_dates = low_series[swing_low_mask].index
    swing_low_prices = low_series[swing_low_mask].values

    # 获取对应的收盘价并存储为额外属性
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
        'price': swing_high_prices,
        'close': swing_high_closes,
        'type': 'high'
    })
    
    swing_lows = pd.DataFrame({
        'date': swing_low_dates,
        'price': swing_low_prices,
        'close': swing_low_closes,
        'type': 'low'
    })
    
    # 过滤掉太接近的摆动点
    swing_highs = filter_close_points_df(swing_highs, WINDOW)
    swing_lows = filter_close_points_df(swing_lows, WINDOW)
    
    return swing_highs, swing_lows

def filter_close_points_df(df, min_distance):
    """
    过滤掉距离太近的摆动点 - DataFrame版本
    
    参数:
    df: 包含摆动点的DataFrame
    min_distance: 最小距离（天数）
    price_col: 价格列名
    
    返回:
    过滤后的DataFrame
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
            filtered_df = filtered_df.dropna(axis=1, how="all")
            filtered_df = pd.concat([filtered_df, pd.DataFrame([row])], ignore_index=True)
            last_date = current_date
        else:
            # 计算与前一个点的距离 (天数)
            days_diff = (current_date - last_date).days
            if days_diff >= min_distance:
                filtered_df = filtered_df.dropna(axis=1, how="all")
                filtered_df = pd.concat([filtered_df, pd.DataFrame([row])], ignore_index=True)
                last_date = current_date
    
    return filtered_df


def classify_all_swing_points(highs_df, lows_df):
    """
    分类所有摆动点为 HH, HL, LH, LL, -H, -L
    
    参数:
    tolerance: 价格相等的容忍度 (0.1%)
    
    返回:
    swing_analysis: 包含所有摆动点及其分类的DataFrame
    """   
    
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
        
        if price_diff_pct <= TOLERANCE:
            high_points.loc[i, 'classification'] = "-H"  # 相同高位
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
        
        if price_diff_pct <= TOLERANCE:
            low_points.loc[i, 'classification'] = "-L"  # 相同低位
        elif current_price > prev_price:
            low_points.loc[i, 'classification'] = 'HL'  # 更高低点
        else:
            low_points.loc[i, 'classification'] = 'LL'  # 更低低点
    
    # 合并分类结果
    classified_swings = pd.concat([high_points, low_points]).sort_values('date')
    
    return classified_swings

def calHHLL(sno, stype):
        
    nowprice = 0
    df = pd.read_csv(PATH+"/"+stype+"/"+sno+".csv",index_col=0)    
    df = convertData(df)
    stock = df.tail(DAYS)    

    # ticker = yf.Ticker(sno)
    # stock = ticker.history(period=PERIOD,auto_adjust=False)
    # stock = stock[stock['Volume'] > 0]
    stock.index = pd.to_datetime(stock.index,utc=True).tz_convert('Asia/Shanghai') 

    if len(stock)!=0:
        nowprice = stock['Close'].iloc[-1]
    
    # 找出摆动点
    swing_highs, swing_lows = find_swing_points(stock['High'], stock['Low'], stock["Close"])
    
    # 分类所有摆动点
    swing_analysis = classify_all_swing_points(swing_highs, swing_lows)

    swing_analysis['PATTERN'] = ""
    swing_analysis['HHClose'] = 0
    swing_analysis['HLLow'] = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(swing_analysis) - 2):
            templist = list(swing_analysis['classification'].iloc[i:i+3])            
            swing_analysis['HHClose'].iloc[i] = swing_analysis['close'].iloc[i+2]
            swing_analysis['HLLow'].iloc[i] = swing_analysis['price'].iloc[i+1]
            swing_analysis['PATTERN'].iloc[i] = ''.join(templist)

    swing_analysis["BOSS1"] = ((swing_analysis['PATTERN']=="LHLLHH") & (swing_analysis['HHClose']>swing_analysis['price']))
    swing_analysis["BOSS2"] = ((swing_analysis['PATTERN']=="HHHLHH") & (nowprice>swing_analysis['HLLow']))
    swing_analysis.insert(1,"sno", sno)
    swing_analysis.insert(2,"stype", stype)
    swing_analysis.to_csv(OUTPATH+"/HHLL/HL_"+sno+".csv",index=False)

    return swing_analysis



def YFProcessBOSS(stype):

    snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(PATH+"/"+stype)))
    SLIST = pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST[:]

    with cf.ProcessPoolExecutor(max_workers=5) as executor:
        list(tqdm(executor.map(calHHLL,SLIST["sno"],SLIST["stype"],chunksize=1),total=len(SLIST)))



if __name__ == '__main__':
    start = t.perf_counter()

    YFProcessBOSS("L")
    YFProcessBOSS("M")
    YFProcessBOSS("S")

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    
    