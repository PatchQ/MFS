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


def calHHLL(df, window, trend_window, min_swing_change):
        
    stock = df.copy()
    stock.index = pd.to_datetime(stock.index,utc=True).tz_convert('Asia/Shanghai') 
    stock = extendData(stock)

    swing_highs, swing_lows = find_swing_points_advanced(stock['High'], stock['Low'], stock["Close"], window, trend_window, min_swing_change)
    swing_analysis = classify_all_swing_points(swing_highs, swing_lows)    
        
    return swing_analysis

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

def find_swing_points_advanced(high_series, low_series, close_series, window, trend_window, min_swing_change):
    """
    高级版本：使用趋势检测来识别主要摆动点
    
    参数:
    high_series: 最高价序列
    low_series: 最低价序列
    close_series: 收盘价序列
    window: 用于识别摆动点的窗口大小
    trend_window: 用于检测趋势的窗口大小
    min_trend_change: 最小趋势变化百分比
    """
    # 计算趋势
    trend_highs = high_series.rolling(window=trend_window).max()
    trend_lows = low_series.rolling(window=trend_window).min()
    
    # 找出摆动高点
    swing_high_mask = (high_series == high_series.rolling(window=window, center=True).max())
    swing_high_dates = high_series[swing_high_mask].index
    swing_high_prices = high_series[swing_high_mask].values
    
    # 找出摆动低点
    swing_low_mask = (low_series == low_series.rolling(window=window, center=True).min())
    swing_low_dates = low_series[swing_low_mask].index
    swing_low_prices = low_series[swing_low_mask].values
    
    # 获取对应的收盘价
    swing_high_closes = [close_series[date] if date in close_series.index else None for date in swing_high_dates]
    swing_low_closes = [close_series[date] if date in close_series.index else None for date in swing_low_dates]
    
    # 创建DataFrame
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
    
    # 过滤：只保留趋势转折点
    swing_highs = filter_trend_swings(swing_highs, trend_highs, is_high=True)
    swing_lows = filter_trend_swings(swing_lows, trend_lows, is_high=False)

    # 过滤掉太接近的摆动点
    swing_highs = filter_close_points_df(swing_highs, window, price_col='high')
    swing_lows = filter_close_points_df(swing_lows, window, price_col='low')

    # # 轻微过滤小回调，但不移除任何类型的摆动点
    # swing_highs = filter_minor_swings_light(swing_highs, min_swing_change, is_high=True)
    # swing_lows = filter_minor_swings_light(swing_lows, min_swing_change, is_high=False)
    
    return swing_highs, swing_lows

def filter_trend_swings(swing_df, trend_series, is_high=True):
    """
    过滤摆动点，只保留趋势转折点
    """
    if len(swing_df) == 0:
        return swing_df
    
    # 按日期排序
    swing_df = swing_df.sort_values('date').reset_index(drop=True)
    
    price_col = 'high' if is_high else 'low'
    filtered_swings = []
    
    for i, row in swing_df.iterrows():
        date = row['date']
        price = row[price_col]
        
        if date not in trend_series.index:
            continue
            
        trend_value = trend_series[date]
        
        # 检查是否是趋势转折点
        if is_high:
            # 对于高点，检查是否接近趋势高点
            if abs(price - trend_value) / trend_value <= 0.01:  # 1% 容差
                filtered_swings.append(i)
        else:
            # 对于低点，检查是否接近趋势低点
            if abs(price - trend_value) / trend_value <= 0.01:  # 1% 容差
                filtered_swings.append(i)
    
    return swing_df.iloc[filtered_swings].reset_index(drop=True)

def filter_minor_swings_light(swing_df, min_change, is_high=True):
    """
    轻微过滤小回调，但不移除任何类型的摆动点
    
    参数:
    swing_df: 摆动点DataFrame
    min_change: 最小变化百分比
    is_high: 是否是高点（True为高点，False为低点）
    """
    if len(swing_df) < 2:
        return swing_df
    
    # 按日期排序
    swing_df = swing_df.sort_values('date').reset_index(drop=True)
    
    # 轻微过滤：只移除极其接近的连续点
    price_col = 'high' if is_high else 'low'
    prices = swing_df[price_col].values
    
    # 保留所有点，但标记质量
    swing_df['quality'] = 'good'
    
    for i in range(1, len(prices)):
        prev_price = prices[i-1]
        current_price = prices[i]
        
        # 计算价格变化百分比
        price_diff_pct = abs((current_price - prev_price) / prev_price)
        
        # 如果变化非常小，标记为低质量，但不移除
        if price_diff_pct < min_change * 0.5:  # 使用更宽松的阈值
            swing_df.loc[i, 'quality'] = 'minor'
    
    return swing_df

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


def checkLHHHLL(df, sno, stype, swing_analysis):

    #print(sno)

    df.index = pd.to_datetime(df.index)

    swing_analysis = swing_analysis.reset_index()
    swing_analysis['date'] = swing_analysis['date'].dt.strftime("%Y-%m-%d")

    swing_analysis['PATTERN'] = ""
    swing_analysis['LLLow'] = 0
    swing_analysis['LLDate'] = ""
    swing_analysis['HHClose'] = 0
    swing_analysis['HHDate'] = ""
    swing_analysis['HHHigh'] = 0
    swing_analysis['sno'] = sno
    swing_analysis['stype'] = stype
    
    df['classification'] = ""
    df['PATTERN'] = ""
    df['LLLow'] = 0
    df['LLDate'] = ""
    df['HHClose'] = 0
    df['HHDate'] = ""
    df['HHHigh'] = 0
    df['22DLow'] = 0
    df['BOSS_STATUS'] = ""
    df['BOSSB'] = False
    df['BOSSTP1'] = False
    df['BOSSTP2'] = False
    df['BOSSCL1'] = False
    df['BOSSCL2'] = False
    

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(swing_analysis) - 2):
            templist = list(swing_analysis['classification'].iloc[i:i+3])
            swing_analysis['PATTERN'].iloc[i] = ''.join(templist)

            swing_analysis['LLLow'].iloc[i] = swing_analysis['price'].iloc[i+1]
            swing_analysis['LLDate'].iloc[i] = swing_analysis['date'].iloc[i+1]
            swing_analysis['HHClose'].iloc[i] = swing_analysis['close'].iloc[i+2]            
            swing_analysis['HHDate'].iloc[i] = swing_analysis['date'].iloc[i+2]
            swing_analysis['HHHigh'].iloc[i] = swing_analysis['price'].iloc[i+2]

            sadate = pd.to_datetime(swing_analysis['date'].iloc[i])

            date_match = (df.index == sadate)
            df.loc[date_match, "classification"] = swing_analysis["classification"].iloc[i]
            df.loc[date_match, "LLLow"] = swing_analysis["LLLow"].iloc[i]
            df.loc[date_match, "LLDate"] = swing_analysis["LLDate"].iloc[i]
            df.loc[date_match, "HHClose"] = swing_analysis["HHClose"].iloc[i]
            df.loc[date_match, "HHDate"] = swing_analysis["HHDate"].iloc[i]
            df.loc[date_match, "HHHigh"] = swing_analysis["HHHigh"].iloc[i]
            df.loc[date_match, "PATTERN"] = swing_analysis["PATTERN"].iloc[i]

            etempdate = pd.to_datetime(swing_analysis["LLDate"].iloc[i])
            stempdate = etempdate - timedelta(days=22)

            df.loc[date_match, "22DLow"] = df.loc[(df.index>=stempdate) & (df.index<etempdate), "Low"].min()
            

    swing_analysis["BOSS1"] = ((swing_analysis['PATTERN']=="LHLLHH") & (swing_analysis['HHClose']>swing_analysis['price']))
    #swing_analysis.to_csv(OUTPATH+"/HHLL/HL_"+sno+".csv",index=False)

    BOSSRule1 = df['PATTERN']=="LHLLHH"
    BOSSRule2 = df['HHClose']>df['High']        
    BOSSRule3 = df['LLLow']<df['22DLow'] 

    df["BOSS1"] = (BOSSRule1 & BOSSRule2 & BOSSRule3)    
    
    tempdf = df.loc[df["BOSS1"]]
    tempdf = tempdf.reset_index()

    df['bullish_ratio'] = 0.00
    
    for i in range(len(tempdf)):
        sdate = pd.to_datetime(tempdf["LLDate"].iloc[i])
        edate = pd.to_datetime(tempdf["HHDate"].iloc[i])
        fdf = df.loc[(df.index>sdate) & (df.index<=edate)]

        bullish_count, bullish_ratio = calCandleStick(fdf)
        strong_bullish, medium_bullish, weak_bullish = calCandleStickBody(fdf)        

        date_match = (df.index == tempdf["Date"].iloc[i])
        df.loc[date_match, "bullish_count"] = bullish_count
        df.loc[date_match, "bullish_ratio"] = bullish_ratio
        df.loc[date_match, "strong_bullish"] = strong_bullish
        df.loc[date_match, "medium_bullish"] = medium_bullish
        df.loc[date_match, "weak_bullish"] = weak_bullish        
    
    df["BOSS2"] = (df["BOSS1"] & (df["bullish_ratio"]>=0.6))  
    
    df.loc[df["BOSS2"], "buy_price"] = round(((df["HHHigh"] + df["LLLow"]) / 2),2)
    df.loc[df["BOSS2"], "tp1_price"] = df["HHHigh"]
    df.loc[df["BOSS2"], "tp2_price"] = df["buy_price"] + (df["HHHigh"] - df["buy_price"]) * 2 
    df.loc[df["BOSS2"], "BOSS_STATUS"] = "SB1-"+df.loc[df["BOSS2"]].index.strftime("%Y/%m/%d")

    tempdf = df.loc[df["BOSS2"]]    
    tempdf = tempdf.reset_index()

    for i in range(len(tempdf)):

        lastbuydate = pd.to_datetime("1900-01-01") 
        lastcl1date = pd.to_datetime("1900-01-01") 
        lasttp1date = pd.to_datetime("1900-01-01") 
        lastcl2date = pd.to_datetime("1900-01-01") 
        lasttp2date = pd.to_datetime("1900-01-01") 
        tp1 = False
        tp2 = False
        cl1 = False
        cl2 = False
        buy = False        

        diffdate = timedelta(days=100)
        hhdate = pd.to_datetime(tempdf["HHDate"].iloc[i])
        startbossdate = tempdf['Date'].iloc[i].strftime("%Y/%m/%d")        

        # if (i==len(tempdf)-1):            
        #     nextbossdate = (datetime.now() + timedelta(days=180))            
        # else:      
        #     nextbossdate = pd.to_datetime(tempdf['Date'].iloc[i+1])

        buy_price = tempdf["buy_price"].iloc[i]
        cl_price = tempdf["LLLow"].iloc[i]
        tp1_price = tempdf["tp1_price"].iloc[i]
        tp2_price = tempdf["tp2_price"].iloc[i]

        buydate_mask = (df.index < hhdate+diffdate) & (df.index > hhdate) & (buy_price>=df["Low"]) #& df["EMA2"]
        buydates = df[buydate_mask].index

        if len(buydates)!=0:
            buy = True            
            lastbuydate = buydates[0]

        if buy:
            df.loc[lastbuydate,'BOSS_STATUS'] = "BY1-"+startbossdate
            df.loc[lastbuydate,'BOSSB'] = True
            #print("BUYDate : "+lastbuydate.strftime("%Y-%m-%d"))

            #tp1date_mask = (df.index < lastbuydate+diffdate) & (df.index >= lastbuydate) & (df["High"]>=tp1_price*0.995)
            tp1date_mask = (df.index >= lastbuydate) & (df["High"]>=tp1_price*0.995)
            tp1dates = df[tp1date_mask].index

            #cl1date_mask = (df.index < lastbuydate+diffdate) & (df.index >= lastbuydate) & (cl_price>=df["Low"])
            cl1date_mask = (df.index >= lastbuydate) & (df["Close"]<cl_price)
            cl1dates = df[cl1date_mask].index            

            if len(tp1dates)!=0:
                tp1=True
                lasttp1date = tp1dates[0]

            if len(cl1dates)!=0:                
                cl1=True
                lastcl1date = cl1dates[0]                

            if cl1:
                if tp1:
                    if (lasttp1date>=lastcl1date):
                        df.loc[lastcl1date,'BOSS_STATUS'] = "CL1-"+startbossdate
                        df.loc[lastcl1date,'BOSSCL1'] = True                  
                        tp1=False      
                        #print("CL1 : "+lastcl1date.strftime("%Y-%m-%d"))
                    else:
                        df.loc[lasttp1date,'BOSS_STATUS'] = "TP1-"+startbossdate
                        df.loc[lasttp1date,'BOSSTP1'] = True
                        tp1=True
                        #print("TP1 : "+lasttp1date.strftime("%Y-%m-%d"))                             
                else:
                    df.loc[lastcl1date,'BOSS_STATUS'] = "CL1-"+startbossdate
                    df.loc[lastcl1date,'BOSSCL1'] = True
                    #print("CL1 : "+lastcl1date.strftime("%Y-%m-%d"))
            
            if tp1:
                df.loc[lasttp1date,'BOSS_STATUS'] = "TP1-"+startbossdate
                df.loc[lasttp1date,'BOSSTP1'] = True
                #print("TP1 : "+lasttp1date.strftime("%Y-%m-%d"))     

                #tp2date_mask = (df.index < lasttp1date+diffdate) & (df.index >= lasttp1date) & (df["High"]>=tp2_price*0.99)
                tp2date_mask = (df.index >= lasttp1date) & (df["High"]>=tp2_price*0.99)
                tp2dates = df[tp2date_mask].index        
            
                #cl2date_mask = (df.index < lasttp1date+diffdate) & (df.index >= lasttp1date) & (cl_price>=df["Low"])
                cl2date_mask = (df.index >= lasttp1date) & (df["Close"]<cl_price) 
                cl2dates = df[cl2date_mask].index

                if len(tp2dates)!=0:
                    tp2 = True                
                    lasttp2date = tp2dates[0]

                if len(cl2dates)!=0:
                    cl2 = True
                    lastcl2date = cl2dates[0]

                if cl2:
                    if tp2:
                        if (lasttp2date>=lastcl2date):
                            df.loc[lastcl2date,'BOSS_STATUS'] = "CL2-"+startbossdate
                            df.loc[lastcl2date,'BOSSCL2'] = True
                            tp2=False
                            #print("CL2 : "+lastcl2date.strftime("%Y-%m-%d")+ " : "+startbossdate)
                        else:
                            df.loc[lasttp2date,'BOSS_STATUS'] = "TP2-"+startbossdate
                            df.loc[lasttp2date,'BOSSTP2'] = True
                            tp2=True
                            #print("TP2 : "+lasttp2date.strftime("%Y-%m-%d")+ " : "+startbossdate)
                    else:
                        df.loc[lastcl2date,'BOSS_STATUS'] = "CL2-"+startbossdate
                        df.loc[lastcl2date,'BOSSCL2'] = True
                        #print("CL2 : "+lastcl2date.strftime("%Y-%m-%d")+ " : "+startbossdate)               
                else:
                    if tp2:
                        df.loc[lasttp2date,'BOSS_STATUS'] = "TP2-"+startbossdate
                        df.loc[lasttp2date,'BOSSTP2'] = True
                        #print("TP2 : "+lasttp2date.strftime("%Y-%m-%d")+ " : "+startbossdate)

    return df



def AnalyzeData(sno,stype):
       
    df = pd.read_csv(PATH+"/"+stype+"/"+sno+".csv",index_col=0)    
    df = convertData(df)
    
    #df = calEMA(df)

    window = 10
    trend_window = 6
    min_swing_change = 0.03

    tempdf = calHHLL(df, window, trend_window, min_swing_change)    

    tempdf.to_csv()
    tempdf.to_csv("Data/WT_0011.HK.csv",index=False)

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
    