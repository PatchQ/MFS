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
    # df['bullish_count'] = 0
    # df['strong_bullish'] = 0
    # df['medium_bullish'] = 0
    # df['weak_bullish'] = 0   
    # df['buy_price'] = 0.00
    # df['tp1_price'] = 0.00
    # df['tp2_price'] = 0.00
    

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
    
    df["BOSS1"] = (df["BOSS1"] & (df["bullish_ratio"]>=0.6))  
    
    df.loc[df["BOSS1"], "buy_price"] = round(((df["HHHigh"] + df["LLLow"]) / 2),2)
    df.loc[df["BOSS1"], "tp1_price"] = df["HHHigh"]
    df.loc[df["BOSS1"], "tp2_price"] = df["buy_price"] + (df["HHHigh"] - df["buy_price"]) * 2 
    df.loc[df["BOSS1"], "BOSS_STATUS"] = "SB1-"+df.loc[df["BOSS1"]].index.strftime("%Y/%m/%d")

    tempdf = df.loc[df["BOSS1"]]    
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

        diffdate = timedelta(days=90)
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

            tp1date_mask = (df.index < lastbuydate+diffdate) & (df.index >= lastbuydate) & (df["High"]>=tp1_price)
            tp1dates = df[tp1date_mask].index

            cl1date_mask = (df.index < lastbuydate+diffdate) & (df.index >= lastbuydate) & (cl_price>=df["Low"])
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

                tp2date_mask = (df.index < lasttp1date+diffdate) & (df.index >= lasttp1date) & (df["High"]>=tp2_price*0.99)
                tp2dates = df[tp2date_mask].index        
            
                cl2date_mask = (df.index < lasttp1date+diffdate) & (df.index >= lasttp1date) & (cl_price>=df["Low"])
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
                            #print("CL2 : "+lastcl2date.strftime("%Y-%m-%d"))
                        else:
                            df.loc[lasttp2date,'BOSS_STATUS'] = "TP2-"+startbossdate
                            df.loc[lasttp2date,'BOSSTP2'] = True
                            tp2=True
                            #print("TP2 : "+lasttp2date.strftime("%Y-%m-%d"))
                    else:
                        df.loc[lastcl2date,'BOSS_STATUS'] = "CL2-"+startbossdate
                        df.loc[lastcl2date,'BOSSCL2'] = True
                        #print("CL2 : "+lastcl2date.strftime("%Y-%m-%d"))               
                
                if tp2:
                    df.loc[lasttp2date,'BOSS_STATUS'] = "TP2-"+startbossdate
                    df.loc[lasttp2date,'BOSSTP2'] = True
                    #print("TP2 : "+lasttp2date.strftime("%Y-%m-%d"))

    return df


def calHHLL(df):
        
    stock = df.copy()
    stock.index = pd.to_datetime(stock.index,utc=True).tz_convert('Asia/Shanghai') 
    stock = extendData(stock)
   
    # 找出摆动点
    # swing_highs1, swing_lows1 = find_swing_points(stock['High'], stock['Low'], stock["Close"], 5)
    # swing_analysis1 = classify_all_swing_points(swing_highs1, swing_lows1)

    swing_highs2, swing_lows2 = find_swing_points(stock['High'], stock['Low'], stock["Close"], 10)
    swing_analysis2 = classify_all_swing_points(swing_highs2, swing_lows2)
        
    # 分类所有摆动点
    # swing_analysis = pd.concat([swing_analysis1, swing_analysis2]).sort_values('date').drop_duplicates(subset=['date'], keep='last')        

    return swing_analysis2
   

def find_swing_points(high_series, low_series, close_series, window_days):
    """
    找出摆动高点和摆动低点
    """
    # 使用滚动窗口找到局部高点和低点
    highs = high_series.rolling(window=window_days, center=True).max()
    lows = low_series.rolling(window=window_days, center=True).min()  
    
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
    swing_highs = filter_close_points_df(swing_highs, window_days)
    swing_lows = filter_close_points_df(swing_lows, window_days)
    
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

def extendData(df, extension_days=10):

    if df.empty:
        return df
    
    # 创建扩展数据
    last_date = df.index[-1]
    last_row = df.iloc[-1]
    
    # 生成新日期
    new_dates = [last_date + pd.Timedelta(days=i) for i in range(1, extension_days+1)]
    
    # 创建新数据框
    extended_df = pd.DataFrame(
        [last_row.values] * extension_days,
        index=new_dates,
        columns=df.columns
    )
    
    # 合并数据
    result = pd.concat([df, extended_df])
        
    return result


def calCandleStick(df):

    bullish_ratio = 0
    total_candles = len(df)
    bullish_condition = df['Close'] > df['Open']    
    bullish_count = bullish_condition.sum()
    
    if bullish_count!=0:
        bullish_ratio = round((bullish_count / total_candles),2)

    return bullish_count, bullish_ratio

def calCandleStickBody(df):

    bullish_condition = df['Close'] > df['Open']
    bullish_df = df[bullish_condition].copy()
    strong_bullish = 0
    medium_bullish = 0
    weak_bullish = 0
    
    if len(bullish_df) > 0:
        bullish_df['Body_Size'] = abs(bullish_df['Close'] - bullish_df['Open'])
        bullish_df['Body_Ratio'] = bullish_df['Body_Size'] / (bullish_df['High'] - bullish_df['Low'])                
        # Strong（body > 60%）
        strong_bullish = len(bullish_df[bullish_df['Body_Ratio'] > 0.6])
        # medium（body 30%-60%）
        medium_bullish = len(bullish_df[(bullish_df['Body_Ratio'] >= 0.3) &  (bullish_df['Body_Ratio'] <= 0.6)])        
        # weak（body < 30%）
        weak_bullish = len(bullish_df[bullish_df['Body_Ratio'] < 0.3])  

    return strong_bullish, medium_bullish, weak_bullish
    

def AnalyzeData(sno,stype):
       
    df = pd.read_csv(PATH+"/"+stype+"/"+sno+".csv",index_col=0)
    df = convertData(df)

    #df = calEMA(df)

    tempdf = calHHLL(df)    
    df = checkLHHHLL(df, sno, stype, tempdf)

    # df = calT1(df,22)
    # df = calT1(df,50)
        
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

    YFprocessData("L")
    # YFprocessData("M")
    # YFprocessData("S")

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    