import pandas as pd
import numpy as np
import yfinance as yf
import concurrent.futures as cf
import os
from tqdm import tqdm
import time as t
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import warnings

PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/" 
DAYS=0
TOLERANCE=0.001
WINDOW=10

def convertData(df):

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=numeric_cols, how='all', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def calATR(df, period):
    # 1. 计算真实波幅 (TR)
    # 由于需要前一天的close，所以使用.shift()来获取前一期数据
    prev_close = df['Close'].shift(1)
    
    # 计算TR的三项组成部分
    tr1 = df['High'] - df['Low'] # 当日波幅
    tr2 = (df['High'] - prev_close).abs() # 向上跳空缺口
    tr3 = (df['Low'] - prev_close).abs() # 向下跳空缺口
    
    # TR是这三者的最大值
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 使用Wilder平滑方法 (alpha = 1/period)
    # 注意: 这里使用adjust=False确保与TA-Lib计算一致
    atr = tr.ewm(alpha = 1/period, min_periods=period, adjust=False).mean()

    return atr

def calADX(df, period):
    # 1. 计算真实波幅 (TR)、正向移动和负向移动 (+DM和-DM)
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # 使用 .shift() 获取前一期数据
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    
    # 计算真实波幅 (TR)，与ATR计算中的TR相同
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 计算方向移动 (Directional Movement)
    up_move = high - prev_high   # 今日最高 - 昨日最高
    down_move = prev_low - low   # 昨日最低 - 今日最低
    
    # 初始化 +DM 和 -DM
    plus_dm = pd.Series(0, index=df.index)
    minus_dm = pd.Series(0, index=df.index)
    
    # 确定有效的方向移动
    # +DM 的条件：上涨幅度大于下跌幅度 AND 上涨幅度 > 0
    plus_dm_condition = (up_move > down_move) & (up_move > 0)
    plus_dm[plus_dm_condition] = up_move[plus_dm_condition]
    
    # -DM 的条件：下跌幅度大于上涨幅度 AND 下跌幅度 > 0
    minus_dm_condition = (down_move > up_move) & (down_move > 0)
    minus_dm[minus_dm_condition] = down_move[minus_dm_condition]
    
    # 2. 平滑TR, +DM, -DM (通常使用Wilder的平滑方法，即EMA的一种变体)
    # 在Pandas中，`ema(alpha=1/period)` 等价于Wilder的平滑
    tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()
    
    # 3. 计算方向指标 (+DI 和 -DI)
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # 4. 计算方向指数 (DX)
    dx = 100 * ( (plus_di - minus_di).abs() / (plus_di + minus_di) )
    
    # 5. 计算平均方向指数 (ADX) - 对DX进行平滑
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    # 将结果组合成一个DataFrame
    result_df = pd.DataFrame({
        'PlusDI': plus_di,
        'MinusDI': minus_di,
        'ADX': adx
    })
    
    return result_df

def calRSI(df,period):
    # 计算价格变化
    delta = df['Adj Close'].diff()
    
    # 分离上涨和下跌的变化
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 计算平均上涨和平均下跌（使用指数移动平均）
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    # 计算相对强度 (RS)
    rs = avg_gain / avg_loss
    
    # 计算RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calRSI_SMA(df,period):
    
    delta = df['Adj Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 使用简单移动平均
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def CheckEMA(df):

    try:
        # Volatility indicators
        df['EMA10'] = df['Adj Close'].ewm(span=10, min_periods=5, adjust=False).mean()
        df['EMA22'] = df['Adj Close'].ewm(span=22, min_periods=11, adjust=False).mean()     
        df['EMA50'] = df['Adj Close'].ewm(span=50, min_periods=25, adjust=False).mean()     
        df['EMA100'] = df['Adj Close'].ewm(span=100, min_periods=50, adjust=False).mean()             
        df['EMA250'] = df['Adj Close'].ewm(span=250, min_periods=125, adjust=False).mean()

        df['EMA1'] = ((df["EMA10"] > df["EMA22"]) & (df["EMA22"] > df["EMA50"]) & (df["EMA50"] > df["EMA100"]) & (df["EMA100"] > df["EMA250"]))        
        df['EMA2'] = ((df["EMA10"] > df["EMA22"]) & (df["EMA22"] > df["EMA50"]))        
        
        return df        

    except Exception as e:
        print(f"Indicator error: {str(e)}")
        df['EMA1'] = False
        df['EMA2'] = False
        return df
    

def CheckT1(df, days, threshold=0.1):
    try:

        if len(df) < days:
            print(f"數據不足，無法計算 {days} 天波動")
            df["T1_"+str(days)] = False
            return df
        
        # 取最近N天的數據
        #recent_data = df.tail(days)
        
        # 計算最高價、最低價
        highest = df['High'].rolling(window=days).max()
        lowest = df['Low'].rolling(window=days).min()
                
        # 計算波動幅度
        volatility = (highest - lowest) / lowest
        
        # 檢查波動是否在閾值內
        df["T1_"+str(days)] = volatility <= threshold

        return df

    except Exception as e:
        print(f"Indicator error: {str(e)}")
        df["T1_"+str(days)] = False
        return df

def checkLHHHLL(df, sno, stype, swing_analysis):

    #print(sno)

    df = df.reset_index()    
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
    df['BOSS_STATUS'] = ""
    
    

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


            date_match = (df["Date"] == swing_analysis['date'].iloc[i])
            df.loc[date_match, "classification"] = swing_analysis["classification"].iloc[i]
            df.loc[date_match, "LLLow"] = swing_analysis["LLLow"].iloc[i]
            df.loc[date_match, "LLDate"] = swing_analysis["LLDate"].iloc[i]
            df.loc[date_match, "HHClose"] = swing_analysis["HHClose"].iloc[i]
            df.loc[date_match, "HHDate"] = swing_analysis["HHDate"].iloc[i]
            df.loc[date_match, "HHHigh"] = swing_analysis["HHHigh"].iloc[i]
            df.loc[date_match, "PATTERN"] = swing_analysis["PATTERN"].iloc[i]
            

    swing_analysis["BOSS1"] = ((swing_analysis['PATTERN']=="LHLLHH") & (swing_analysis['HHClose']>swing_analysis['price']))
    #swing_analysis["BOSS2"] = ((swing_analysis['PATTERN']=="HHHLHH") & (nowprice>swing_analysis['HLLow']))    
    #swing_analysis.to_csv(OUTPATH+"/HHLL/HL_"+sno+".csv",index=False)

    BOSSRule1 = df['PATTERN']=="LHLLHH"
    BOSSRule2 = df['HHClose']>df['High']
    BOSSRule3 = df['LLLow']<df['Low'].rolling(window=22).min()  

    #df["BOSS1"] = ((df['PATTERN']=="LHLLHH") & (df['HHClose']>df['High']))    
    df["BOSS1"] = (BOSSRule1 & BOSSRule2 & BOSSRule3)    
    tempdf = df.loc[df["BOSS1"]]

    df.set_index("Date", inplace=True)
    df.index = pd.to_datetime(df.index)

    df['bullish_count'] = 0
    df['bullish_ratio'] = 0.00
    df['strong_bullish'] = 0
    df['medium_bullish'] = 0
    df['weak_bullish'] = 0   
    df['buy_price'] = 0.00
    df['tp1'] = 0.00
    df['tp2'] = 0.00
    

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
    df.loc[df["BOSS1"], "tp1"] = df["HHHigh"]
    df.loc[df["BOSS1"], "tp2"] = df["buy_price"] + (df["HHHigh"] - df["buy_price"]) * 2 
    df.loc[df["BOSS1"], "BOSS_STATUS"] = "SB-"+df.loc[df["BOSS1"]].index.strftime("%Y%m%d")

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

        hhdate = pd.to_datetime(tempdf["HHDate"].iloc[i])
        startbossdate = tempdf['Date'].iloc[i].strftime("%Y%m%d")     
        buy_price = tempdf["buy_price"].iloc[i]
        cl_price = tempdf["LLLow"].iloc[i]
        tp1_price = tempdf["tp1"].iloc[i]
        tp2_price = tempdf["tp2"].iloc[i]

        buydate_mask = (df.index > hhdate) & (buy_price>=df["Low"])
        buydates = df[buydate_mask].index

        if len(buydates)!=0:
            buy = True            
            lastbuydate = buydates[0]

        if buy:
            df.loc[lastbuydate,'BOSS_STATUS'] = "BUY-"+startbossdate
            #print("BUYDate : "+lastbuydate.strftime("%Y-%m-%d"))

            tp1date_mask = (df.index >= lastbuydate) & (df["High"]>=tp1_price)
            tp1dates = df[tp1date_mask].index

            cl1date_mask = (df.index >= lastbuydate) & (cl_price>=df["Low"])
            cl1dates = df[cl1date_mask].index            

            if len(tp1dates)!=0:
                tp1=True
                lasttp1date = tp1dates[0]

            if len(cl1dates)!=0:                
                cl1=True
                lastcl1date = cl1dates[0]

            if ((lastcl1date<=lasttp1date) & tp1 & cl1):
                df.loc[lastcl1date,'BOSS_STATUS'] = "CL1-"+startbossdate
                #print("CL1 : "+lastcl1date.strftime("%Y-%m-%d"))
            else:
                if tp1:
                    df.loc[lasttp1date,'BOSS_STATUS'] = "TP1-"+startbossdate
                    #print("TP1 : "+lasttp1date.strftime("%Y-%m-%d"))     

                    tp2date_mask = (df.index >= lasttp1date) & (df["High"]>=tp2_price)
                    tp2dates = df[tp2date_mask].index        
                
                    cl2date_mask = (df.index >= lasttp1date) & (cl_price>=df["Low"])
                    cl2dates = df[cl2date_mask].index

                    if len(tp2dates)!=0:
                        tp2 = True                
                        lasttp2date = tp2dates[0]

                    if len(cl2dates)!=0:
                        cl2 = True
                        lastcl2date = cl2dates[0]

                    if ((lastcl2date<=lasttp2date) & tp2 & cl2):               
                        df.loc[lastcl2date,'BOSS_STATUS'] = "CL2-"+startbossdate
                        #print("CL2 : "+lastcl2date.strftime("%Y-%m-%d"))
                    else:
                        if tp2:
                            df.loc[lasttp2date,'BOSS_STATUS'] = "TP2-"+startbossdate
                            #print("TP2 : "+lasttp2date.strftime("%Y-%m-%d"))

    return df


def calHHLL(df):
        
    stock = df.copy()

    stock.index = pd.to_datetime(stock.index,utc=True).tz_convert('Asia/Shanghai') 

    stock = extendData(stock)

    
   
    # 找出摆动点
    swing_highs, swing_lows = find_swing_points(stock['High'], stock['Low'], stock["Close"])
    
    # 分类所有摆动点
    swing_analysis = classify_all_swing_points(swing_highs, swing_lows)

    return swing_analysis
   

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
    df = CheckEMA(df)
    df = CheckT1(df,22)
    df = CheckT1(df,50)

    tempdf = calHHLL(df)    

    df = checkLHHHLL(df, sno, stype, tempdf)
    
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
    YFprocessData("M")
    YFprocessData("S")

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    