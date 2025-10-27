import pandas as pd

def convertData(df):

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=numeric_cols, how='all', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

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
    bullish_condition = df['Close'] >= df['Open']    
    bullish_count = bullish_condition.sum()
    
    if bullish_count!=0:
        bullish_ratio = round((bullish_count / total_candles),2)

    return bullish_count, bullish_ratio

def calCandleStickBody(df):

    bullish_condition = df['Close'] >= df['Open']
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


def calEMA(df):

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
    

def calT1(df, days, threshold=0.1):
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
