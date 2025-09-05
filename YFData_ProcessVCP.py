import pandas as pd
import numpy as np
import yfinance as yf
import concurrent.futures as cf
import os
from tqdm import tqdm
import time as t
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

# Configuration
LOOKBACK_PERIOD = "6mo"

MIN_BASE_DURATION = 30

RSI_PERIOD = 14

ATR_PERIOD = 14

KC_PERIOD = 20

VOLUME_SPIKE_MULTIPLIER = 1.5

ADX_PERIOD = 14

PATH = "../SData/YFData/"

OUTPATH = "../SData/P_YFData/" 

def safe_convert_data(df):

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
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

def CalRSI(df,period):
    # 计算价格变化
    delta = df['Close'].diff()
    
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

def CalRSI_SMA(df,period):
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 使用简单移动平均
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def CalIndicators(df):

    df = safe_convert_data(df)

    if len(df) < max(ATR_PERIOD, KC_PERIOD, ADX_PERIOD) * 2:
        return df

    try:
        # Price indicators
        df['MA20'] = df['Close'].rolling(20, min_periods=10).mean()
        df['MA50'] = df['Close'].rolling(50, min_periods=25).mean()

        # Volatility indicators
        df['EMA20'] = df['Close'].ewm(span=KC_PERIOD, min_periods=KC_PERIOD//2, adjust=False).mean()        

        df['ATR'] = calATR(df,ATR_PERIOD)
        df['Upper_KC'] = df['EMA20'] + 2 * df['ATR']
        df['Lower_KC'] = df['EMA20'] - 2 * df['ATR']
        df['KC_Width'] = (df['Upper_KC'] - df['Lower_KC']) / df['EMA20']

        # Momentum indicators
        tempresult = calADX(df,ADX_PERIOD)
        df['ADX'] = tempresult['ADX']
        df['PlusDI'] = tempresult['PlusDI']
        df['MinusDI'] = tempresult['MinusDI']

        df['RSI'] = CalRSI(df, RSI_PERIOD)

        # ML Anomaly Detection
        imputer = SimpleImputer(strategy='median')
        clean_data = imputer.fit_transform(df[['ATR', 'Volume']])
        model = IsolationForest(contamination=0.1, random_state=42)
        df['Anomaly_Score'] = model.fit_predict(clean_data)

        return df        

    except Exception as e:
        print(f"Indicator error: {str(e)}")
        return df
    


def ProcessVCP(sno):
   
    df = pd.read_excel(PATH+sno+".xlsx")
    df = CalIndicators(df)

    try:
        if df.empty or len(df) < MIN_BASE_DURATION:
            df['VCP'] = 'Insufficient data'
            df.to_excel(OUTPATH+"P_"+sno+".xlsx",index=False)

        # Uptrend check
        price_increase = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        df['Price_Increase'] = f"{price_increase*100:.1f}%"

        if price_increase >= 0.3 and df['Close'].iloc[-1] >= df['MA50'].iloc[-1]:
            df['Uptrend'] = True            

        # Contraction analysis
        contractions = []
        closes = df['Close'].values
        i = len(df) - 1
        contraction_count = 0

        while i > 0 and contraction_count < 6:

            if closes[i] < closes[i-1]:
                start = i
                while i > 0 and closes[i] < closes[i-1]:
                    i -= 1
                end = i

                high = df['High'].iloc[start:end+1].max()
                low = df['Low'].iloc[start:end+1].min()
                retracement = (high - low) / high

                if contractions and retracement > contractions[-1]['retracement'] * 0.6:
                    break

                contractions.append({
                    'retracement': retracement,
                    'kc_width': df['KC_Width'].iloc[start:end+1].mean()
                })

                contraction_count += 1

            i -= 1

        df['Contractions'] = len(contractions)

        if len(contractions) >= 2:
            df['Contraction'] = True
            

        # Pattern validation        
        valid_contractions = all(
            contractions[i]['retracement'] < contractions[i-1]['retracement'] * 0.6
            for i in range(1, len(contractions))
        )

        kc_contraction = all(
            contractions[i]['kc_width'] < contractions[i-1]['kc_width']
            for i in range(1, len(contractions))
        )

        # Additional metrics
        df['ADX_Strength'] = df['ADX'].iloc[-1] > 25
        df['DI_Bullish'] = df['PlusDI'].iloc[-1] > df['MinusDI'].iloc[-1]
        df['RSI_Value'] = round(df['RSI'].iloc[-1], 1)
        df['Anomaly_Free'] = df['Anomaly_Score'].iloc[-1] == 1
        df['Volatility_Decrease'] = f"{(df['KC_Width'].iloc[-60:-30].mean()/df['KC_Width'].iloc[-10:].mean()-1)*100:.1f}%" if len(df) > 60 else 'N/A'

        # Volume analysis
        recent_volume = df['Volume'].iloc[-10:].mean()
        contraction_volume = np.mean([c['kc_width'] for c in contractions[-2:]])
        df['Volume_Contraction'] = contraction_volume < recent_volume * 0.7

        # Breakout check
        resistance = df['High'].iloc[-20:-1].max()
        current_close = df['Close'].iloc[-1]
        volume_spike = df['Volume'].iloc[-1] > df['Volume'].rolling(20).mean().iloc[-1] * VOLUME_SPIKE_MULTIPLIER
        df['Breakout_Detected'] = current_close > resistance and volume_spike

        # Final decision
        if valid_contractions and kc_contraction and df['Volume_Contraction'] and df['Breakout_Detected'] and df['Anomaly_Free']:
            df['VCP'] = True
        else:
            df['VCP'] = False

        df.to_excel(OUTPATH+"P_"+sno+".xlsx",index=False)


    except Exception as e:
        df['Reason'] = f"Analysis error: {str(e)}"
        df.to_excel(OUTPATH+"P_"+sno+".xlsx",index=False)

    





def main():

    SLIST = list(map(lambda s: s.replace(".xlsx", ""), os.listdir(PATH)))
    SLIST = SLIST[:1]

    with cf.ProcessPoolExecutor(max_workers=17) as executor:
        list(tqdm(executor.map(ProcessVCP,SLIST,chunksize=2),total=len(SLIST)))


if __name__ == '__main__':
    start = t.perf_counter()

    main()

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    