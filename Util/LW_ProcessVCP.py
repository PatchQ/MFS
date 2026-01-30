import pandas as pd
import numpy as np
import concurrent.futures as cf
import os
from tqdm import tqdm
import time as t
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

try:
    from LW_Calindicator import *
    from LW_CalHHHL import *
    from LW_BossSkill import *

except ImportError:

    from UTIL.LW_Calindicator import *
    from UTIL.LW_CalHHHL import *
    from UTIL.LW_BossSkill import *    

# Configuration

MIN_BASE_DURATION = 30

RSI_PERIOD = 14

ATR_PERIOD = 14

KC_PERIOD = 20

VOLUME_SPIKE_MULTIPLIER = 1.5

ADX_PERIOD = 14

PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/" 

def CalIndicator(df):
            
    df = convertData(df)
    df = calEMA(df)
    
    # Volatility indicators
    df['ATR'] = calATR(df,ATR_PERIOD)
    df['Upper_KC'] = df['EMA22'] + 2 * df['ATR']
    df['Lower_KC'] = df['EMA22'] - 2 * df['ATR']
    df['KC_Width'] = (df['Upper_KC'] - df['Lower_KC']) / df['EMA22']

    # Momentum indicators
    tempresult = calADX(df,ADX_PERIOD)
    df['ADX'] = tempresult['ADX']
    df['PlusDI'] = tempresult['PlusDI']
    df['MinusDI'] = tempresult['MinusDI']
    df['RSI'] = calRSI(df, RSI_PERIOD)

    # ML Anomaly Detection
    imputer = SimpleImputer(strategy='median')
    clean_data = imputer.fit_transform(df[['ATR', 'Volume']])
    model = IsolationForest(contamination=0.1, random_state=42)
    df['Anomaly_Score'] = model.fit_predict(clean_data)

    return df.dropna()       


def AnalyzeData(sno,stype):

    tempdf = pd.read_csv(PATH+"/"+stype+"/"+sno+".csv",index_col=0) 
    df = tempdf.tail(300).copy()
    df = CalIndicator(df)

    try:
        if df.empty or len(df) < MIN_BASE_DURATION:
            df['VCP'] = False       
            df = df.reset_index()    
            df.to_csv(OUTPATH+"/"+stype+"/P_"+sno+".csv",index=False)

        # 1. Uptrend Requirement:
        #30%+ price increase  價格上漲30%以上
        #Price above 50-day MA  價格高於 50 日均線
        price_increase = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]

        df['Price_Increase'] = f"{price_increase*100:.1f}%"

        if price_increase < 0.3 or df['Close'].iloc[-1] < df['EMA50'].iloc[-1]:

            df['Uptrend'] = False
        else:
            df['Uptrend'] = True

        # 2. Contraction Structure: 收縮結構
        #At least 2 successive contractions
        #Each contraction ≤ 50% of previous retracement每次收縮幅度≤前一次回檔幅度的50%
        #Volatility (KC Width) must decrease with each contraction波動率（KC 寬度）必須隨著每次收縮而降低。            

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
        df['VCP'] = all([valid_contractions, kc_contraction, df['Volume_Contraction'], df['Breakout_Detected'], df['Anomaly_Free']])

        df = df.reset_index()
        df.to_csv(OUTPATH+"/"+stype+"/P_"+sno+".csv",index=False)


    except Exception as e:
        df['Reason'] = f"Analysis error: {str(e)}"
        df['VCP'] = False
        df = df.reset_index()
        df.to_csv(OUTPATH+"/"+stype+"/P_"+sno+".csv",index=False)



def ProcessVCP(stype):

    snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(PATH+"/"+stype)))
    SLIST = pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST[:]

    with cf.ProcessPoolExecutor(max_workers=5) as executor:
        list(tqdm(executor.map(AnalyzeData,SLIST["sno"],SLIST["stype"],chunksize=1),total=len(SLIST)))


if __name__ == '__main__':
    start = t.perf_counter()

    ProcessVCP("L")    
    ProcessVCP("M")
    #ProcessBOSS("S")

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
        