
from tracemalloc import start
import pandas as pd
import numpy as np
import os
import openpyxl
import datetime
from datetime import date, timedelta
import yfinance as yf
from tqdm import tqdm
from scipy.stats import linregress


def cal_slope(arr):
    #return (arr[-1]-arr[0])/len(arr)
    y = np.array(arr)
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return slope


#get stock excel file from path
dir_path = "../YFData/"
slist = list(map(lambda s: s.replace(".xlsx", ""), os.listdir(dir_path)))

for sno in tqdm(slist[:]):
    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    df = pd.read_excel(dir_path+"/"+sno+".xlsx")

    df["10SMA"] = round(df["Adj Close"].rolling(10).mean(),2)
    df["20SMA"] = round(df["Adj Close"].rolling(20).mean(),2)
    df["30SMA"] = round(df["Adj Close"].rolling(30).mean(),2)
    df["50SMA"] = round(df["Adj Close"].rolling(50).mean(),2)
    df["100SMA"] = round(df["Adj Close"].rolling(100).mean(),2)
    df["150SMA"] = round(df["Adj Close"].rolling(150).mean(),2)
    df["200SMA"] = round(df["Adj Close"].rolling(200).mean(),2)
    df["250SMA"] = round(df["Adj Close"].rolling(250).mean(),2)

    df["30SMA_Slope"] = df["30SMA"].rolling(20).apply(cal_slope)
    df["200SMA_Slope"] = df["200SMA"].rolling(20).apply(cal_slope)
    
    df["V10"] = df["Volume"].rolling(10).mean()
    df["V20"] = df["Volume"].rolling(20).mean()
    df["V50"] = df["Volume"].rolling(50).mean()
    df["V100"] = df["Volume"].rolling(100).mean()
    df["V250"] = df["Volume"].rolling(250).mean()  

    df["L52Week"] = round(df["Close"].rolling(52*5).min(),2)
    df["H52Week"] = round(df["Close"].rolling(52*5).max(),2)

    #df["Change"] = round((df["Adj Close"] - df.loc[:,"Adj Close"].shift(1))/df.loc[:,"Adj Close"].shift(1)*100,2)
    #df["Change%"] = round(df["Adj Close"].pct_change(periods=1)*100,2)

    #((((C - C63) / C63) * .4) + (((C - C126) / C126) * .2) + (((C - C189) / C189) * .2) + (((C - C252) / C252) * .2)) * 100
    #df["RS"] = df["Adj Close"].pct_change(periods=250)

    #df["5Break"] = round((df["Adj Close"].rolling(5).mean() + df["Adj Close"].rolling(5).max() + df["Adj Close"].rolling(5).min())/3,2)

    #df["10Contraction"] = round((df["Adj Close"].rolling(10).max() - df["Adj Close"].rolling(10).min())/df["Adj Close"].rolling(10).min(),2)

    df["5DayResult"] = round(df["Adj Close"].pct_change(periods=5).shift(periods=-5)*100,2)
    
    
    #VCP
    df = df.fillna(0)

    # Condition 1: Current Price > 150 SMA and Current Price > 200 SMA
    df["cond1"] = ((df["Adj Close"] > df["150SMA"]) & (df["Adj Close"] > df["200SMA"]))

    # Condition 2: 150 SMA > 200 SMA
    df["cond2"] = (df["150SMA"] > df["200SMA"])

    # Condition 3: 200 SMA must be trending up for at least 1 month (ideally 4-5 months)
    df["cond3"] = df["200SMA_Slope"] > 0.0

    # Condition 4: 50 SMA > 150 SMA and 150 SMA > 200 SMA
    df["cond4"] = ((df["50SMA"] > df["150SMA"]) & (df["150SMA"]> df["200SMA"]))

    # Condition 5: Current Price > 50 SMA
    df["cond5"] = (df["Adj Close"] > df["50SMA"])

    # Condition 6: Current Price is at least 30%-40% above 52 week low (Many of the best are up 100-300% before coming out of consolidation)
    df["cond6"] = (df["Adj Close"] - df["L52Week"]) / df["L52Week"] > 0.3

    # Condition 7: Current Price is within 15%-25% of 52 week high
    df["cond7"] = ((df["Adj Close"] - df["H52Week"]) / df["H52Week"] < 0.15) & ((df["Adj Close"] - df["H52Week"]) / df["H52Week"] > -0.15) 

    # Condition 8: Pivot(5 day) Breakout
    df["cond8"] = df["Adj Close"] > round((df["Adj Close"].rolling(5).mean() + df["Adj Close"].rolling(5).max() + df["Adj Close"].rolling(5).min())/3,2)

    # Condition 9: true range in the last 10 days is less than 8% of current price (consolidation)
    df["cond9"] = (df["Adj Close"].rolling(10).max() - df["Adj Close"].rolling(10).min())/df["Adj Close"].rolling(10).min() < 0.1

    # Condition 10: 30 SMA must be trending up
    df["cond10"] = df["30SMA_Slope"] > 0.0
    
    df["VCP"] = (df["cond1"] & df["cond2"] & df["cond3"] & df["cond4"] & df["cond5"] & df["cond6"] & df["cond7"] & df["cond8"] & df["cond9"] & df["cond10"])
   
    df.to_excel(dir_path+"/"+sno+".xlsx",index=False)

    print("Finish")