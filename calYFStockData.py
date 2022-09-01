
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
    y = np.array(arr)
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return slope


end_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

#get stock excel file from path
dir_path = "../YFData/"
slist = list(map(lambda s: s.replace(".xlsx", ""), os.listdir(dir_path)))

for sno in tqdm(slist[:1]):
    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    #get file modified date
    filemdate = date.fromtimestamp(os.path.getmtime(dir_path+"/"+sno+".xlsx"))

    #if(filemdate.strftime("%Y%m%d")!=date.today().strftime("%Y%m%d")):
    df = pd.read_excel(dir_path+"/"+sno+".xlsx")
    start_date = str(max(df["SDate"]))
    df.drop(df.shape[0]-1,inplace=True)

    start_date = datetime.datetime.strptime(start_date, "%Y%m%d").date()
    start_date = start_date.strftime("%Y-%m-%d")

    outputlist = yf.download(tempsno, start=start_date, end=end_date, interval='1d', prepost=False)
    outputlist = outputlist.reset_index()
    outputlist.insert(0,"sno", sno)
    outputlist.insert(1,"SDate", outputlist["Date"].dt.strftime("%Y%m%d"))
    outputlist.drop(columns=["Date"], inplace=True)

    df = pd.concat([df, outputlist], ignore_index=True)

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

    df["L52Week"] = round(df["Adj Close"].rolling(52*5).min(),2)
    df["H52Week"] = round(df["Adj Close"].rolling(52*5).max(),2)

    #df["Change"] = round((df["Adj Close"] - df.loc[:,"Adj Close"].shift(1))/df.loc[:,"Adj Close"].shift(1)*100,2)
    df["Change%"] = round(df["Adj Close"].pct_change(periods=1)*100,2)

    #df["RS"] = ((df["Adj Close"] - df["Adj Close"].shift(250))/df["Adj Close"].shift(250))
    #((((C - C63) / C63) * .4) + (((C - C126) / C126) * .2) + (((C - C189) / C189) * .2) + (((C - C252) / C252) * .2)) * 100
    df["RS"] = df["Adj Close"].pct_change(periods=250)

    df["5Break"] = round((df["Adj Close"].rolling(5).mean() + df["Adj Close"].rolling(5).max() + df["Adj Close"].rolling(5).min())/3,2)

    df["10Contraction"] = round((df["Adj Close"].rolling(10).max() - df["Adj Close"].rolling(10).min())/df["Adj Close"].rolling(10).min(),2)

    df.to_excel(dir_path+"/"+sno+".xlsx",index=False)    

    print("Finish")