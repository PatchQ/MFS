
import pandas as pd
import numpy as np
import openpyxl
from datetime import datetime, timedelta
import yfinance as yf

stocklist = pd.read_excel("filterstock1.xlsx",dtype=str)
outputlist = pd.DataFrame()

for sno in stocklist["股票編號"][:]:
    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    print(sno)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=500)

    history = yf.download(tempsno, start=start_date, end=end_date, interval='1d', prepost=False)


    history = history.dropna(how="all", axis=0).dropna(how="all", axis=1)
    history = history.replace(np.inf,np.nan).dropna()
    history = history.apply(lambda s: s.astype(str).str.replace(",","").astype(float))
    history = history.reset_index()

    history["Date"] = history["Date"].dt.strftime("%Y-%m-%d")
    history["股票編號"] = sno
    
    history["10SMA"] = history["Close"].rolling(10).mean()
    history["20SMA"] = history["Close"].rolling(20).mean()
    history["50SMA"] = history["Close"].rolling(50).mean()
    history["100SMA"] = history["Close"].rolling(100).mean()
    history["250SMA"] = history["Close"].rolling(250).mean()
    
    history["V10"] = history["Volume"].rolling(10).mean()
    history["V20"] = history["Volume"].rolling(20).mean()
    history["V50"] = history["Volume"].rolling(50).mean()
    history["V100"] = history["Volume"].rolling(100).mean()
    history["V250"] = history["Volume"].rolling(250).mean()  

    tmpstocklist = stocklist.loc[stocklist["股票編號"]==sno]
    tmphistory = pd.DataFrame(history.iloc[-1:],columns=history.columns)

    outputlist = pd.concat([outputlist,tmpstocklist.merge(tmphistory,on="股票編號")])


outputlist.to_excel("outputlist.xlsx",index=False)
print("test4")