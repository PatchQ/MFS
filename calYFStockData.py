
from tracemalloc import start
import pandas as pd
import numpy as np
import os
import openpyxl
import datetime
from datetime import date, timedelta
import yfinance as yf


end_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")


#get stock excel file from path
dir_path = "../YFData/"
slist = list(map(lambda s: s.replace(".xlsx", ""), os.listdir(dir_path)))

for sno in slist[:1]:
    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    #get file modified date
    filemdate = date.fromtimestamp(os.path.getmtime(dir_path+"/"+sno+".xlsx"))

    #if(filemdate.strftime("%Y%m%d")!=date.today().strftime("%Y%m%d")):
    print(sno)
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
    df["50SMA"] = round(df["Adj Close"].rolling(50).mean(),2)
    df["100SMA"] = round(df["Adj Close"].rolling(100).mean(),2)
    df["150SMA"] = round(df["Adj Close"].rolling(150).mean(),2)
    df["200SMA"] = round(df["Adj Close"].rolling(200).mean(),2)
    df["250SMA"] = round(df["Adj Close"].rolling(250).mean(),2)
    
    df["V10"] = df["Volume"].rolling(10).mean()
    df["V20"] = df["Volume"].rolling(20).mean()
    df["V50"] = df["Volume"].rolling(50).mean()
    df["V100"] = df["Volume"].rolling(100).mean()
    df["V250"] = df["Volume"].rolling(250).mean()  

    df["L52Week"] = round(df["Adj Close"].rolling(250).min(),2)
    df["H52Week"] = round(df["Adj Close"].rolling(250).max(),2)

    df["Change"] = round((df["Adj Close"] - df.loc[:,"Adj Close"].shift(1))/df.loc[:,"Adj Close"].shift(1)*100,2)

    df.to_excel(dir_path+"/"+sno+".xlsx",index=False)    
