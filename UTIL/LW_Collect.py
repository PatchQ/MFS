import pandas as pd
import time as t
from datetime import datetime, timedelta
import yfinance as yf
import concurrent.futures as cf
from tqdm import tqdm
import os
import platform

PATH = "../SData/YFData/"

def getYFDaily(sno, sdate):
    try:
        stock = yf.Ticker(sno)
                        
        data = stock.history(start=sdate, auto_adjust=False)
        
        if data.empty:
            return None

        data = data[data['Volume'] > 0]
        data.insert(0,"sno", sno)
        data = data.reset_index()        
        data['Date'] = pd.to_datetime(data['Date']).dt.date        
        
        return data
        
    except Exception as e:
        print(f"Get {sno} Data Error: {str(e)}")
        return None

def getYFAll(sno,stype,period):        
    ticker = yf.Ticker(sno)
    outputlist = ticker.history(period=period,auto_adjust=False)
    #outputlist = ticker.history(start="2023-01-01",end="2025-12-31",auto_adjust=False)
    
    outputlist.index = pd.to_datetime(pd.to_datetime(outputlist.index).strftime('%Y%m%d'))
    outputlist = outputlist[outputlist['Volume'] > 0]
    outputlist.insert(0,"sno", sno)    
    outputlist.to_csv(PATH+"/"+stype+"/"+sno+".csv")

def getDataDaily(sno,stype):        
    
    existing_data = pd.read_csv(PATH+"/"+stype+"/"+sno+".csv")
    existing_data['Date'] = pd.to_datetime(existing_data['Date']).dt.date
        
    if len(existing_data) == 0:
        last_date = datetime(1900, 1, 1).date()
    else:        
        last_date = existing_data['Date'].max()
        if isinstance(last_date, str):
            last_date = datetime.strptime(last_date, '%Y-%m-%d').date()

    start_date = last_date - timedelta(days=1)
    today = datetime.now().date()

    if start_date > today:
        print(f"{sno} 已是最新資料，無需更新")
        return
        
    new_data = getYFDaily(sno, start_date.strftime('%Y-%m-%d'))
    
    if new_data is None or new_data.empty:
        print(f"{sno} No New Data")
        return

    updated_data = pd.concat([existing_data, new_data], ignore_index=True)        
    updated_data = updated_data.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')        
    updated_data.to_csv(PATH+"/"+stype+"/"+sno+".csv", index=False)
        

def YFgetAll(stype,period="max"):
    STOCKLIST = pd.read_csv("Data/stocklist_"+stype+".csv",dtype=str)
    #INDEXLIST = pd.Series(["^HSI","^DJI","^IXIC","^GSPC","^N225","^FTSE","^GDAXI","^FCHI","000001.SS","399001.SZ"])
    SLIST = STOCKLIST[["sno"]]
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST.assign(period=period+"")
    SLIST = SLIST[:]

    if platform.system()=="Windows":
        executor = cf.ProcessPoolExecutor(max_workers=5)
    elif platform.system()=="Darwin":
        executor = cf.ThreadPoolExecutor(max_workers=4)    

    with executor:
        list(tqdm(executor.map(getYFAll,SLIST["sno"],SLIST["stype"],SLIST["period"],chunksize=1),total=len(SLIST)))

def YFgetDaily(stype):

    snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(PATH+"/"+stype+"/")))
    SLIST = pd.DataFrame(snolist, columns=["sno"])
    #INDEXLIST = pd.Series(["^HSI","^DJI","^IXIC","^GSPC","^N225","^FTSE","^GDAXI","^FCHI","000001.SS","399001.SZ"])    
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST[:]

    if platform.system()=="Windows":
        executor = cf.ProcessPoolExecutor(max_workers=5)
    elif platform.system()=="Darwin":
        executor = cf.ThreadPoolExecutor(max_workers=4)

    with executor:
        list(tqdm(executor.map(getDataDaily,SLIST["sno"],SLIST["stype"],chunksize=1),total=len(SLIST)))        


if __name__ == '__main__':
    start = t.perf_counter()

    YFgetAll("L")
    YFgetAll("M")

    #YFgetAll("L","2y")
    #YFgetAll("M","2y")

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')





    
   
