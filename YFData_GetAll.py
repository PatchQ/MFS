import pandas as pd
import time as t
from datetime import datetime, timedelta
import yfinance as yf
import concurrent.futures as cf
from tqdm import tqdm


PATH = "../SData/YFData/"
#SDATE = "2024-01-01"
SDATE = "1980-01-01"
EDATE = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")


def getData(sno,stype):        
    ticker = yf.Ticker(sno)
    outputlist = ticker.history(period="max",auto_adjust=False)
    outputlist.index = pd.to_datetime(pd.to_datetime(outputlist.index).strftime('%Y%m%d'))
    outputlist = outputlist[outputlist['Volume'] > 0]
    outputlist.insert(0,"sno", sno)    
    outputlist.to_excel(PATH+"/"+stype+"/"+sno+".xlsx")

def main(stype):
    STOCKLIST = pd.read_excel("Data/stocklist_"+stype+"A.xlsx",dtype=str)
    #INDEXLIST = pd.Series(["^HSI","^DJI","^IXIC","^GSPC","^N225","^FTSE","^GDAXI","^FCHI","000001.SS","399001.SZ"])
    SLIST = STOCKLIST[["sno"]]
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST[:]

    with cf.ProcessPoolExecutor(max_workers=12) as executor:
        list(tqdm(executor.map(getData,SLIST["sno"],SLIST["stype"],chunksize=1),total=len(SLIST)))


def main_ipad(stype):
    STOCKLIST = pd.read_excel("Data/stocklist_"+stype+"A.xlsx",dtype=str)
    #INDEXLIST = pd.Series(["^HSI","^DJI","^IXIC","^GSPC","^N225","^FTSE","^GDAXI","^FCHI","000001.SS","399001.SZ"])
    SLIST = STOCKLIST[["sno"]]
    SLIST = SLIST.assign(stype=stype+"")    
    SLIST = SLIST[:]

    with cf.ThreadPoolExecutor(max_workers=12) as executor:
        list(tqdm(executor.map(getData,SLIST["sno"],SLIST["stype"],chunksize=1),total=len(SLIST)))


if __name__ == '__main__':
    start = t.perf_counter()

    main("L")
    main("M")
    main("S")

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')





    
   
