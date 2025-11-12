import pandas as pd
import time as t
from datetime import datetime, timedelta
import yfinance as yf
import concurrent.futures as cf
from tqdm import tqdm
from curl_cffi import requests


PATH = "../SData/USData/"
#SDATE = "2024-01-01"
SDATE = "1980-01-01"
EDATE = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")


def getData(sno,stype):

    session = requests.Session(impersonate="chrome")
    ticker = yf.Ticker(sno+"", session=session)        
    outputlist = ticker.history(period="max",auto_adjust=False)
    outputlist.index = pd.to_datetime(pd.to_datetime(outputlist.index).strftime('%Y%m%d'))
    outputlist = outputlist[outputlist['Volume'] > 0]
    outputlist.insert(0,"sno", sno)    
    outputlist.to_csv(PATH+"/"+stype+"/"+sno+".csv")

    t.sleep(1)    

def YFgetAll(stype):    

    STOCKLIST = pd.read_csv("Data/"+stype+".csv",dtype=str, keep_default_na=False)
    SLIST = STOCKLIST[["SNO"]]    
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST[:]

    with cf.ProcessPoolExecutor(max_workers=5) as executor:
        list(tqdm(executor.map(getData,SLIST["SNO"],SLIST["stype"],chunksize=1),total=len(SLIST)))

    t.sleep(5)

if __name__ == '__main__':
    start = t.perf_counter()

    YFgetAll("XASE")    
    YFgetAll("XNMS")
    YFgetAll("XNCM")
    YFgetAll("XNGS")
    YFgetAll("XNYS")

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')





    
   
