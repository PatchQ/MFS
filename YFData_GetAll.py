import pandas as pd
import time as t
from datetime import datetime, timedelta
import yfinance as yf
import concurrent.futures as cf
from tqdm import tqdm


PATH = "../../SData/YFData/"
SDATE = "2024-01-01"
#SDATE = "1980-01-01"
EDATE = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")


STOCKLIST_L = pd.read_excel("Data/stocklist_L.xlsx",dtype=str)
STOCKLIST_M = pd.read_excel("Data/stocklist_M.xlsx",dtype=str)
STOCKLIST_S = pd.read_excel("Data/stocklist_S.xlsx",dtype=str)

INDEXLIST = pd.Series(["^HSI","^DJI","^IXIC","^GSPC","^N225","^FTSE","^GDAXI","^FCHI","000001.SS","399001.SZ"])

#SLIST_L = STOCKLIST_L["sno"].append(INDEXLIST)
SLIST_L = STOCKLIST_L[["sno"]]
SLIST_L = SLIST_L.assign(type="L")

SLIST_M = STOCKLIST_M[["sno"]]
SLIST_M = SLIST_M.assign(type="M")

SLIST_S = STOCKLIST_S[["sno"]]
SLIST_S = SLIST_S.assign(type="S")

SLIST = pd.concat([SLIST_L, SLIST_M], ignore_index=True)
SLIST = pd.concat([SLIST, SLIST_S], ignore_index=True)

SLIST = SLIST[:1]


def getData(sno,type):    
    outputlist = yf.download(sno, threads=False, interval='1d', auto_adjust=True, start=SDATE, end=EDATE, progress=False, show_errors=False)
    outputlist.insert(0,"sno", sno)
    outputlist = outputlist.loc[outputlist["Volume"]>0]
    outputlist.to_excel(PATH+"/"+sno+".xlsx")

def main():
    with cf.ProcessPoolExecutor(max_workers=17) as executor:
        list(tqdm(executor.map(getData,SLIST["sno"],SLIST["type"],chunksize=2),total=len(SLIST)))

def main_ipad():
    with cf.ThreadPoolExecutor(max_workers=17) as executor:
        list(tqdm(executor.map(getData,SLIST["sno"],SLIST["type"],chunksize=2),total=len(SLIST)))


if __name__ == '__main__':
    start = t.perf_counter()

    #main_ipad()
    hk005 = yf.Ticker("0005.HK")
    outputlist = hk005.history(period="1mo")
    print(outputlist)

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')





    
   
