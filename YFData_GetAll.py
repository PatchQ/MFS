import pandas as pd
import time as t
from datetime import datetime, timedelta
import yfinance as yf
import concurrent.futures as cf
from tqdm import tqdm


PATH = "../SData/YFData/"
SDATE = "2021-01-01"
#SDATE = "1980-01-01"
EDATE = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

STOCKLIST = pd.read_excel("Data/stocklist.xlsx",dtype=str)
INDEXLIST = pd.Series(["^HSI","^DJI","^IXIC","^GSPC","^N225","^FTSE","^GDAXI","^FCHI","000001.SS","399001.SZ"])
SLIST = STOCKLIST["sno"][:].append(INDEXLIST)
SLIST = SLIST[:]

def getData(sno):
    outputlist = yf.download(sno, interval='1d', auto_adjust=True, start=SDATE, end=EDATE, progress=False)
    outputlist.insert(0,"sno", sno)
    outputlist = outputlist.loc[outputlist["Volume"]>0]
    outputlist.to_excel(PATH+sno+".xlsx")

def main():
    with cf.ProcessPoolExecutor(max_workers=17) as executor:
        list(tqdm(executor.map(getData,SLIST,chunksize=2),total=len(SLIST)))


def main2():
    with cf.ThreadPoolExecutor(max_workers=17) as executor:
        executor.map(getData,SLIST)


if __name__ == '__main__':
    start = t.perf_counter()

    main()

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')





    
   
