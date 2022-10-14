
import pandas as pd
import numpy as np
import openpyxl
from datetime import datetime, timedelta
import yfinance as yf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

PATH = "../SData/YFData/"
SDATE = "2021-01-01"
#SDATE = "1980-01-01"
EDATE = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

STOCKLIST = pd.read_excel("Data/stocklist.xlsx",dtype=str)
INDEXLIST = pd.Series(["^HSI","^DJI","^IXIC","^GSPC","^N225","^FTSE","^GDAXI","^FCHI","000001.SS","399001.SZ"])
SLIST = STOCKLIST["sno"][:].append(INDEXLIST)


def getData(sno):
    outputlist = yf.download(sno, interval='1d', auto_adjust=True, start=SDATE, end=EDATE)
    outputlist.insert(0,"sno", sno)
    outputlist = outputlist.loc[outputlist["Volume"]>0]
    outputlist.to_excel(PATH+sno+".xlsx")

def main():
    with ProcessPoolExecutor(max_workers=17) as executor:
        list(tqdm(executor.map(getData,SLIST,chunksize=2),total=len(SLIST)))

if __name__ == '__main__':
    main()






    
   
