
import pandas as pd
import numpy as np
import openpyxl
from datetime import datetime, timedelta
import yfinance as yf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

stocklist = pd.read_excel("Data/stocklist.xlsx",dtype=str)
outputlist = pd.DataFrame()
slist = stocklist["股票編號"][:]

def getData(sno):
    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    outputlist = yf.download(tempsno, interval='1d', prepost=False)
    outputlist.insert(0,"sno", sno)
    outputlist = outputlist.loc[outputlist["Volume"]>0]
    outputlist.to_excel("../YFData/"+sno+".xlsx")


def main():
    with ProcessPoolExecutor(max_workers=17) as executor:
        list(tqdm(executor.map(getData,slist,chunksize=2),total=len(slist)))

if __name__ == '__main__':
    main()






    
   
