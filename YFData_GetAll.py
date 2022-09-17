
import pandas as pd
import numpy as np
import openpyxl
from datetime import datetime, timedelta
import yfinance as yf
from tqdm import tqdm

stocklist = pd.read_excel("Data/outputlist.xlsx",dtype=str)
outputlist = pd.DataFrame()

for sno in tqdm(stocklist["股票編號"][:1]):
    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    outputlist = yf.download(tempsno, interval='1d', prepost=False)
    outputlist.insert(0,"sno", sno)

    outputlist.to_excel("../YFData/"+sno+".xlsx")







    
   
