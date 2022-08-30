
import pandas as pd
import numpy as np
import openpyxl
from datetime import datetime, timedelta
import yfinance as yf

stocklist = pd.read_excel("Data/outputlist.xlsx",dtype=str)
outputlist = pd.DataFrame()

for sno in stocklist["股票編號"][:]:
    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    print(sno)

    msft = yf.Ticker(tempsno)
    outputlist = msft.history(period="max")
    outputlist = outputlist.reset_index()
    outputlist["Date"] = outputlist["Date"].dt.strftime("%Y-%m-%d")

    outputlist.to_excel("../StockData/"+sno+".xlsx",index=False)







    
   
