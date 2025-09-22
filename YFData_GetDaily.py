
import pandas as pd
import numpy as np
import os
import openpyxl
import datetime
from datetime import date, timedelta
import yfinance as yf


end_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")


#get stock excel file from path
dir_path = "../../SData/YFData/"
slist = list(map(lambda s: s.replace(".xlsx", ""), os.listdir(dir_path)))

for sno in slist:
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

    df.to_excel(dir_path+"/"+sno+".xlsx",index=False)    
