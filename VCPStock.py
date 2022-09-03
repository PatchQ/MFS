import pandas as pd
import numpy as np
import os
import openpyxl
import datetime
from datetime import date, timedelta
import yfinance as yf
from tqdm import tqdm
from scipy.stats import linregress

#get stock excel file from path
dir_path = "../YFData/"
slist = list(map(lambda s: s.replace(".xlsx", ""), os.listdir(dir_path)))

for sno in tqdm(slist[:]):
    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    #get file modified date
    filemdate = date.fromtimestamp(os.path.getmtime(dir_path+"/"+sno+".xlsx"))

    #if(filemdate.strftime("%Y%m%d")!=date.today().strftime("%Y%m%d")):
    df = pd.read_excel(dir_path+"/"+sno+".xlsx", converters = {"sno":str,"SDate":str}, dtype=float)
    df = df.fillna(0)

    # Condition 1: Current Price > 150 SMA and Current Price > 200 SMA
    df["cond1"] = ((df["Adj Close"] > df["150SMA"]) & (df["Adj Close"] > df["200SMA"]))

    # Condition 2: 150 SMA > 200 SMA
    df["cond2"] = (df["150SMA"] > df["200SMA"])

    # Condition 3: 200 SMA must be trending up for at least 1 month (ideally 4-5 months)
    df["cond3"] = df["200SMA_Slope"] > 0.0

    # Condition 4: 50 SMA > 150 SMA and 150 SMA > 200 SMA
    df["cond4"] = ((df["50SMA"] > df["150SMA"]) & (df["150SMA"]> df["200SMA"]))

    # Condition 5: Current Price > 50 SMA
    df["cond5"] = (df["Adj Close"] > df["50SMA"])

    # Condition 6: Current Price is at least 30%-40% above 52 week low (Many of the best are up 100-300% before coming out of consolidation)
    df["cond6"] = (df["Adj Close"] - df["L52Week"]) / df["L52Week"] > 0.3

    # Condition 7: Current Price is within 15%-25% of 52 week high
    df["cond7"] = ((df["Adj Close"] - df["H52Week"]) / df["H52Week"] < 0.15) & ((df["Adj Close"] - df["H52Week"]) / df["H52Week"] > -0.15) 

    # Condition 8: Pivot(5 day) Breakout
    df["cond8"] = df["Adj Close"] > df["5Break"]

    # Condition 9: true range in the last 10 days is less than 8% of current price (consolidation)
    df["cond9"] = df["10Contraction"] < 0.1

    df["VCP"] = (df["cond1"] & df["cond2"] & df["cond3"] & df["cond4"] & df["cond5"] & df["cond6"] & df["cond7"] & df["cond8"] & df["cond9"])

    df.to_excel(dir_path+"/"+sno+".xlsx",index=False)
               