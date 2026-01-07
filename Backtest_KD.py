import pandas as pd
import numpy as np
import openpyxl
import os
import datetime
from tqdm import tqdm
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
from talib import abstract

stype="L"

#get stock excel file from path
OUTPATH = "../SData/YFData/" 
snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(OUTPATH+"/"+stype)))
snolist = snolist[7:8]

for sno in tqdm(snolist):
    print(sno)
    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)
    
    df = pd.read_csv(OUTPATH+"/"+stype+"/"+sno+".csv")

    ## 整理資料格式
    #df = df.rename(columns={"date": "Date"})
    df.set_index("Date" , inplace=True)
    df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df.index)))
    ## backtesting.py 格式
    df1 = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "Volume"})
    ## ta-lib 格式
    #df2 = df.rename(columns={"max": "high", "min": "low", "Trading_Volume": "Volume"})
    ## 取得 KD 值
    df_kd = abstract.STOCH(df1,fastk_period=9, slowk_period=3,slowd_period=3)
    ## 合併資料
    df = pd.merge(df, df_kd, on="Date") 
    print(df)   
    #df = df.loc[df.index>"2019-12-31"]
    #df.drop(columns=["sno"], inplace=True)


class KdCross(Strategy):

    #sl_ratio = 99     # stop loss ratio, 99 means 1% loss

    def init(self):
        super().init()

    def next(self):
        if crossover(20, self.data.slowk):
            self.buy()
            print(self.data.index, self.trades, self.position.pl_pct , self.position.size)
        
        elif crossover(self.data.slowk, 80): 
            self.position.close()
            print(self.data.index, self.trades, self.position.pl_pct , self.position.size)
            
bt = Backtest(df, KdCross,cash=10000,commission=.002)
output = bt.run()
bt.plot()
print(output)

