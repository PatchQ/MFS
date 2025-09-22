import pandas as pd
import numpy as np
import openpyxl
import os
import datetime
from tqdm import tqdm
from backtesting import Backtest, Strategy
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


#get stock excel file from path
dir_path = "../SData/P_YFData/"
slist = list(map(lambda s: s.replace(".xlsx", ""), os.listdir(dir_path)))
slist = slist[:]
slist = ["P_^HSI"]

for sno in tqdm(slist):
    print(sno)
    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    df = pd.read_excel(dir_path+sno+".xlsx",index_col=0)
    #df = df.loc[df.index>"2019-12-31"]
    df.drop(columns=["sno"], inplace=True)


class AI_test(Strategy):

    sl_ratio = 99     # stop loss ratio, 99 means 1% loss

    def init(self):
        return

    def next(self):
        if self.data.VCP:
            self.buy()
            print(self.data.index, self.trades, self.position.pl_pct , self.position.size)
        
        if self.position.pl_pct < -.04:
            self.position.close()
            print(self.data.index, self.trades, self.position.pl_pct , self.position.size)
            
        if self.position.pl_pct > .05:
            self.position.close()
            print(self.data.index, self.trades, self.position.pl_pct , self.position.size)


bt = Backtest(df, AI_test,
              cash=1000000, commission=.002,
              exclusive_orders=True, trade_on_close=True)

output = bt.run()
bt.plot()
print(output)

