import pandas as pd
import numpy as np
import openpyxl
import os
import datetime
from tqdm import tqdm
from backtesting import Backtest, Strategy
from sklearn.tree import DecisionTreeClassifier


#get stock excel file from path
dir_path = "../YFData/"
slist = list(map(lambda s: s.replace(".xlsx", ""), os.listdir(dir_path)))

for sno in tqdm(slist[343:344]):
    print(sno)
    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    train_data = pd.read_excel(dir_path+"/"+sno+".xlsx",index_col=0)
    train_data.drop(columns=["sno"], inplace=True)

    df = train_data.copy()
    
    train_data = train_data.loc[train_data.index<="2020-12-31"]
    train_data["Y"] = train_data["5DayResult"] > 0.05
    train_data_y = train_data.pop("Y")
    train_data.drop(columns=["5DayResult","VCP"], inplace=True)

    clf = DecisionTreeClassifier(max_depth=10).fit(train_data,train_data_y)

    pp = df.loc[df.index>"2021-12-31"]
    pp.drop(columns=["5DayResult","VCP"], inplace=True)

    pp["Prediction"] = [ float(i[1]) for i in clf.predict_proba(pp)]

    pp.to_excel("Data/"+sno+"_tree.xlsx")




class AI_test(Strategy):

    def init(self):
        return

    def next(self):
        if self.data.Prediction > 0.7:
            self.buy()
        if self.data.Prediction < 0.5 and self.position.is_long:
            self.position.close()

bt = Backtest(pp, AI_test,
              cash=1000000, commission=.001,
              exclusive_orders=True)

output = bt.run()
bt.plot()
print(output)

