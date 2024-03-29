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
dir_path = "../SData/YFData"
slist = list(map(lambda s: s.replace(".xlsx", ""), os.listdir(dir_path)))
slist = slist[343:344]
slist = ["0700.HK"]

for sno in tqdm(slist):
    print(sno)
    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    train_data = pd.read_excel(dir_path+"/"+sno+".xlsx",index_col=0)
    train_data.drop(columns=["sno"], inplace=True)

    df = train_data.copy()
    
    train_data = train_data.loc[train_data.index<="2020-12-31"]
    train_data["Y"] = train_data["5DayResult"] > 0.05

    train_data_y = train_data.pop("Y")
    train_data.drop(columns=["10DayChange","5DayResult","VCP"], inplace=True)

    train_data.to_excel("Data/"+sno+"_tree.xlsx")

    xtrain, xtest, ytrain, ytest = train_test_split(train_data, train_data_y, test_size=0.2, random_state=1)
    model = DecisionTreeClassifier(max_depth=14).fit(xtrain,ytrain)
    pred = model.predict(xtest)
    accuracy = accuracy_score(ytest, pred)

    clf_report = metrics.classification_report(ytest, pred)
    conf_mat = confusion_matrix(ytest, pred)

    print("accuracy:" +str(accuracy))
    print(clf_report)
    
    pp = df.loc[df.index>"2020-12-31"]
    pp.drop(columns=["10DayChange","5DayResult","VCP"], inplace=True)

    print(model.predict_proba(pp))

    pp["Prediction"] = [ float(i[1]) for i in model.predict_proba(pp)]

    pp.to_excel("Data/"+sno+"_tree.xlsx")

    print(pp)



class AI_test(Strategy):
    sl_ratio = 99     # stop loss ratio, 99 means 1% loss

    def init(self):
        return

    def next(self):
        if self.data.Prediction > 0.7:
            self.buy(size=.99,sl=self.data.Close[-1]*self.sl_ratio/100)
        if self.data.Prediction < 0.5 and self.position.is_long:
            self.position.close()

bt = Backtest(pp, AI_test,
              cash=1000000, commission=.002,
              exclusive_orders=True)

output = bt.run()
bt.plot()
print(output)

