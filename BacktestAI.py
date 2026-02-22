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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


OUTPATH = "../SData/P_YFData/" 
#OUTPATH = "../SData/FP_YFData/"
stype = "L"
tdate = "2020-12-31"

snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(OUTPATH+"/"+stype)))
snolist = snolist[:]

resultdf = pd.DataFrame()

for sno in tqdm(snolist):
    
    tempsno = str(sno).replace('P_','').replace('.HK','')
    tempsno = str(tempsno).lstrip("0")

    df = pd.read_csv(OUTPATH+"/"+stype+"/"+sno+".csv",index_col=0)        
    #df.drop(columns=["DT"], inplace=True)

    train_data = df.copy()
    
    train_data["sno"] = tempsno
    train_data = train_data.apply(pd.to_numeric, errors='coerce')
    train_data = train_data.loc[train_data.index<=tdate]

    if len(train_data)>500:

        train_data["Y"] = train_data["F10D"] > 0.10

        train_data_y = train_data.pop("Y")
        train_data.drop(columns=["F10D","F20D","F30D","DT"], inplace=True)
        xtrain, xtest, ytrain, ytest = train_test_split(train_data, train_data_y, test_size=0.2, random_state=1)
        
        # 1. 创建一个填补器（例如：用均值填补，你也可以用 'median' 或 'most_frequent'）
        imputer = SimpleImputer(strategy='mean',keep_empty_features=True) # 或用 'median', 'most_frequent'

        # 2. 将填补器和分类器组合成一个管道
        model = make_pipeline(imputer, DecisionTreeClassifier(max_depth=10,random_state=1))

        # 3. 直接用管道进行训练（它会先自动填补，再训练）
        model.fit(xtrain, ytrain)                        
        #model = DecisionTreeClassifier(max_depth=14).fit(xtrain,ytrain)
        
        pred = model.predict(xtest)
        accuracy = accuracy_score(ytest, pred)

        clf_report = metrics.classification_report(ytest, pred)
        conf_mat = confusion_matrix(ytest, pred)

        #print("accuracy:" +str(accuracy))
        #print(clf_report)
        
        pp = df.loc[df.index>tdate].copy()    
        pp.drop(columns=["F10D","F20D","F30D","DT"], inplace=True)
        pp = pp.apply(pd.to_numeric, errors='coerce')

        #print(model.predict_proba(pp))

        pp["Prediction"] = [float(i[1]) for i in model.predict_proba(pp)]
                
        df["DT"] = pp["Prediction"]>0.9
        df["DT"] = df["DT"].fillna("")
        df.loc[df["DT"].astype(str).str.strip() == "", "DT"] = False
        df.to_csv(f"{OUTPATH}/{stype}/{sno}.csv")

        tempdf = df.loc[df["DT"]]

        #tempdf.insert(0, 'Date', pd.to_datetime(tempdf.index))                

        resultdf = pd.concat([resultdf, tempdf], ignore_index=True)

resultdf.to_csv("Data/DecisionTree.csv")
print(resultdf)

    



# class AI_test(Strategy):
#     sl_ratio = 99     # stop loss ratio, 99 means 1% loss

#     def init(self):
#         return

#     def next(self):
#         if self.data.Prediction > 0.7:
#             self.buy(size=.99,sl=self.data.Close[-1]*self.sl_ratio/100)
#         if self.data.Prediction < 0.5 and self.position.is_long:
#             self.position.close()

# bt = Backtest(pp, AI_test,
#               cash=1000000, commission=.002,
#               exclusive_orders=True)

# output = bt.run()
# bt.plot()
# print(output)

