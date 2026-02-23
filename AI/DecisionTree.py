import pandas as pd
import numpy as np
import openpyxl
import os
import time as t
import datetime
from tqdm import tqdm
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import joblib


OUTPATH = "../SData/P_YFData/" 
#OUTPATH = "../SData/FP_YFData/"
stype = "M"
tdate = "2026-01-31"


def CalDTModel():

    snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(OUTPATH+"/"+stype)))
    snolist = snolist[:]

    resultdf = pd.DataFrame()

    for sno in tqdm(snolist):

        df = pd.read_csv(OUTPATH+"/"+stype+"/"+sno+".csv",index_col=0)      
        
        tempsno = str(sno).replace('P_','').replace('.HK','')
        tempsno = str(tempsno).lstrip("0")    

        print(tempsno)

        train_data = df.copy()
        train_data = train_data.loc[train_data.index<=tdate]    
        #train_data = train_data.apply(pd.to_numeric, errors='coerce')
        
        if len(train_data)>500:

            train_data["Y"] = train_data["F10D"] > 0.10
            train_data_y = train_data.pop("Y")

            train_data["sno"] = tempsno
            train_data.drop(columns=["F10D","F20D","F30D","DT"], inplace=True)
            train_data.drop(columns=["classification","BOSS_PATTERN","BOSS_STATUS","HHHL_PATTERN"], inplace=True)
            train_data.drop(columns=["LLDate","HHDate","WLDate","WHDate","Volatility_Decrease"], inplace=True)

            xtrain, xtest, ytrain, ytest = train_test_split(train_data, train_data_y, test_size=0.2, random_state=1)
            
            # 1. 创建一个填补器（例如：用均值填补，你也可以用 'median' 或 'most_frequent'）
            #imputer = SimpleImputer(strategy='mean',keep_empty_features=True) # 或用 'median', 'most_frequent'

            # 2. 将填补器和分类器组合成一个管道
            #model = make_pipeline(imputer, DecisionTreeClassifier(max_depth=10,random_state=1))

            # 3. 直接用管道进行训练（它会先自动填补，再训练）
            #model.fit(xtrain, ytrain)                        
            model = DecisionTreeClassifier(max_depth=10,random_state=1).fit(xtrain,ytrain)

            # 4. save model
            joblib.dump(model, f"{OUTPATH}/MODEL/{sno}_DT.pkl")

            #model2 = joblib.load(f"{OUTPATH}/MODEL/{sno}_DT.pkl")
            pred = model.predict(xtest)
            accuracy = accuracy_score(ytest, pred)

            clf_report = metrics.classification_report(ytest, pred)
            conf_mat = confusion_matrix(ytest, pred)

            #print("accuracy:" +str(accuracy))
            #print(clf_report)
            
            pp = df.loc[df.index>tdate].copy()    

            if len(pp)>0:
                pp["sno"] = tempsno        
                pp.drop(columns=["F10D","F20D","F30D","DT"], inplace=True)
                pp.drop(columns=["classification","BOSS_PATTERN","BOSS_STATUS","HHHL_PATTERN"], inplace=True)
                pp.drop(columns=["LLDate","HHDate","WLDate","WHDate","Volatility_Decrease"], inplace=True)
                #pp = pp.apply(pd.to_numeric, errors='coerce')

                #print(model.predict_proba(pp))

                pp["Prediction"] = [float(i[1]) for i in model.predict_proba(pp)]
                        
                df["DT"] = pp["Prediction"]>0.9
                df["DT"] = df["DT"].fillna("")
                df.loc[df["DT"].astype(str).str.strip() == "", "DT"] = False
                df.to_csv(f"{OUTPATH}/{stype}/{sno}.csv")

                tempdf = df.loc[df["DT"]]

                tempdf.insert(0, 'Date', pd.to_datetime(tempdf.index))                
                resultdf = pd.concat([resultdf, tempdf], ignore_index=True)

    resultdf.to_csv("Data/DecisionTree.csv")
    print(resultdf)

def calDT(sno, df):

    file_path = f"{OUTPATH}/MODEL/P_{sno}_DT.pkl"
    
    if os.path.exists(file_path):

        pp = df.copy()

        model = joblib.load(file_path)

        tempsno = str(sno).replace('.HK','')
        tempsno = str(tempsno).lstrip("0") 

        #print(tempsno)

        pp["sno"] = tempsno
        pp.drop(columns=["F10D","F20D","F30D","DT"], inplace=True)
        pp.drop(columns=["classification","BOSS_PATTERN","BOSS_STATUS","HHHL_PATTERN"], inplace=True)
        pp.drop(columns=["LLDate","HHDate","WLDate","WHDate","Volatility_Decrease"], inplace=True)
        
        pp["Prediction"] = [float(i[1]) for i in model.predict_proba(pp)]
                
        df["DT"] = pp["Prediction"]>0.9
        df["DT"] = df["DT"].fillna("")
        df.loc[df["DT"].astype(str).str.strip() == "", "DT"] = False
    
    return df


if __name__ == '__main__':
    start = t.perf_counter()

    #calDT(sno, df)

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    