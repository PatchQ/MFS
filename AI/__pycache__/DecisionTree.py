import pandas as pd
import numpy as np
import os
import platform
import time as t
import concurrent.futures as cf
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


#OUTPATH = "../SData/P_YFData/" 
OUTPATH = "../SData/FP_YFData/"


def CalDTModel(sno,stype,tdate):

    tempdf = pd.DataFrame()
    
    df = pd.read_csv(OUTPATH+"/"+stype+"/"+sno+".csv",index_col=0)      
    
    tempsno = str(sno).replace('P_','').replace('.HK','')
    tempsno = str(tempsno).lstrip("0")    

    #print(tempsno)

    train_data = df.copy()
    train_data = train_data.loc[train_data.index<=tdate]    
    #train_data = train_data.apply(pd.to_numeric, errors='coerce')
    
    if len(train_data)>500:

        train_data["Y"] = train_data["F20D"] > 0.15
        train_data_y = train_data.pop("Y")
        #print(train_data_y.value_counts())

        #train_data["sno"] = tempsno
        train_data.drop(columns=["sno","F10D","F20D","F30D","DT"], inplace=True)
        train_data.drop(columns=["classification","BOSS_PATTERN","BOSS_STATUS","HHHL_PATTERN"], inplace=True)
        train_data.drop(columns=["LLDate","HHDate","WLDate","WHDate","Volatility_Decrease"], inplace=True)

        xtrain, xtest, ytrain, ytest = train_test_split(train_data, train_data_y, test_size=0.2, random_state=1, stratify=train_data_y)
        
         # 1. 创建一个填补器（例如：用均值填补，你也可以用 'median' 或 'most_frequent'）
        imputer = SimpleImputer(strategy='mean',keep_empty_features=True) # 或用 'median', 'most_frequent'

        # 2. 将填补器和分类器组合成一个管道
        model = make_pipeline(imputer, DecisionTreeClassifier(max_depth=10,random_state=1))

        # 3. 直接用管道进行训练（它会先自动填补，再训练）
        model.fit(xtrain, ytrain)                        
        
        # 4. save model
        joblib.dump(model, f"{OUTPATH}/MODEL/{sno}_DT.pkl")
        
        pred = model.predict(xtest)
        accuracy = accuracy_score(ytest, pred)

        train_score = model.score(xtrain, ytrain)
        test_score = model.score(xtest, ytest)

        print(f"訓練分數:{train_score}  測試分數:{test_score}")

        #clf_report = metrics.classification_report(ytest, pred)
        #conf_mat = confusion_matrix(ytest, pred)

        #print("accuracy:" +str(accuracy))
        #print(clf_report)
        
        pp = df.loc[df.index>tdate].copy()    

        if len(pp)>0:
            #pp["sno"] = tempsno        
            pp.drop(columns=["sno","F10D","F20D","F30D","DT"], inplace=True)
            pp.drop(columns=["classification","BOSS_PATTERN","BOSS_STATUS","HHHL_PATTERN"], inplace=True)
            pp.drop(columns=["LLDate","HHDate","WLDate","WHDate","Volatility_Decrease"], inplace=True)
            #pp = pp.apply(pd.to_numeric, errors='coerce')

            #print(model.predict_proba(pp))
            proba = model.predict_proba(pp)
            #print(proba)

            if proba.shape[1] > 1:
                pp["Prediction"] = [float(i[1]) for i in proba]
            else:
                pp["Prediction"] = [float(i[0]) for i in proba]
                    
            df["DT"] = pp["Prediction"]>0.9
            df["DT"] = df["DT"].fillna("")
            df.loc[df["DT"].astype(str).str.strip() == "", "DT"] = False
            df.to_csv(f"{OUTPATH}/{stype}/{sno}.csv")

            tempdf = df.loc[df["DT"]]
            tempdf.insert(0, 'Date', pd.to_datetime(tempdf.index))                

    return tempdf
            

    
def calDT(sno, df):

    file_path = f"{OUTPATH}/MODEL/P_{sno}_DT.pkl"
    
    if os.path.exists(file_path):

        pp = df.copy()

        model = joblib.load(file_path)

        tempsno = str(sno).replace('.HK','')
        tempsno = str(tempsno).lstrip("0") 

        #print(tempsno)

        #pp["sno"] = tempsno
        pp.drop(columns=["sno","F10D","F20D","F30D","DT"], inplace=True)
        pp.drop(columns=["classification","BOSS_PATTERN","BOSS_STATUS","HHHL_PATTERN"], inplace=True)
        pp.drop(columns=["LLDate","HHDate","WLDate","WHDate","Volatility_Decrease"], inplace=True)
        pp = pp.replace([np.inf, -np.inf], np.nan)

        proba = model.predict_proba(pp)
        #print(proba)

        if proba.shape[1] > 1:
            pp["Prediction"] = [float(i[1]) for i in proba]
        else:
            pp["Prediction"] = [float(i[0]) for i in proba]        
                
        df["DT"] = pp["Prediction"]>0.9
        df["DT"] = df["DT"].fillna("")
        df.loc[df["DT"].astype(str).str.strip() == "", "DT"] = False
    
    return df

