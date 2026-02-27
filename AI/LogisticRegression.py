import pandas as pd
import numpy as np
import os
import platform
import time as t
import concurrent.futures as cf
import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

PROD = True
#OUTPATH = "../SData/FP_YFData/"
OUTPATH = "../SData/P_YFData/"
MODEL = "LR"
MODELLIST = ["DT","XGB","LGBM","LR","MLP","RF","SVM","VOTING"]

def LR(sno, stype, tdate):
    
    tempdf = pd.DataFrame()
    df = pd.read_csv(f"{OUTPATH}/{stype}/{sno}.csv", index_col=0)
    
    # 清理可能存在的舊預測欄位
    for modelname in MODELLIST:
        if modelname in df.columns:
            df.drop(columns=[modelname], inplace=True)
    
    train_data = df.copy()
    train_data = train_data.loc[train_data.index <= tdate]
    
    if len(train_data) > 500:
        # 建立目標變數
        train_data["Y"] = train_data["F20D"] > 0.15
        train_data_y = train_data.pop("Y")
        
        # 刪除不需要的特徵
        drop_cols = ["sno", "F10D", "F20D", "F30D", "classification",
                     "BOSS_PATTERN", "BOSS_STATUS", "HHHL_PATTERN",
                     "LLDate", "HHDate", "WLDate", "WHDate", "Volatility_Decrease"]
        train_data.drop(columns=[c for c in drop_cols if c in train_data.columns], inplace=True)
        
        # 處理無窮大
        train_data = train_data.replace([np.inf, -np.inf], np.nan)
        
        # 分割訓練/測試集 (保留 20% 作為測試，但此處未使用，僅為與原格式一致)
        xtrain, xtest, ytrain, ytest = train_test_split(
            train_data, train_data_y, test_size=0.2, random_state=1, stratify=train_data_y
        )

        # 若訓練集只有一個類別，則跳過 
        if len(ytrain.unique()) < 2:
            return pd.DataFrame() 
        
        # 建立 pipeline: 填補缺失值 + 標準化 + 邏輯回歸 (處理類別不平衡)
        imputer = SimpleImputer(strategy='mean', keep_empty_features=True)
        scaler = StandardScaler()
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=1)
        
        model = make_pipeline(imputer, scaler, lr)
        model.fit(xtrain, ytrain)
        
        # 儲存模型 (若為生產環境)
        if PROD:
            joblib.dump(model, f"{OUTPATH}/MODEL/{MODEL}/{sno}.pkl")
        
        # 呼叫預測函數
        tempdf = Prediction(model, df, sno, stype, tdate, fulldata=True)
        if len(tempdf) > 0:
            tempdf = tempdf.loc[tempdf[MODEL]]
            tempdf.insert(0, 'Date', pd.to_datetime(tempdf.index))
    
    return tempdf
           

def Prediction(model,df,sno,stype,tdate,fulldata):

    pp = df.loc[df.index>tdate].copy()
    df[MODEL] = False

    if len(pp)>0:
         
        pp.drop(columns=["sno","F10D","F20D","F30D"], inplace=True)
        pp.drop(columns=["classification","BOSS_PATTERN","BOSS_STATUS","HHHL_PATTERN"], inplace=True)
        pp.drop(columns=["LLDate","HHDate","WLDate","WHDate","Volatility_Decrease"], inplace=True)
        #pp = pp.apply(pd.to_numeric, errors='coerce')

        pp = pp.replace([np.inf, -np.inf], np.nan)
        
        proba = model.predict_proba(pp)

        if proba.shape[1] > 1:
            pp["Prediction"] = [float(i[1]) for i in proba]
        else:
            pp["Prediction"] = [float(i[0]) for i in proba]


        df[MODEL] = pp["Prediction"]>0.9
        df[MODEL] = df[MODEL].fillna("")
        df.loc[df[MODEL].astype(str).str.strip() == "", MODEL] = False

    if fulldata:        
        df.to_csv(f"{OUTPATH}/{stype}/{sno}.csv")        

    return df            

def loadLR(sno, df):
    
    templist = []
    file_path = f"{OUTPATH}/MODEL/{MODEL}/P_{sno}.pkl"    

    if os.path.exists(file_path):
        model = joblib.load(file_path)
        df = Prediction(model,df,sno,"X","1900-01-01",fulldata=False)

    if MODEL in df.columns:
        templist = df.pop(MODEL)
    
    return templist
    

def Process(stype,tdate):

    resultdf = pd.DataFrame()

    snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(OUTPATH+"/"+stype)))
    SLIST = pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST.assign(tdate=tdate+"")
    SLIST = SLIST[:]

    #print(SLIST)

    if platform.system()=="Windows":
        executor = cf.ProcessPoolExecutor(max_workers=5)
    elif platform.system()=="Darwin":
        executor = cf.ThreadPoolExecutor(max_workers=1)
    
    with executor:
        for tempdf in tqdm(executor.map(LR,SLIST["sno"],SLIST["stype"],SLIST["tdate"],chunksize=1),total=len(SLIST)):                        
            resultdf = pd.concat([tempdf, resultdf], ignore_index=True)

    resultdf.to_csv(f"Data/{MODEL}.csv",index=False)    




if __name__ == '__main__':
    start = t.perf_counter()

    tdate = "2019-12-31"

    #Process("L",tdate)
    Process("M",tdate)

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    