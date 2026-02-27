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
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

PROD = True
#OUTPATH = "../SData/FP_YFData/"
OUTPATH = "../SData/P_YFData/"
MODEL = "VOTING"
MODELLIST = ["DT","XGB","LGBM","LR","MLP","RF","SVM","VOTING"]

def VOTING(sno, stype, tdate):
    
    tempdf = pd.DataFrame()
    df = pd.read_csv(f"{OUTPATH}/{stype}/{sno}.csv", index_col=0)
    
    for modelname in MODELLIST:
        if modelname in df.columns:
            df.drop(columns=[modelname], inplace=True)
    
    train_data = df.copy()
    train_data = train_data.loc[train_data.index <= tdate]
    
    if len(train_data) > 500:
        train_data["Y"] = train_data["F20D"] > 0.15
        train_data_y = train_data.pop("Y")
        
        drop_cols = ["sno", "F10D", "F20D", "F30D", "classification",
                     "BOSS_PATTERN", "BOSS_STATUS", "HHHL_PATTERN",
                     "LLDate", "HHDate", "WLDate", "WHDate", "Volatility_Decrease"]
        train_data.drop(columns=[c for c in drop_cols if c in train_data.columns], inplace=True)
        train_data = train_data.replace([np.inf, -np.inf], np.nan)
        
        xtrain, xtest, ytrain, ytest = train_test_split(
            train_data, train_data_y, test_size=0.2, random_state=1, stratify=train_data_y
        )

        # 若訓練集只有一個類別，則跳過 
        if len(ytrain.unique()) < 2:
            return pd.DataFrame() 
        
        # 建立三個基礎模型，各自包含預處理 pipeline
        # 邏輯回歸 (需縮放)
        lr_pipe = make_pipeline(
            SimpleImputer(strategy='mean', keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(class_weight='balanced', max_iter=1000, random_state=1)
        )
        
        # 隨機森林 (不需縮放)
        rf_pipe = make_pipeline(
            SimpleImputer(strategy='mean', keep_empty_features=True),
            RandomForestClassifier(n_estimators=100, max_depth=10,
                                   class_weight='balanced', random_state=1, n_jobs=-1)
        )
        
        # LightGBM (使用 scale_pos_weight 處理不平衡)
        pos_count = ytrain.sum()
        neg_count = len(ytrain) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        
        lgbm_pipe = make_pipeline(
            SimpleImputer(strategy='mean', keep_empty_features=True),
            LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                           scale_pos_weight=scale_pos_weight, random_state=1, verbose=-1)
        )
        
        # 建立軟投票集成 (使用預測機率加權)
        voting = VotingClassifier(
            estimators=[
                ('lr', lr_pipe),
                ('rf', rf_pipe),
                ('lgbm', lgbm_pipe)
            ],
            voting='soft',   # 使用機率投票
            weights=[1, 1, 1]  # 可依表現調整權重
        )
        
        # 注意：VotingClassifier 本身不是 pipeline，但可直接 fit
        voting.fit(xtrain, ytrain)
        model = voting  # 後續預測用此 model
        
        if PROD:
            joblib.dump(model, f"{OUTPATH}/MODEL/{MODEL}/{sno}.pkl")
        
        # 呼叫預測函數 (注意：Prediction 中使用的 model 須有 predict_proba)
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

def loadVOTING(sno, df):
    
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
        for tempdf in tqdm(executor.map(VOTING,SLIST["sno"],SLIST["stype"],SLIST["tdate"],chunksize=1),total=len(SLIST)):                        
            resultdf = pd.concat([tempdf, resultdf], ignore_index=True)

    resultdf.to_csv(f"Data/{MODEL}.csv",index=False)    




if __name__ == '__main__':
    start = t.perf_counter()

    tdate = "2019-12-31"

    #Process("L",tdate)
    Process("M",tdate)

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    