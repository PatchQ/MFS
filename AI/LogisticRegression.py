import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

def LR(sno, stype, tdate):
    
    thismodel = cc.sys._getframe().f_code.co_name.upper()

    tempdf = cc.pd.DataFrame()

    df = cc.pd.read_csv(cc.OUTPATH+"/"+stype+"/"+sno+".csv",index_col=0)    

    # 清理可能存在的舊預測欄位
    for modelname in cc.MODELLIST:
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
                     "LLDate", "HHDate", "WLDate", "WHDate"]
        
        train_data.drop(columns=[c for c in drop_cols if c in train_data.columns], inplace=True)                
        train_data = train_data.replace([cc.np.inf, -cc.np.inf], cc.np.nan)

        
        # 分割訓練/測試集 (保留 20% 作為測試，但此處未使用，僅為與原格式一致)
        xtrain, xtest, ytrain, ytest = train_test_split(
            train_data, train_data_y, test_size=0.2, random_state=1, stratify=train_data_y
        )

        # 若訓練集只有一個類別，則跳過 
        if len(ytrain.unique()) < 2:
            return cc.pd.DataFrame() 
        
        # 建立 pipeline: 填補缺失值 + 標準化 + 邏輯回歸 (處理類別不平衡)
        imputer = SimpleImputer(strategy='mean', keep_empty_features=True)
        scaler = StandardScaler()
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=1)
        
        model = make_pipeline(imputer, scaler, lr)
        model.fit(xtrain, ytrain)
        
        # 儲存模型 (若為生產環境)
        if cc.PROD:
            joblib.dump(model, f"{cc.OUTPATH}/MODEL/{thismodel}/{sno}.pkl")
        
        # 呼叫預測函數
        tempdf = cc.Prediction(thismodel, model, df, sno, stype, tdate, fulldata=True)
        if len(tempdf) > 0:
            tempdf = tempdf.loc[tempdf[thismodel]]
            tempdf.insert(0, 'Date', cc.pd.to_datetime(tempdf.index))
    
    return tempdf
           