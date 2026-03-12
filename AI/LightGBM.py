import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')



def LGBM(sno, stype, tdate):
    
    thismodel = cc.sys._getframe().f_code.co_name.upper()
    
    tempdf = cc.pd.DataFrame()
    df = cc.pd.read_csv(f"{cc.OUTPATH}/{stype}/{sno}.csv", index_col=0)
    
    for modelname in cc.MODELLIST:
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
        train_data = train_data.replace([cc.np.inf, -cc.np.inf], cc.np.nan)
        
        xtrain, xtest, ytrain, ytest = train_test_split(
            train_data, train_data_y, test_size=0.2, random_state=1, stratify=train_data_y
        )
        
        # 計算不平衡權重
        pos_count = ytrain.sum()
        neg_count = len(ytrain) - pos_count
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        
        imputer = SimpleImputer(strategy='mean', keep_empty_features=True)
        lgbm = LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, random_state=1, verbose=-1
        )
        
        model = make_pipeline(imputer, lgbm)
        model.fit(xtrain, ytrain)
        
        if cc.PROD:
            joblib.dump(model, f"{cc.OUTPATH}/MODEL/{thismodel}/{sno}.pkl")
        
        tempdf = cc.zp.Prediction(thismodel, model, df, sno, stype, tdate, fulldata=True)
        if len(tempdf) > 0:
            tempdf = tempdf.loc[tempdf[thismodel]]
            tempdf.insert(0, 'Date', cc.pd.to_datetime(tempdf.index))
    
    return tempdf
           