import UTIL.CommonConfig as cc

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import joblib


def DT(sno,stype,tdate):

    thismodel = cc.sys._getframe().f_code.co_name.upper()

    tempdf = cc.pd.DataFrame()

    df = cc.pd.read_csv(cc.OUTPATH+"/"+stype+"/"+sno+".csv",index_col=0)
    
    tempsno = str(sno).replace('P_','').replace('.HK','')
    tempsno = str(tempsno).lstrip("0")    

    #print(tempsno)
    for modelname in cc.MODELLIST:
        if modelname in df.columns:
            df.drop(columns=[modelname], inplace=True)

    train_data = df.copy()
    train_data = train_data.loc[train_data.index<=tdate]    
    #train_data = train_data.apply(pd.to_numeric, errors='coerce')
    
    if len(train_data)>500:

        train_data["Y"] = train_data["F20D"] > 0.15        
        train_data_y = train_data.pop("Y")
        #print(train_data_y.value_counts())

        train_data.drop(columns=["sno","F10D","F20D","F30D"], inplace=True)
        train_data.drop(columns=["classification","BOSS_PATTERN","BOSS_STATUS","HHHL_PATTERN"], inplace=True)
        train_data.drop(columns=["LLDate","HHDate","WLDate","WHDate","Volatility_Decrease"], inplace=True)

        train_data = train_data.replace([cc.np.inf, -cc.np.inf], cc.np.nan)

        xtrain, xtest, ytrain, ytest = train_test_split(train_data, train_data_y, test_size=0.2, random_state=1, stratify=train_data_y)
        
        # create Imputer with mean/median/most_frequent
        imputer = SimpleImputer(strategy='mean',keep_empty_features=True)
        model = make_pipeline(imputer, DecisionTreeClassifier(max_depth=10,random_state=1))
        
        model.fit(xtrain, ytrain)                        

        #pred = model.predict(xtest)
        #accuracy = accuracy_score(ytest, pred)

        #train_score = model.score(xtrain, ytrain)
        #test_score = model.score(xtest, ytest)

        #print(f"訓練分數:{train_score}  測試分數:{test_score}")

        #clf_report = metrics.classification_report(ytest, pred)
        #conf_mat = confusion_matrix(ytest, pred)

        #print("accuracy:" +str(accuracy))
        #print(clf_report)
        
        # save model
        if cc.PROD:
            joblib.dump(model, f"{cc.OUTPATH}/MODEL/{thismodel}/{sno}.pkl")
        
        tempdf = cc.zp.Prediction(thismodel,model,df,sno,stype,tdate,fulldata=True)
        
        tempdf = tempdf.loc[tempdf[thismodel]]
        tempdf.insert(0, 'Date', cc.pd.to_datetime(tempdf.index))   

    return tempdf

        
