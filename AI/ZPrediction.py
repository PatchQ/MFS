import UTIL.CommonConfig as cc
import joblib


       
def Prediction(modelname,model,df,sno,stype,tdate,fulldata):

    pp = df.loc[df.index>tdate].copy()
    
    if len(pp)>0:

        pp.drop(columns=["sno","F10D","F20D","F30D"], inplace=True)
        pp.drop(columns=["classification","BOSS_PATTERN","BOSS_STATUS","HHHL_PATTERN"], inplace=True)
        pp.drop(columns=["LLDate","HHDate","WLDate","WHDate","Volatility_Decrease"], inplace=True)
        #pp = pp.apply(pd.to_numeric, errors='coerce')
        pp = pp.replace([cc.np.inf, -cc.np.inf], cc.np.nan)
        
        proba = model.predict_proba(pp)

        if proba.shape[1] > 1:
            pp["Prediction"] = [float(i[1]) for i in proba]
        else:
            pp["Prediction"] = [float(i[0]) for i in proba]
                
        df[modelname] = pp["Prediction"]>0.9
        df[modelname] = df[modelname].fillna("")
        df.loc[df[modelname].astype(str).str.strip() == "", modelname] = False

        if fulldata:
            df.to_csv(f"{cc.OUTPATH}/{stype}/{sno}.csv")    
    else:
        df[modelname] = False


    return df
                
def loadModel(modelname, sno, df): 

    templist = []
    file_path = f"{cc.OUTPATH}/MODEL/{modelname}/P_{sno}.pkl"    

    if cc.os.path.exists(file_path):
        model = joblib.load(file_path)
        df = Prediction(modelname,model,df,sno,"X","1900-01-01",fulldata=False)

    if modelname in df.columns:
        templist = df.pop(modelname)
    
    return templist
