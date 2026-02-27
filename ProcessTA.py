import os
import pandas as pd
import time as t
import concurrent.futures as cf
import platform
from tqdm import tqdm

from TA.LW_CalHHLL import *
from TA.LW_Calindicator import *
from TA.LW_CheckWave import *
from TA.LW_CheckBoss import *
from TA.LW_CheckT1 import *
from TA.LW_CheckVCP import *

from AI.DecisionTree import *
from AI.XGBoost import *
from AI.LightGBM import *
from AI.LogisticRegression import *
from AI.MLP import *
from AI.RandomForest import *
from AI.SVM import *
from AI.VOTING import *


PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/"

# PATH = "../SData/FYFData/"
# OUTPATH = "../SData/FP_YFData/" 

def AnalyzeStock(sno,stype):

    df = pd.read_csv(PATH+"/"+stype+"/"+sno+".csv",index_col=0) 
    df.index = pd.to_datetime(df.index)  

    df = extendData(df)
    df = convertData(df)

    #EMA
    df = calEMA(df)

    #T1
    df = checkT1(df,50)

    #cal HHHL
    HHLLdf = calHHLL(df)

    if HHLLdf is not None:
        if len(HHLLdf)>0:        
            #Boss
            df = checkBoss(df, sno, stype, HHLLdf)
            #HHHL
            df = checkWave(df, sno, stype, HHLLdf)                     
    
    #VCP
    df = checkVCP(df)





    #AI Signal
    #1. XGBoost   
    xgblist = loadXGB(sno, df)    

    #2. DecisionTree
    dtlist = loadDT(sno, df)

    #3. RandomForest
    rflist = loadRF(sno, df)

    #4. LogisticRegression
    lrlist = loadLR(sno, df)

    #5. MLP
    mlplist = loadMLP(sno, df)

    #6. VOTING
    votinglist = loadVOTING(sno, df)

    #7. LightGBM
    lgbmlist = loadLGBM(sno, df)

    #8. SVM
    svmlist = loadSVM(sno, df)


    if len(xgblist)>0:
        df["XGB"] = xgblist
    else:
        df["XGB"] = False

    if len(dtlist)>0:
        df["DT"] = dtlist
    else:
        df["DT"] = False

    if len(rflist)>0:
        df["RF"] = rflist
    else:
        df["RF"] = False

    if len(lrlist)>0:
        df["LR"] = lrlist
    else:
        df["LR"] = False

    if len(mlplist)>0:
        df["MLP"] = mlplist
    else:
        df["MLP"] = False

    if len(votinglist)>0:
        df["VOTING"] = votinglist
    else:
        df["VOTING"] = False                    

    if len(lgbmlist)>0:
        df["LGBM"] = lgbmlist
    else:
        df["LGBM"] = False                                

    if len(svmlist)>0:
        df["SVM"] = svmlist
    else:
        df["SVM"] = False                                

    df = df.reset_index()
    #df = df.sort_values(by=['index'],ascending=[True])
    df = df[:-10].copy()
    df.to_csv(OUTPATH+"/"+stype+"/P_"+sno+".csv",index=False)
    


def ProcessTA(stype):

    snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(PATH+"/"+stype)))
    SLIST = pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST[:]

    #print(SLIST)

    if platform.system()=="Windows":
        executor = cf.ProcessPoolExecutor(max_workers=5)
    elif platform.system()=="Darwin":
        executor = cf.ThreadPoolExecutor(max_workers=1)

    with executor:
        list(tqdm(executor.map(AnalyzeStock,SLIST["sno"],SLIST["stype"],chunksize=1),total=len(SLIST)))


if __name__ == '__main__':
    start = t.perf_counter()

    ProcessTA("L")    
    ProcessTA("M")    

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    