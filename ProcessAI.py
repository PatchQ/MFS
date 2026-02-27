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

#OUTPATH = "../SData/P_YFData/"
OUTPATH = "../SData/FP_YFData/"

def dispatch(sno, model, stype, tdate):
    if model == 'XGB':
        return XGB(sno, stype, tdate)
    elif model == 'DT':
        return DT(sno, stype, tdate)


def ProcessAI(stype,model,tdate):

   resultdf = pd.DataFrame()

   snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(OUTPATH+"/"+stype)))
   SLIST = pd.DataFrame(snolist, columns=["sno"])
   SLIST = SLIST.assign(model=model+"")
   SLIST = SLIST.assign(stype=stype+"")
   SLIST = SLIST.assign(tdate=tdate+"")
   SLIST = SLIST[:]

    #print(SLIST)

   if platform.system()=="Windows":
       executor = cf.ProcessPoolExecutor(max_workers=5)
   elif platform.system()=="Darwin":
       executor = cf.ThreadPoolExecutor(max_workers=1)

   with executor:
       for tempdf in tqdm(executor.map(dispatch,SLIST["sno"],SLIST["model"],SLIST["stype"],SLIST["tdate"],chunksize=1),total=len(SLIST)):           
           resultdf = pd.concat([tempdf, resultdf], ignore_index=True)

   resultdf.to_csv(f"Data/{stype}_{model}.csv",index=False)    


if __name__ == '__main__':
    start = t.perf_counter()

    tdate = "2019-12-31"

    ProcessAI("L","XGB",tdate)
    ProcessAI("M","XGB",tdate)

    ProcessAI("L","DT",tdate)
    ProcessAI("M","DT",tdate)

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    