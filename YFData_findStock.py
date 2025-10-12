import pandas as pd
import concurrent.futures as cf
import os
import time as t
from tqdm import tqdm
from datetime import datetime, timedelta


#get stock csv file from path
OUTPATH = "../SData/P_YFData/"
#EDATE = "2025-09-30"
EDATE = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")


def findStock(sno,stype,signal,days,ruleout):

    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    datadate = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    df = pd.read_csv(OUTPATH+"/"+stype+"/"+sno+".csv",index_col=0)
    df = df.loc[df.index>=datadate]        
    df = df.loc[df[signal.split('&')].all(axis=1)]    
    
    if ruleout!="":
        df = df.loc[df[ruleout.split('&')].all(axis=1)==False]
        
    df = df.reset_index()

    return df



def YFfindSignal(stype,signal,days=0,ruleout=""):
    signaldf = pd.DataFrame()

    snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(OUTPATH+"/"+stype+"/")))
    SLIST = pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST.assign(signal=signal+"")
    SLIST = SLIST.assign(ruleout=ruleout+"")
    SLIST = SLIST.assign(days=days)
    SLIST = SLIST[:]

    with cf.ProcessPoolExecutor(max_workers=12) as executor:
        for tempdf in tqdm(executor.map(findStock,SLIST["sno"],SLIST["stype"],SLIST["signal"],SLIST["days"],SLIST["ruleout"],chunksize=1),total=len(SLIST)):            
            tempdf = tempdf.dropna(axis=1, how="all")
            signaldf = pd.concat([tempdf, signaldf], ignore_index=True)

        if len(signaldf)!=0:
            signaldf["sno"] = signaldf["sno"].str.replace(r'^0+', '', regex=True)

        signaldf.to_csv("Data/"+stype+"_"+signal+"_"+datetime.now().strftime("%Y%m%d")+".csv",index=False)
        print("Finish")        



if __name__ == '__main__':

    start = t.perf_counter()
    
    YFfindSignal("L","T1_22&EMA2",2,"EMA1")
    YFfindSignal("M","T1_22&EMA2",2,"EMA1")
    YFfindSignal("S","T1_22&EMA2",2,"EMA1")    

    YFfindSignal("L","T1_50&EMA1",2)
    YFfindSignal("M","T1_50&EMA1",2)
    YFfindSignal("S","T1_50&EMA1",2)        

    YFfindSignal("L","T1_22&EMA1",2,"T1_50")
    YFfindSignal("M","T1_22&EMA1",2,"T1_50")
    YFfindSignal("S","T1_22&EMA1",2,"T1_50")

    YFfindSignal("HHLL","BOSS",30)
    
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    


