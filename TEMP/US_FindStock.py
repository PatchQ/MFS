import pandas as pd
import concurrent.futures as cf
import os
import time as t
from tqdm import tqdm
from datetime import datetime, timedelta


#get stock csv file from path
OUTPATH = "../SData/P_USData/"
EDATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
#EDATE = "2025-09-30"


def findStock(sno,stype,signal,ruleout):

    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    df = pd.read_csv(OUTPATH+"/"+stype+"/"+sno+".csv",index_col=0)
    df = df.loc[df.index>=EDATE]        
    df = df.loc[df[signal.split('&')].all(axis=1)]    
    
    if ruleout!="":
        df = df.loc[df[ruleout.split('&')].all(axis=1)==False]
        
    df = df.reset_index()

    return df



def YFfindSignal(stype,signal,ruleout=""):
    signaldf = pd.DataFrame()

    snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(OUTPATH+"/"+stype+"/")))
    SLIST = pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST.assign(signal=signal+"")
    SLIST = SLIST.assign(ruleout=ruleout+"")
    SLIST = SLIST[:]

    with cf.ProcessPoolExecutor(max_workers=5) as executor:
        for tempdf in tqdm(executor.map(findStock,SLIST["sno"],SLIST["stype"],SLIST["signal"],SLIST["ruleout"],chunksize=1),total=len(SLIST)):            
            tempdf = tempdf.dropna(axis=1, how="all")
            signaldf = pd.concat([tempdf, signaldf], ignore_index=True)
            
        signaldf["sno"] = signaldf["sno"].str.replace(r'^0+', '', regex=True)
        signaldf.to_csv("Data/"+signal+"_"+stype+"_"+datetime.now().strftime("%Y%m%d")+".csv",index=False)
        print("Finish")        



if __name__ == '__main__':

    start = t.perf_counter()

    YFfindSignal("XASE","T1_22&EMA1")
    YFfindSignal("XNGS","T1_22&EMA1")
    YFfindSignal("XNMS","T1_22&EMA1")
    YFfindSignal("XNCM","T1_22&EMA1")
    YFfindSignal("XNYS","T1_22&EMA1")

    YFfindSignal("XASE","T1_10&EMA2","EMA1")
    YFfindSignal("XNGS","T1_10&EMA2","EMA1")
    YFfindSignal("XNMS","T1_10&EMA2","EMA1")
    YFfindSignal("XNCM","T1_10&EMA2","EMA1")
    YFfindSignal("XNYS","T1_10&EMA2","EMA1")
    
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    


