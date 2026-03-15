import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  

#get stock csv file from path
OUTPATH = "../Sdata/P_USdata/"
EDATE = (cc.datetime.now() - cc.timedelta(days=1)).strftime("%Y-%m-%d")
#EDATE = "2025-09-30"


def findStock(sno,stype,signal,ruleout):

    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    df = cc.pd.read_csv(OUTPATH+"/"+stype+"/"+sno+".csv",index_col=0)
    df = df.loc[df.index>=EDATE]        
    df = df.loc[df[signal.split('&')].all(axis=1)]    
    
    if ruleout!="":
        df = df.loc[df[ruleout.split('&')].all(axis=1)==False]
        
    df = df.reset_index()

    return df



def YFfindSignal(stype,signal,ruleout=""):
    signaldf = cc.pd.DataFrame()

    snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(OUTPATH+"/"+stype+"/")))
    SLIST = cc.pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST.assign(signal=signal+"")
    SLIST = SLIST.assign(ruleout=ruleout+"")
    SLIST = SLIST[:]

    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        for tempdf in cc.tqdm(executor.map(findStock,SLIST["sno"],SLIST["stype"],SLIST["signal"],SLIST["ruleout"],chunksize=1),total=len(SLIST)):            
            tempdf = tempdf.dropna(axis=1, how="all")
            signaldf = cc.pd.concat([tempdf, signaldf], ignore_index=True)
            
        signaldf["sno"] = signaldf["sno"].str.replace(r'^0+', '', regex=True)
        signaldf.to_csv("data/"+signal+"_"+stype+"_"+cc.datetime.now().strftime("%Y%m%d")+".csv",index=False)
        print("Finish")        



if __name__ == '__main__':

    start = cc.t.perf_counter()

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
    
    finish = cc.t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    


