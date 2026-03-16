import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  

#EDATE = "2025-09-30"
EDATE = (cc.datetime.now() - cc.timedelta(days=30)).strftime("%Y-%m-%d")
FILESTAMP = "_"+cc.datetime.now().strftime("%Y%m%d")
FILESTAMP = ""

def filterStock(sno,stype,signal,days,ruleout):

    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    df = cc.pd.read_csv(cc.OUTPATH+"/"+stype+"/"+sno+".csv",index_col=0)
    #df = df[:-10].copy()

    if "~" in days:    
        sdate = days.split('~')[0]
        edate = days.split('~')[1]
        df = df.loc[(df.index >= sdate) & (df.index<edate)]         
    else:
        datadate = (cc.datetime.now() - cc.timedelta(days=int(days))).strftime("%Y-%m-%d")
        df = df.loc[(df.index > datadate)] 
            
    if "~" in signal:
        if all(col in df.columns for col in signal.split('~')):
            df = df.loc[df[signal.split('~')].any(axis=1)]
        else:
            df = cc.pd.DataFrame()
    else:
        if all(col in df.columns for col in signal.split('&')):
            df = df.loc[df[signal.split('&')].all(axis=1)]
        else:
            df = cc.pd.DataFrame()
    
    if ruleout!="":
        if all(col in df.columns for col in ruleout.split('&')):
            df = df.loc[df[ruleout.split('&')].all(axis=1)==False]
        else:
            df = cc.pd.DataFrame()                        
        
    df = df.reset_index()

    return df


def YFGetSLIST(stype,signal,days=0,ruleout=""):

    snolist = list(map(lambda s: s.replace(".csv", ""), cc.os.listdir(cc.OUTPATH+"/"+stype+"/")))
    SLIST = cc.pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST.assign(signal=signal+"")
    SLIST = SLIST.assign(ruleout=ruleout+"")
    SLIST = SLIST.assign(days=days)
    SLIST = SLIST[:]

    return SLIST


def YFFilter(SLIST,signaldf):    

    with cc.ExecutorType(max_workers=10) as executor:
        for tempdf in cc.tqdm(executor.map(filterStock,SLIST["sno"],SLIST["stype"],SLIST["signal"],SLIST["days"],SLIST["ruleout"],chunksize=1),total=len(SLIST)):            
            tempdf = tempdf.dropna(axis=1, how="all")
            signaldf = cc.pd.concat([tempdf, signaldf], ignore_index=True)

        if len(signaldf)!=0:
            signaldf["SNO"] = signaldf["sno"].str.replace(r'^0+', '', regex=True)        

        return signaldf
    



def YFSignal(stype,signal,days,sdate="2024/01/01",edate="2026/12/31",ruleout=""):
    
    signaldf = cc.pd.DataFrame()
    
    SLIST = YFGetSLIST(stype,signal,days,ruleout)
    signaldf = YFFilter(SLIST, signaldf)

    if len(signaldf)>0:


        signaldf = signaldf.sort_values(by=['SNO','index'],ascending=[True, True])

        if str(signal).startswith("BOSS2") or str(signal).startswith("BOSSB"):
            print(signaldf[['index','sno','BOSS_STATUS']])
        else:
            print(signaldf[['index','sno',signal]])

        
        signaldf.to_csv("Data/"+stype+"_"+signal+FILESTAMP+".csv",index=False)

    print(f"{signal} - {stype} : {len(signaldf)}")
    print("Finish")
        
    

if __name__ == '__main__':

    #SDATE, EDATE = "2003/01/01", "2007/10/31"  #up
    #SDATE, EDATE = "2007/11/01", "2008/12/31"  #down
    #SDATE, EDATE = "2009/01/01", "2009/12/31"  #up
    #SDATE, EDATE = "2010/01/01", "2015/12/31"  #consolidation
    #SDATE, EDATE = "2016/01/01", "2017/12/31"  #up
    #SDATE, EDATE = "2018/01/01", "2020/03/31"  #down&onsolidation
    #SDATE, EDATE = "2020/04/01", "2020/12/31"  #up
    #SDATE, EDATE = "2021/01/01", "2022/10/31"  #down
    #SDATE, EDATE = "2022/11/01", "2023/01/31"  #up
    #SDATE, EDATE = "2023/02/01", "2024/01/31"  #down
    #SDATE, EDATE = "2024/02/01", "2025/12/31"  #up
    #SDATE, EDATE = "2021/01/01", "2025/12/31"  
    SDATE, EDATE = "1999/01/01", "2026/12/31"  
       
    #DAYS = str((datetime.strptime(EDATE, "%Y/%m/%d") - datetime.strptime(SDATE, "%Y/%m/%d")).days)

    DAYS = "20000"        

    start = cc.t.perf_counter()

    YFSignal("L","BOSS2~BOSSB~BOSSCL1","60")
    
    # for taname in cc.TALIST:
    #     YFSignal("L",taname,"1")

    # for modelname in cc.MODELLIST:
    #     YFSignal("L",modelname,"1")


    # YFSignal("L","EMA2","1")
    # YFSignal("M","EMA2","1")

    # YFSignal("L","T1_50&EMA2",DAYS)
    # YFSignal("M","T1_50&EMA2",DAYS)    

    #YFSignal("L","BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2~BOSSTU1~BOSSTU2",DAYS,SDATE,EDATE)
    #YFSignal("M","BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2~BOSSTU1~BOSSTU2",DAYS,SDATE,EDATE)

    finish = cc.t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')    