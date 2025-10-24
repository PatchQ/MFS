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


def filterStock(sno,stype,signal,days,ruleout):

    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    df = pd.read_csv(OUTPATH+"/"+stype+"/"+sno+".csv",index_col=0)


    if "~" in days:    
        sdate = days.split('~')[0]
        edate = days.split('~')[1]
        df = df.loc[(df.index>=sdate) & (df.index<edate)]         
    else:
        datadate = (datetime.now() - timedelta(days=int(days))).strftime("%Y-%m-%d")
        df = df.loc[(df.index>=datadate)] 
            
    if "~" in signal:
        df = df.loc[df[signal.split('~')].any(axis=1)]    
    else:
        df = df.loc[df[signal.split('&')].all(axis=1)]    
    
    if ruleout!="":
        df = df.loc[df[ruleout.split('&')].all(axis=1)==False]
        
    df = df.reset_index()

    return df


def YFGetSLIST(stype,signal,days=0,ruleout=""):

    snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(OUTPATH+"/"+stype+"/")))
    SLIST = pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST.assign(signal=signal+"")
    SLIST = SLIST.assign(ruleout=ruleout+"")
    SLIST = SLIST.assign(days=days)
    SLIST = SLIST[:]

    return SLIST


def YFFilter(stype,signal,SLIST,signaldf):    

    with cf.ProcessPoolExecutor(max_workers=12) as executor:
        for tempdf in tqdm(executor.map(filterStock,SLIST["sno"],SLIST["stype"],SLIST["signal"],SLIST["days"],SLIST["ruleout"],chunksize=1),total=len(SLIST)):            
            tempdf = tempdf.dropna(axis=1, how="all")
            signaldf = pd.concat([tempdf, signaldf], ignore_index=True)

        if len(signaldf)!=0:
            signaldf["sno"] = signaldf["sno"].str.replace(r'^0+', '', regex=True)

        signaldf.to_csv("Data/"+stype+"_"+signal+"_"+datetime.now().strftime("%Y%m%d")+".csv",index=False)

        print(f"{signal} - {stype} : {len(signaldf)}")
        print("Finish")
        return signaldf
    

def countBOSS(stype,signal,df):
    groups = {}
    
    # 遍历DataFrame的每一行
    for index, row in df.iterrows():
        sno = row['sno']
        bosstatus = row['BOSS_STATUS']
        
        # 分割事件和日期
        event_parts = bosstatus.split('-')
        if len(event_parts) < 2:
            continue
        event_code = event_parts[0]
        date = event_parts[1]  # 日期格式为 YYYY/MM/DD
        
        key = (sno, date)
        if key not in groups:
            groups[key] = []
        groups[key].append(event_code)

    print(groups)        
    
    # 定义目标序列
    seq1 = ['BY1', 'TP1', 'TP2']
    seq2 = ['BY1', 'TP1']
    seq3 = ['BY1', 'TP2']
    
    seq4 = ['BY1', 'TP1', 'CL2']

    seq5 = ['BY1', 'CL1']

    seq6 = ['BY1']
    
    # 初始化计数器
    count_seq1 = 0
    count_seq2 = 0
    count_seq3 = 0
    count_seq4 = 0
    count_seq5 = 0
    count_seq6 = 0
    
    # 存储匹配的组详细信息
    matches_seq1 = []
    matches_seq2 = []
    matches_seq3 = []
    
    rdf = pd.DataFrame()
    tempdf = pd.DataFrame()

    # 遍历每个组
    for key, events in groups.items():
        stock, date = key

        if events == seq1:
            count_seq1 += 1            
        elif events == seq2:
            count_seq2 += 1
        elif events == seq3:
            count_seq3 += 1
        elif events == seq4:
            count_seq4 += 1
        elif events == seq5:
            count_seq5 += 1
        elif events == seq6:
            count_seq6 += 1

        print(f"1:{count_seq1},2:{count_seq2},3:{count_seq3},4:{count_seq4},5:{count_seq5},6:{count_seq6}")
        #matches_seq3.append(f"{stock} at {date}")            
        tempdf['sno'] = stock
        tempdf['TP1TP2'] = count_seq1
        tempdf['TP1CL2'] = count_seq2
        tempdf['CL1'] = count_seq3
        tempdf['BY1'] = count_seq4
        tempdf['TOTAL'] = count_seq1+count_seq2+count_seq3+count_seq4

        rdf = pd.concat([tempdf, rdf], ignore_index=True)

    
    rdf.to_csv("Data/"+stype+"_"+signal+"_Stat_"+datetime.now().strftime("%Y%m%d")+".csv",index=False)


def YFSignal(stype,signal,days=0,ruleout=""):
    
    signaldf = pd.DataFrame()
    SLIST = YFGetSLIST(stype,signal,days,ruleout)
    signaldf = YFFilter(stype,signal, SLIST, signaldf)

    countBOSS(stype,signal,signaldf)
    

if __name__ == '__main__':

    # DAYS = "2000-01-01~2007-10-31"
    # DAYS = "2007-11-01~2008-12-31"
    # DAYS = "2009-01-01~2009-12-31"
    # DAYS = "2010-01-01~2015-12-31"
    # DAYS = "2016-01-01~2017-12-31"
    # DAYS = "2018-01-01~2020-12-31"
    # DAYS = "2021-01-01~2023-12-31"
    # DAYS = "2024-01-01~2025-12-31"
    #DAYS = "2009-01-01~2025-12-31"
    DAYS = "10000"
    start = t.perf_counter()
    
    # YFSignal("L","T1_22&EMA2",DAYS,"EMA1")
    # YFSignal("M","T1_22&EMA2",DAYS,"EMA1")
    # YFSignal("S","T1_22&EMA2",DAYS,"EMA1")    

    # YFSignal("L","T1_50&EMA1",DAYS)
    # YFSignal("M","T1_50&EMA1",DAYS)
    # YFSignal("S","T1_50&EMA1",DAYS)        

    # YFSignal("L","T1_22&EMA1",DAYS,"T1_50")
    # YFSignal("M","T1_22&EMA1",DAYS,"T1_50")
    # YFSignal("S","T1_22&EMA1",DAYS,"T1_50")
    
    # YFSignal("L","BOSS1~BOSSCL1",DAYS)
    # YFSignal("M","BOSS1~BOSSCL1",DAYS)
    # YFSignal("S","BOSS1~BOSSCL1",DAYS)    

    YFSignal("L","BOSSB~BOSSTP1~BOSSTP2~BOSSCL1~BOSSCL2",DAYS)
    #YFSignal("M","BOSS1~BOSSB~BOSSTP1~BOSSTP2~BOSSCL1~BOSSCL2",DAYS)


    #YFSignal("HHLL","BOSS2",30)
    
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    


