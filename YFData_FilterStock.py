import pandas as pd
import concurrent.futures as cf
import os
import time as t
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import defaultdict


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


def YFFilter(SLIST,signaldf):    

    with cf.ProcessPoolExecutor(max_workers=12) as executor:
        for tempdf in tqdm(executor.map(filterStock,SLIST["sno"],SLIST["stype"],SLIST["signal"],SLIST["days"],SLIST["ruleout"],chunksize=1),total=len(SLIST)):            
            tempdf = tempdf.dropna(axis=1, how="all")
            signaldf = pd.concat([tempdf, signaldf], ignore_index=True)

        if len(signaldf)!=0:
            signaldf["SNO"] = signaldf["sno"].str.replace(r'^0+', '', regex=True)        

        return signaldf
    

def countBOSS(stype,signal,df,sdate,edate):
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

    
    # 定义目标序列
    seq1 = ['BY1', 'TP1', 'TP2']
    seq2 = ['BY1', 'TP1']
    seq2_1 = ['TP1']

    seq3 = ['BY1', 'TP2']
    seq3_1 = ['TP2']
    
    seq4 = ['BY1', 'TP1', 'CL2']

    seq5 = ['BY1', 'CL1']
    seq5_1 = ['CL1']

    seq6 = ['BY1']

    # 使用defaultdict来存储每只股票的计数
    stock_counts_dict = defaultdict(lambda: {'TP12': 0, 'TP1': 0, 'TP2': 0, 'TP1C': 0, 'CL1': 0, 'BY1': 0})
    
   # 遍历每个组
    for key, events in groups.items():
        stock, date = key

        if ((date>=sdate) and (date<=edate)):
            if events == seq1:
                stock_counts_dict[stock]['TP12'] += 1
            elif (events == seq2) or (events == seq2_1):
                stock_counts_dict[stock]['TP1'] += 1
            elif (events == seq3) or (events == seq3_1):
                stock_counts_dict[stock]['TP2'] += 1
            elif events == seq4:
                stock_counts_dict[stock]['TP1C'] += 1
            elif (events == seq5) or (events == seq5_1):
                stock_counts_dict[stock]['CL1'] += 1
            elif events == seq6:
                stock_counts_dict[stock]['BY1'] += 1

    # 将字典转换为DataFrame
    stock_counts_df = pd.DataFrame.from_dict(stock_counts_dict, orient='index')
    stock_counts_df = stock_counts_df.reset_index()
    stock_counts_df.columns = ['sno', 'TP12', 'TP1', 'TP2', 'TP1C', 'CL1', 'BY1']
    stock_counts_df = stock_counts_df.sort_values('sno')

    stock_counts_df.to_csv("Data/"+stype+"_"+signal+"_Stat_"+datetime.now().strftime("%Y%m%d")+".csv",index=False)


def YFSignal(stype,signal,days,sdate,edate,ruleout=""):
    
    signaldf = pd.DataFrame()
    
    SLIST = YFGetSLIST(stype,signal,days,ruleout)
    signaldf = YFFilter(SLIST, signaldf)

    if len(signaldf)>0:

        if int(days)>100:
            countBOSS(stype,signal,signaldf,sdate,edate)

        signaldf = signaldf.sort_values(by=['SNO','Date'],ascending=[True, True])
        
        signaldf.to_csv("Data/"+stype+"_"+signal+"_"+datetime.now().strftime("%Y%m%d")+".csv",index=False)

    print(f"{signal} - {stype} : {len(signaldf)}")        
    print("Finish")
        
    

if __name__ == '__main__':

    # DAYS = "2000-01-01~2007-10-31"
    # DAYS = "2007-11-01~2008-12-31"
    # DAYS = "2009-01-01~2009-12-31"
    # DAYS = "2010-01-01~2015-12-31"
    # DAYS = "2016-01-01~2017-12-31"
    # DAYS = "2018-01-01~2020-12-31"
    # DAYS = "2021-01-01~2023-12-31"
    # DAYS = "2024-01-01~2025-12-31"
    # DAYS = "2009-01-01~2025-12-31"
    DAYS = "10000"
    SDATE = "2021/01/01"
    EDATE = "2023/12/31"
    SDATE = "1900/01/01"
    EDATE = "2024/12/31"
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

    #YFSignal("L","T1_150&EMA2","250")

    
    # YFSignal("L","BOSS1~BOSSB~BOSSTP1~BOSSTP2~BOSSCL1~BOSSCL2","60",SDATE,EDATE)
    # YFSignal("M","BOSS1~BOSSB~BOSSTP1~BOSSTP2~BOSSCL1~BOSSCL2","60",SDATE,EDATE)
    # YFSignal("S","BOSS1~BOSSB~BOSSTP1~BOSSTP2~BOSSCL1~BOSSCL2","60",SDATE,EDATE)    

    YFSignal("L","BOSSB~BOSSTP1~BOSSTP2~BOSSCL1~BOSSCL2",DAYS,SDATE,EDATE)
    YFSignal("M","BOSSB~BOSSTP1~BOSSTP2~BOSSCL1~BOSSCL2",DAYS,SDATE,EDATE)
    # YFSignal("S","BOSSB~BOSSTP1~BOSSTP2~BOSSCL1~BOSSCL2",DAYS,SDATE,EDATE)

    # YFSignal("L","T1_150&EMA2","250")
    # YFSignal("M","T1_150&EMA2","250")
    # YFSignal("S","T1_150&EMA2","250")




    #YFSignal("HHLL","BOSS2",30)
    
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    


