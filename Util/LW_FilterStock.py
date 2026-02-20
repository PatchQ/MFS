import pandas as pd
import concurrent.futures as cf
import os
import time as t
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import defaultdict


#get stock csv file from path
OUTPATH = "../SData/P_YFData/"
#OUTPATH = "../SData/FP_YFData/"

#EDATE = "2025-09-30"
EDATE = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
FILESTAMP = "_"+datetime.now().strftime("%Y%m%d")
FILESTAMP = ""


def filterStock(sno,stype,signal,days,ruleout):

    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    df = pd.read_csv(OUTPATH+"/"+stype+"/"+sno+".csv",index_col=0)
    #df = df[:-10].copy()

    if "~" in days:    
        sdate = days.split('~')[0]
        edate = days.split('~')[1]
        df = df.loc[(df.index>=sdate) & (df.index<edate)]         
    else:
        datadate = (datetime.now() - timedelta(days=int(days))).strftime("%Y-%m-%d")
        df = df.loc[(df.index>=datadate)] 
            
    if "~" in signal:
        if all(col in df.columns for col in signal.split('~')):
            df = df.loc[df[signal.split('~')].any(axis=1)]
        else:
            df = pd.DataFrame()
    else:
        if all(col in df.columns for col in signal.split('&')):
            df = df.loc[df[signal.split('&')].all(axis=1)]
        else:
            df = pd.DataFrame()
    
    if ruleout!="":
        if all(col in df.columns for col in ruleout.split('&')):
            df = df.loc[df[ruleout.split('&')].all(axis=1)==False]
        else:
            df = pd.DataFrame()                        
        
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

    with cf.ProcessPoolExecutor(max_workers=5) as executor:
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
    seq1 = ['BY1','TP1','TP2','TP3']

    seq2 = ['BY1','TP1','TP2']

    seq3 = ['BY1','TP1']
    seq3_1 = ['TP1']

    seq4 = ['BY1','TP2']
    seq4_1 = ['TP1','TP2']
    seq4_2 = ['TP2']

    seq5 = ['BY1','TP3']
    seq5_1 = ['BY1','TP1','TP3']
    seq5_2 = ['BY1','TP2','TP3']
    seq5_3 = ['TP1','TP3']
    seq5_4 = ['TP2','TP3']
    seq5_5 = ['TP3']

    seq6 = ['BY1','TP1','CL2']
    seq6_1 = ['TP1','CL2']
    
    seq7 = ['BY1','CL1']
    seq7_1 = ['CL1']    

    seq8 = ['BY1','TU1']

    seq9 = ['BY1','TU2']

    seq10 = ['BY1']

    # 使用defaultdict来存储每只股票的计数
    stock_counts_dict = defaultdict(lambda: {'TP123': 0, 'TP12': 0, 'TP1': 0, 'TP2': 0, 'TP3': 0, 'TP1C': 0, 'CL1': 0, 'TU1': 0, 'TU2': 0, 'BY1': 0})
    
   # 遍历每个组
    for key, events in groups.items():
        stock, date = key

        if ((date>=sdate) and (date<=edate)):
            if events == seq1:
                stock_counts_dict[stock]['TP123'] += 1
            elif events == seq2:
                stock_counts_dict[stock]['TP12'] += 1
            elif (events == seq3) or (events == seq3_1):
                stock_counts_dict[stock]['TP1'] += 1
            elif (events == seq4) or (events == seq4_1) or (events == seq4_2):
                stock_counts_dict[stock]['TP2'] += 1
            elif (events == seq5) or (events == seq5_1) or (events == seq5_2) or (events == seq5_3) or (events == seq5_4) or (events == seq5_5):
                stock_counts_dict[stock]['TP3'] += 1
            elif (events == seq6) or (events == seq6_1):
                stock_counts_dict[stock]['TP1C'] += 1
            elif (events == seq7) or (events == seq7_1):
                stock_counts_dict[stock]['CL1'] += 1
            elif events == seq8:
                stock_counts_dict[stock]['TU1'] += 1
            elif events == seq9:
                stock_counts_dict[stock]['TU2'] += 1
            elif events == seq10:
                stock_counts_dict[stock]['BY1'] += 1

    # 将字典转换为DataFrame    
    stock_counts_df = pd.DataFrame.from_dict(stock_counts_dict, orient='index')

    if len(stock_counts_df)>0:
        stock_counts_df = stock_counts_df.reset_index()
        stock_counts_df.columns = ['sno', 'TP123', 'TP12', 'TP1', 'TP2', 'TP3', 'TP1C', 'CL1', 'TU1', 'TU2', 'BY1']
        stock_counts_df = stock_counts_df.sort_values('sno')

        #stock_counts_df["TOTAL"] = stock_counts_df.sum(axis=1,numeric_only=True) 
        stock_counts_df["TOTAL"] = stock_counts_df[['TP123','TP12','TP1','TP2','TP3','TP1C','CL1','TU2']].sum(axis=1,numeric_only=True) 
        stock_counts_df.loc['SUM'] = stock_counts_df.sum(numeric_only=True)
        stock_counts_df["WR"] = round(((stock_counts_df[['TP123','TP12','TP1','TP2','TP3','TP1C']].sum(axis=1) / stock_counts_df["TOTAL"]) * 100),2)    

        print(stock_counts_df.iloc[:,-1].tail(1))

        stock_counts_df.to_csv("Data/"+stype+"_"+signal+"_Stat"+FILESTAMP+".csv",index=False)


def YFSignal(stype,signal,days,sdate="2024/01/01",edate="2026/12/31",ruleout=""):
    
    signaldf = pd.DataFrame()
    
    SLIST = YFGetSLIST(stype,signal,days,ruleout)
    signaldf = YFFilter(SLIST, signaldf)

    if len(signaldf)>0:

        if int(days)>30000:
            countBOSS(stype,signal,signaldf,sdate,edate)

        signaldf = signaldf.sort_values(by=['SNO','index'],ascending=[True, True])

        if str(signal).startswith("BOSS2") or str(signal).startswith("BOSSB"):
            print(signaldf[['index','sno','BOSS_STATUS']])
        elif str(signal).startswith("HHHL"):
            print(signaldf[['index','sno','HHHL']])
        elif str(signal).startswith("VCP"):
            print(signaldf[['index','sno','VCP']])
        
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

    start = t.perf_counter()

    YFSignal("L","BOSS2~BOSSB~BOSSCL1","10")
    YFSignal("M","BOSS2~BOSSB~BOSSCL1","10")
    
    YFSignal("L","HHHL","10")
    YFSignal("M","HHHL","10")

    YFSignal("L","VCP","10")
    YFSignal("M","VCP","10")

    YFSignal("L","EMA2","1")
    YFSignal("M","EMA2","1")

    YFSignal("L","T1_50&EMA2",DAYS)
    YFSignal("M","T1_50&EMA2",DAYS)    

    #YFSignal("L","BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2~BOSSTU1~BOSSTU2",DAYS,SDATE,EDATE)
    #YFSignal("M","BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2~BOSSTU1~BOSSTU2",DAYS,SDATE,EDATE)

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')    