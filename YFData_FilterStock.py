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

    seq8 = ['BY1']

    # 使用defaultdict来存储每只股票的计数
    stock_counts_dict = defaultdict(lambda: {'TP123': 0, 'TP12': 0, 'TP1': 0, 'TP2': 0, 'TP3': 0, 'TP1C': 0, 'CL1': 0, 'BY1': 0})
    
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
                stock_counts_dict[stock]['BY1'] += 1

    # 将字典转换为DataFrame    
    stock_counts_df = pd.DataFrame.from_dict(stock_counts_dict, orient='index')

    if len(stock_counts_df)>0:
        stock_counts_df = stock_counts_df.reset_index()
        stock_counts_df.columns = ['sno', 'TP123', 'TP12', 'TP1', 'TP2', 'TP3', 'TP1C', 'CL1', 'BY1']
        stock_counts_df = stock_counts_df.sort_values('sno')

        stock_counts_df["TOTAL"] = stock_counts_df.sum(axis=1,numeric_only=True)
        stock_counts_df.loc['SUM'] = stock_counts_df.sum(numeric_only=True)
        stock_counts_df["WR"] = round(((stock_counts_df[['TP123','TP12','TP1','TP2','TP3']].sum(axis=1) / stock_counts_df["TOTAL"]) * 100),2)    

        print(stock_counts_df.iloc[:,-1].tail(1))

        stock_counts_df.to_csv("Data/"+stype+"_"+signal+"_Stat_"+datetime.now().strftime("%Y%m%d")+".csv",index=False)


def YFSignal(stype,signal,days,sdate="2024/01/01",edate="2026/12/31",ruleout=""):
    
    signaldf = pd.DataFrame()
    
    SLIST = YFGetSLIST(stype,signal,days,ruleout)
    signaldf = YFFilter(SLIST, signaldf)

    if len(signaldf)>0:

        if int(days)>10000:
            countBOSS(stype,signal,signaldf,sdate,edate)

        signaldf = signaldf.sort_values(by=['SNO','Date'],ascending=[True, True])
        print(signaldf[['Date','sno','BOSS_STATUS']])
        
        signaldf.to_csv("Data/"+stype+"_"+signal+"_"+datetime.now().strftime("%Y%m%d")+".csv",index=False)

    print(f"{signal} - {stype} : {len(signaldf)}")
    print("Finish")
        
    

if __name__ == '__main__':

    #SDATE, EDATE = "2000/01/01", "2007/10/31"
    #SDATE, EDATE = "2007/11/01", "2008/12/31"
    #SDATE, EDATE = "2009/01/01", "2009/12/31"
    #SDATE, EDATE = "2010/01/01", "2015/12/31" 
    #SDATE, EDATE = "2016/01/01", "2017/12/31" 
    #SDATE, EDATE = "2018/01/01", "2020/12/31" 
    #SDATE, EDATE = "2021/01/01", "2023/12/31"
    SDATE, EDATE = "1900/01/01", "2025/12/31"  
    
       
    #DAYS = str((datetime.strptime(EDATE, "%Y/%m/%d") - datetime.strptime(SDATE, "%Y/%m/%d")).days)

    DAYS = "20000"
    DAYS = "60"
    start = t.perf_counter()

    YFSignal("L","BOSS2~BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2",DAYS)
    YFSignal("M","BOSS2~BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2",DAYS)
    #YFSignal("S","BOSS2~BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2",DAYS)    

    #YFSignal("L","BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2",DAYS,SDATE,EDATE)
    #YFSignal("M","BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2",DAYS,SDATE,EDATE)
    # YFSignal("S","BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2",DAYS,SDATE,EDATE)

    #YFSignal("L","T1_100",DAYS)
    #YFSignal("M","T1_100",DAYS)
    #YFSignal("S","T1_100",DAYS)
    
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    


