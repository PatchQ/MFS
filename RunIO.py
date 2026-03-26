from fileinput import filename
import sys
import os
import holidays
import calendar
from datetime import datetime, timedelta

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  

#EDATE = "2025-09-30"
EDATE = (cc.datetime.now() - cc.timedelta(days=30)).strftime("%Y-%m-%d")
FILESTAMP = "_"+cc.datetime.now().strftime("%Y%m%d")
FILESTAMP = ""

def filterOption(filename, oyear, omonth, oday, strike):

    op = filename.split("_")[0]
    filedate = filename.split("_")[1].replace(".csv", "")
    df = cc.pd.read_csv(cc.IOPATH+"/"+op+"/"+filename,index_col=0)    
    df = df.loc[(df['year'] == oyear) & (df['month_abbr'] == omonth) & (df['month_num'] == oday) & (df['strike'] == strike)]
    df.drop(columns=['month_num', 'month_abbr', 'year'], inplace=True)    
    df.insert(0, "filedate", filedate)
    df["lotgross"] = (df["call_gross"]>=1000) | (df["put_gross"]>=1000)
    df["lotnet"] = (df["call_net_change"]>=1000) | (df["put_net_change"]>=1000)
    return df


def getStrike(op, oyear, omonth, oday, strike, sdate, edate):   

   resultdf = cc.pd.DataFrame()
   filenamelist = []   
   templist = list(map(lambda s: s.replace(".csv", ""), cc.os.listdir(cc.IOPATH+"/"+op)))
   
   for tp in templist:
       filedate = tp.split("_")[1]       
       if filedate>=sdate and filedate<=edate:
           filename = op+"_"+filedate+".csv"
           filenamelist.append(filename)

   FLIST = cc.pd.DataFrame(filenamelist, columns=["filename"])
   FLIST = FLIST.assign(oyear=oyear)
   FLIST = FLIST.assign(omonth=omonth+"")  
   FLIST = FLIST.assign(oday=oday)
   FLIST = FLIST.assign(strike=strike)
   FLIST = FLIST[:]

   with cc.ExecutorType(max_workers=10) as executor:
        for tempdf in cc.tqdm(executor.map(filterOption,FLIST["filename"],FLIST["oyear"],FLIST["omonth"],FLIST["oday"],FLIST["strike"],chunksize=1),total=len(FLIST)):
            resultdf = cc.pd.concat([tempdf, resultdf], ignore_index=True)

   if resultdf["lotgross"].any() or resultdf["lotnet"].any():
        if resultdf["lotnet"].any():
            resultdf.to_csv(f"{cc.IOPATH}/P_{op}/C_{op}{oyear}{omonth}{strike}.csv",index=False)
        else:
            resultdf.to_csv(f"{cc.IOPATH}/P_{op}/{op}{oyear}{omonth}{strike}.csv",index=False)

if __name__ == '__main__':
       
    op="HSI"

    oyear=26
    omonth="MAR"
    
    oday=cc.getLastTradeDay(oyear, omonth).day

    start_strike=24800
    stop_strike=25800

    sdate="20260301"
    edate="20260324"

    start = cc.t.perf_counter()

    for strike in range(start_strike, stop_strike+1, 200):    
        getStrike(op, oyear, omonth, oday, strike, sdate, edate)        

    finish = cc.t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')    