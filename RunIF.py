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

def filterFuture(filename, oyear, omonth):

    op = filename.split("_")[0]
    filedate = filename.split("_")[1].replace(".csv", "")
    df = cc.pd.read_csv(cc.IFPATH+"/"+op+"/"+filename,index_col=0)    
    df = df.loc[(df['year'] == oyear) & (df['month_abbr'] == omonth)]
    df.drop(columns=['month_num', 'month_abbr', 'year'], inplace=True)
    df.insert(0, "filedate", filedate)
    return df


def getFData(op, oyear, omonth, sdate, edate):   

   resultdf = cc.pd.DataFrame()
   filenamelist = []   
   templist = list(map(lambda s: s.replace(".csv", ""), cc.os.listdir(cc.IFPATH+"/"+op)))
   
   for tp in templist:
       filedate = tp.split("_")[1]       
       if filedate>=sdate and filedate<=edate:
           filename = op+"_"+filedate+".csv"
           filenamelist.append(filename)

   FLIST = cc.pd.DataFrame(filenamelist, columns=["filename"])
   FLIST = FLIST.assign(oyear=oyear)
   FLIST = FLIST.assign(omonth=omonth+"")  
   FLIST = FLIST[:]

   with cc.ExecutorType(max_workers=10) as executor:
        for tempdf in cc.tqdm(executor.map(filterFuture,FLIST["filename"],FLIST["oyear"],FLIST["omonth"],chunksize=1),total=len(FLIST)):
            resultdf = cc.pd.concat([tempdf, resultdf], ignore_index=True)

        resultdf.to_csv(f"{cc.IFPATH}/P_{op}/{op}{oyear}{omonth}.csv",index=False)

if __name__ == '__main__':
       
    op="HTI"

    oyear=26
    omonth="MAR"
    
    sdate="20260101"
    edate="20260323"

    start = cc.t.perf_counter()

    getFData(op, oyear, omonth, sdate, edate)        

    finish = cc.t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')    