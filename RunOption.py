import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  

IOPATH = "../Sdata/HKEX/IO/"

#EDATE = "2025-09-30"
EDATE = (cc.datetime.now() - cc.timedelta(days=30)).strftime("%Y-%m-%d")
FILESTAMP = "_"+cc.datetime.now().strftime("%Y%m%d")
FILESTAMP = ""

def getStrike(op, oyear, omonth, sdate, edate):    
   
   op="HSI"
   oyear="26"
   omonth="MAR"

   sdate="20260201"
   edate="20260316"
   
   
   SLIST = SLIST[:]


   with cc.ExecutorType(max_workers=10) as executor:
        for tempdf in cc.tqdm(executor.map(filterOption,SLIST["oyear"],SLIST["omonth"],SLIST["oday"],chunksize=1),total=len(SLIST)):                        
            signaldf = cc.pd.concat([tempdf, signaldf], ignore_index=True)

        if len(signaldf)!=0:
            signaldf["SNO"] = signaldf["sno"].str.replace(r'^0+', '', regex=True)        

        return signaldf
    
    
    

if __name__ == '__main__':

    SDATE, EDATE = "1999/01/01", "2026/12/31"  
       
    DAYS = "20000"        

    start = cc.t.perf_counter()

    getStrike("L","BOSS2~BOSSB~BOSSCL1","60")
    
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