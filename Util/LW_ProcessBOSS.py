import os
import pandas as pd
import time as t
import concurrent.futures as cf
from tqdm import tqdm

try:

    from LW_Calindicator import *
    from LW_CalHHHL import *
    from LW_BossSkill import *

except ImportError:

    from UTIL.LW_Calindicator import *
    from UTIL.LW_CalHHHL import *
    from UTIL.LW_BossSkill import *    


PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/" 

def AnalyzeData(sno,stype):

    stockdata = pd.read_csv(PATH+"/"+stype+"/"+sno+".csv",index_col=0) 
    stockdata.index = pd.to_datetime(stockdata.index)        

    analyzer = SwingPointAnalyzer(sno=sno, stockdata=stockdata)

    analyzer.df = calEMA(analyzer.df)
    analyzer.df = calT1(analyzer.df,50)

    analyzer.df = extendData(analyzer.df)
    analyzer.df = convertData(analyzer.df)

    analyzer.calculate_daily_volatility(window=20)
    analyzer.find_swing_points(window=20)
    classifications = analyzer.identify_HH_HL_LH_LL()        


    if classifications is not None:
        if len(classifications)>0:        
            analyzer.df = checkBoss(analyzer.df, sno, stype, pd.DataFrame(classifications))
            analyzer.df = checkWave(analyzer.df, sno, stype, pd.DataFrame(classifications))                     
            analyzer.df = analyzer.df.reset_index()
            analyzer.df = analyzer.df.sort_values(by=['index'],ascending=[True])
            analyzer.df.to_csv(OUTPATH+"/"+stype+"/P_"+sno+".csv",index=False)


def ProcessBOSS(stype):

    snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(PATH+"/"+stype)))
    SLIST = pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST[:]

    with cf.ProcessPoolExecutor(max_workers=5) as executor:
        list(tqdm(executor.map(AnalyzeData,SLIST["sno"],SLIST["stype"],chunksize=1),total=len(SLIST)))


if __name__ == '__main__':
    start = t.perf_counter()

    #ProcessBOSS("L")    
    ProcessBOSS("M")
    #ProcessBOSS("S")

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    