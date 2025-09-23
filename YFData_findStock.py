import pandas as pd
import concurrent.futures as cf
import os
import time as t
from tqdm import tqdm
from datetime import datetime


#get stock excel file from path
PATH = "../SData/P_YFData/"
#EDATE = "2025-09-22"
EDATE = datetime.now().strftime("%Y-%m-%d")


def findStock(sno,stype,signal):

    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    df = pd.read_excel(PATH+"/"+stype+"/"+sno+".xlsx",index_col=0)
    df = df.loc[df.index>=EDATE]
    #df = df.loc[df[''+signal+''] & df["EMA"]]
    df = df.loc[df[signal.split('_')].all(axis=1)]
    df = df.reset_index()

    return df



def main(stype,signal):
    signaldf = pd.DataFrame()

    snolist = list(map(lambda s: s.replace(".xlsx", ""), os.listdir(PATH+"/"+stype+"/")))
    SLIST = pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST.assign(signal=signal+"")
    SLIST = SLIST[:]

    with cf.ProcessPoolExecutor(max_workers=12) as executor:
        for tempdf in tqdm(executor.map(findStock,SLIST["sno"],SLIST["stype"],SLIST["signal"],chunksize=1),total=len(SLIST)):            
            signaldf = pd.concat([tempdf, signaldf], ignore_index=True)
            
        signaldf.to_excel("Data/"+signal+"_"+stype+".xlsx",index=False)
        print("Finish")        



if __name__ == '__main__':

    start = t.perf_counter()

    #main("L","T1")
    #main("M","T1")
    #main("S","T1")
    
    main("L","T1_EMA")
    main("M","T1_EMA")
    main("S","T1_EMA")
    
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    


