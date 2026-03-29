import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  

def AnalyzeStock(sno,stype,ai):

    df = cc.pd.read_csv(cc.PATH+"/"+stype+"/"+sno+".csv",index_col=0) 
    df.index = cc.pd.to_datetime(df.index)  

    df = cc.extendData(df)
    df = cc.convertData(df)

    #EMA
    df = cc.calEMA(df)

    #T1
    df = cc.checkT1(df,50)

    #cal HHHL
    HHLLdf = cc.calHHLL(df)

    if HHLLdf is not None:
        if len(HHLLdf)>0:        
            #Boss
            df = cc.checkBoss(df, sno, stype, HHLLdf)
            #HHHL
            df = cc.checkWave(df, sno, stype, HHLLdf)                     
    
    #VCP
    df = cc.checkVCP(df)

    #AI Signal
    if ai=="True":
        signals = {}
        for modelname in cc.MODELLIST:
            signals[modelname] = cc.loadModel(modelname, sno, df)

        for modelname, sig in signals.items():
            df[modelname] = sig if len(sig) > 0 else False                        

    df = df.reset_index()
    #df = df.sort_values(by=['index'],ascending=[True])
    df = df[:-10].copy()
    df.to_csv(cc.OUTPATH+"/"+stype+"/P_"+sno+".csv",index=False)
    


def ProcessTA(stype,ai):

    snolist = list(map(lambda s: s.replace(".csv", ""), cc.os.listdir(cc.PATH+"/"+stype)))
    SLIST = cc.pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST.assign(ai=ai+"")
    SLIST = SLIST[:]

    #print(SLIST)

    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        list(cc.tqdm(executor.map(AnalyzeStock,SLIST["sno"],SLIST["stype"],SLIST["ai"],chunksize=1),total=len(SLIST)))


if __name__ == '__main__':
    start = cc.t.perf_counter()

    ProcessTA("L",ai="False")    
    #ProcessTA("L",ai="True")    
    #ProcessTA("M")    

    finish = cc.t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    