import UTIL.CommonConfig as cc
from AI.ZPrediction import *

def AnalyzeStock(sno,stype):

    df = cc.pd.read_csv(cc.PATH+"/"+stype+"/"+sno+".csv",index_col=0) 
    df.index = cc.pd.to_datetime(df.index)  

    df = extendData(df)
    df = convertData(df)

    #EMA
    df = calEMA(df)

    #T1
    df = checkT1(df,50)

    #cal HHHL
    HHLLdf = calHHLL(df)

    if HHLLdf is not None:
        if len(HHLLdf)>0:        
            #Boss
            df = checkBoss(df, sno, stype, HHLLdf)
            #HHHL
            df = checkWave(df, sno, stype, HHLLdf)                     
    
    #VCP
    df = checkVCP(df)

    #AI Signal
    signals = {}
    for modelname in cc.MODELLIST:
        signals[modelname] = loadModel(modelname, sno, df)

    for modelname, sig in signals.items():
        df[modelname] = sig if len(sig) > 0 else False                        

    df = df.reset_index()
    #df = df.sort_values(by=['index'],ascending=[True])
    df = df[:-10].copy()
    df.to_csv(cc.OUTPATH+"/"+stype+"/P_"+sno+".csv",index=False)
    


def ProcessTA(stype):

    snolist = list(map(lambda s: s.replace(".csv", ""), cc.os.listdir(cc.PATH+"/"+stype)))
    SLIST = cc.pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST[:]

    #print(SLIST)

    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        list(cc.tqdm(executor.map(AnalyzeStock,SLIST["sno"],SLIST["stype"],chunksize=1),total=len(SLIST)))


if __name__ == '__main__':
    start = cc.t.perf_counter()

    ProcessTA("L")    
    ProcessTA("M")    

    finish = cc.t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    