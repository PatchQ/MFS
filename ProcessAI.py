import UTIL.CommonConfig as cc

def ProcessAI(stype,modelfunction,tdate):

   resultdf = cc.pd.DataFrame()

   snolist = list(map(lambda s: s.replace(".csv", ""), cc.os.listdir(cc.OUTPATH+"/"+stype)))
   SLIST = cc.pd.DataFrame(snolist, columns=["sno"])   
   SLIST = SLIST.assign(stype=stype+"")
   SLIST = SLIST.assign(tdate=tdate+"")
   SLIST = SLIST[:]

    #print(SLIST)

   with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
       for tempdf in cc.tqdm(executor.map(modelfunction,SLIST["sno"],SLIST["stype"],SLIST["tdate"],chunksize=1),total=len(SLIST)):           
           resultdf = cc.pd.concat([tempdf, resultdf], ignore_index=True)

   #resultdf.to_csv(f"Data/{stype}_{model}.csv",index=False)    


if __name__ == '__main__':
    start = cc.t.perf_counter()
    
    MODELLIST = [cc.RF,cc.SVM,cc.MLP]    

    for modelfunction in MODELLIST:
        print(modelfunction.__name__)
        ProcessAI("L",modelfunction,cc.DATADATE)        

    finish = cc.t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    