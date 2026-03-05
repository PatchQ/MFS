import UTIL.CommonConfig as cc
from ProcessAI import ProcessAI
from ProcessTA import ProcessTA

from HKEX.AA_GetIndustryList import getIndustryList
from HKEX.AA_GetStockListData import getStockListData
from UTIL.LW_Collect import YFgetAll


if __name__ == '__main__':

    start = cc.t.perf_counter()
    
    #get IndustryList from AA
    getIndustryList()

    #get All number from AA
    getStockListData()

    # #get All History Data from YF
    YFgetAll("L")    

    #process All Data
    ProcessTA("L",ai="False")
 
    #cal AI Model
    MODELLIST = [cc.RF,cc.SVM,cc.MLP,cc.LR,cc.DT]

    for modelfunction in MODELLIST:
        print(modelfunction.__name__)
        ProcessAI("L",modelfunction,cc.DATADATE)     