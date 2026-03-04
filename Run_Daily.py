import UTIL.CommonConfig as cc

from HKEX.AA_GetIndustryList import getIndustryList
from HKEX.AA_GetStockListData import getStockListData
from UTIL.LW_Collect import YFgetAll
from UTIL.LW_FilterStock import YFSignal
from ProcessTA import ProcessTA

if __name__ == '__main__':

    start = cc.t.perf_counter()
    
    #get IndustryList from AA
    #getIndustryList()

    #get All number from AA
    #getStockListData()

    # #get All History Data from YF
    #YFgetAll("L")
    #YFgetDaily("L")

    #YFgetAll("L",cc.DATADATE)

    #process All Data
    #ProcessTA("L",ai="True")
 
    #YFSignal("L","BOSS2~BOSSB~BOSSCL1","30")
    
    for taname in cc.TALIST:
        YFSignal("L",taname,"1")        

    for modelname in cc.MODELLIST:
        YFSignal("L",modelname,"1")
    
    finish = cc.t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
