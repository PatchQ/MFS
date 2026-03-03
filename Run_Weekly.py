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

    #YFgetAll("L",cc.DATADATE)

    #process All Data
    #ProcessTA("L")
 
    YFSignal("L","BOSS2~BOSSB~BOSSCL1","60")
    
    for taname in cc.TALIST:
        YFSignal("L",taname,"5")
        #YFSignal("M",taname,"5")

    for modelname in cc.MODELLIST:
        YFSignal("L",modelname,"5")
        #YFSignal("M",modelname,"5")
    
    finish = cc.t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
