import time as t

from HKEX.AA_GetIndustryList import getIndustryList
from HKEX.AA_GetStockListData import getStockListData
from UTIL.LW_Collect import YFgetAll
from UTIL.LW_ProcessTA import ProcessTA
from UTIL.LW_FilterStock import YFSignal
import platform

if __name__ == '__main__':

    start = t.perf_counter()
    
    #get IndustryList from AA
    #getIndustryList()

    #get All number from AA
    #getStockListData()

    # #get All History Data from YF
    #YFgetAll("L")
    #YFgetAll("M")
    #YFgetAll("S")

    #YFgetAll("L","3y")
    #YFgetAll("M","3y")
    #YFgetAll("S","2y")

    #process All Data
    #ProcessTA("L")
    ProcessTA("M")
 
    # YFSignal("L","BOSS2~BOSSB~BOSSCL1","20")
    # YFSignal("M","BOSS2~BOSSB~BOSSCL1","20")
#      
    # YFSignal("L","HHHL&EMA1","5")
    # YFSignal("M","HHHL&EMA1","5")
# 
    # YFSignal("L","VCP","5")
    # YFSignal("M","VCP","5")
# 
    # YFSignal("L","EMA1","1")
    # YFSignal("M","EMA1","1")
#     
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
