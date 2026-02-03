import time as t

from HKEX.AA_GetIndustryList import getIndustryList
from HKEX.AA_GetStockListData import getStockListData
from UTIL.LW_Collect import YFgetAll
from UTIL.LW_ProcessBOSS import ProcessBOSS
from UTIL.LW_FilterStock import YFSignal
import UTIL.LW_ProcessBOSS

UTIL.LW_ProcessBOSS.PATH = "../SData/YFData/"
UTIL.LW_ProcessBOSS.OUTPATH = "../SData/P_YFData/" 

DAYS = "20"

if __name__ == '__main__':

    start = t.perf_counter()
    
    #get IndustryList from AA
    getIndustryList()

    #get All number from AA
    getStockListData()

    # #get All History Data from YF
    #YFgetAll("L")
    #YFgetAll("M")
    #YFgetAll("S")

    YFgetAll("L","3y")
    YFgetAll("M","3y")
    #YFgetAll("S","2y")


    #process All Data
    #ProcessBOSS("L")
    #ProcessBOSS("M")
    #ProcessBOSS("S")

    #get Signal to excelfile
    ##YFSignal("L","BOSS2~BOSSB~BOSSCL1",DAYS)
    #YFSignal("M","BOSS2~BOSSB~BOSSCL1",DAYS)

    #YFSignal("L","BOSS2~BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2~BOSSTU1~BOSSTU2",DAYS)
    #YFSignal("M","BOSS2~BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2~BOSSTU1~BOSSTU2",DAYS)
    #YFSignal("S","BOSS2~BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2~BOSSTU1~BOSSTU2",DAYS)

    # YFSignal("L","T1_50",DAYS)
    # YFSignal("M","T1_50",DAYS)
    # YFSignal("S","T1_50",DAYS)
    
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
