import time as t

from HK.AA_GetIndustryList import getIndustryList
from HK.AA_GetStockListData import getStockListData
from Util.LW_Collect import YFgetAll
from Util.LW_Collect import YFgetDaily
from Util.LW_ProcessBOSS import YFprocessData
from Util.LW_FilterStock import YFSignal
import Util.LW_ProcessBOSS

Util.LW_ProcessBOSS.PATH = "../SData/YFData/"
Util.LW_ProcessBOSS.OUTPATH = "../SData/P_YFData/" 

DAYS = "60"

if __name__ == '__main__':

    start = t.perf_counter()
    
    #get IndustryList from AA
    #getIndustryList()

    #get All number from AA
    #getStockListData()

    # #get All History Data from YF
    # YFgetAll("L")
    # YFgetAll("M")
    # YFgetAll("S")

    # YFgetAll("L","2y")
    # YFgetAll("M","2y")
    # YFgetAll("S","2y")


    #process All Data
    YFprocessData("L")
    YFprocessData("M")
    YFprocessData("S")

    #get Signal to excelfile
    YFSignal("L","BOSS2~BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2",DAYS)
    YFSignal("M","BOSS2~BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2",DAYS)
    YFSignal("S","BOSS2~BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2",DAYS)    

    YFSignal("L","T1_50",DAYS)
    YFSignal("M","T1_50",DAYS)
    YFSignal("S","T1_50",DAYS)
    
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
