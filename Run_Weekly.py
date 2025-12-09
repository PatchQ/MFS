import time as t

import YFData_Process_sk 
from AA_GetIndustryList import getIndustryList
from AA_GetStockListData import getStockListData
from YFData_Collect import YFgetAll
from YFData_Process_sk import YFprocessData
from YFData_FilterStock import YFSignal

YFData_Process_sk.PATH = "../SData/YFData/"
YFData_Process_sk.OUTPATH = "../SData/P_YFData/" 
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
