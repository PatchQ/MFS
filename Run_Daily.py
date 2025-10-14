import time as t

from AA_GetIndustryList import getIndustryList
from AA_GetStockListData import getStockListData
from YFData_Collect import YFgetAll
from YFData_Collect import YFgetDaily
from YFData_Process import YFprocessData
from YFData_FilterStock import YFSignal
from YFData_ProcessBOSS import YFProcessBOSS

PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/" 

if __name__ == '__main__':

    start = t.perf_counter()

    DAYS = 0
    #get IndustryList from AA
    #getIndustryList()

    #get All number from AA
    #getStockListData()

    # #get All History Data from YF
    # YFgetAll("L")
    # YFgetAll("M")
    # YFgetAll("S")

    #get Daily Data from YF
    YFgetDaily("L")
    YFgetDaily("M")
    YFgetDaily("S")

    #process All Data
    YFprocessData("L")
    YFprocessData("M")
    YFprocessData("S")

    #process All Data to BOSS
    YFProcessBOSS("L")
    YFProcessBOSS("M")
    YFProcessBOSS("S")

    # #get Signal to excelfile
    YFSignal("L","T1_22&EMA2",DAYS,"EMA1")
    YFSignal("M","T1_22&EMA2",DAYS,"EMA1")
    YFSignal("S","T1_22&EMA2",DAYS,"EMA1")    

    YFSignal("L","T1_50&EMA1",DAYS)
    YFSignal("M","T1_50&EMA1",DAYS)
    YFSignal("S","T1_50&EMA1",DAYS)        

    YFSignal("L","T1_22&EMA1",DAYS,"T1_50")
    YFSignal("M","T1_22&EMA1",DAYS,"T1_50")
    YFSignal("S","T1_22&EMA1",DAYS,"T1_50")

    YFSignal("HHLL","BOSS1",30)
    YFSignal("HHLL","BOSS2",30)
    
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
