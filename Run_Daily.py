import time as t

from AA_GetIndustryList import getIndustryList
from AA_GetStockListData import getStockListData
from YFData_GetAll import YFgetAll
from YFData_Process import YFprocessData
from YFData_FindStock import YFfindSignal

PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/" 

if __name__ == '__main__':

    start = t.perf_counter()

    #get IndustryList from AA
    getIndustryList()

    #get All number from AA
    getStockListData()

    #get All History Data from YF
    YFgetAll("L")
    YFgetAll("M")
    YFgetAll("S")

    #process All Data
    YFprocessData("L")
    YFprocessData("M")
    YFprocessData("S")

    #get Signal to excelfile
    YFfindSignal("L","T1_22&EMA1",0)
    YFfindSignal("M","T1_22&EMA1",0)
    YFfindSignal("S","T1_22&EMA1",0)
    
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')