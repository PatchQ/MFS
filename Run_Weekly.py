import time as t

from AA_GetIndustryList import getIndustryList
from AA_GetStockListData import getStockListData
from YFData_Collect import YFgetAll
from YFData_Process_sk import YFprocessData
from YFData_FilterStock import YFSignal

PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/" 

if __name__ == '__main__':

    start = t.perf_counter()
    
    #get IndustryList from AA
    getIndustryList()

    #get All number from AA
    getStockListData()

    # #get All History Data from YF
    YFgetAll("L")
    YFgetAll("M")
    YFgetAll("S")

    #process All Data
    # YFprocessData("L")
    # YFprocessData("M")
    # YFprocessData("S")

    # #get Signal to excelfile
    # YFSignal("L","T1_150&EMA2","250")
    # YFSignal("M","T1_150&EMA2","250")
    # YFSignal("S","T1_150&EMA2","250")

    # YFSignal("L","BOSS1~BOSSCL1","30")
    # YFSignal("M","BOSS1~BOSSCL1","30")
    # YFSignal("S","BOSS1~BOSSCL1","30")       
    
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
