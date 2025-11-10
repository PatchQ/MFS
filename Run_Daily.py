import time as t

from AA_GetIndustryList import getIndustryList
from AA_GetStockListData import getStockListData
from YFData_Collect import YFgetAll
from YFData_Collect import YFgetDaily
from YFData_Process_sk import YFprocessData
from YFData_FilterStock import YFSignal
from YFData_ProcessBOSS import YFProcessBOSS

PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/" 

if __name__ == '__main__':

    start = t.perf_counter()

    DAYS = "5"
    #get Daily Data from YF
    YFgetDaily("L")
    YFgetDaily("M")
    YFgetDaily("S")

    #process All Data
    # YFprocessData("L")
    # YFprocessData("M")
    # YFprocessData("S")

    # YFSignal("L","BOSS1~BOSSCL1",DAYS)
    # YFSignal("M","BOSS1~BOSSCL1",DAYS)
    # YFSignal("S","BOSS1~BOSSCL1",DAYS)       
    
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
