import pandas as pd
import time as t
import yfinance as yf


from AA_GetIndustryList import getIndustryList
from AA_GetStockListData import getStockListData
from YFData_Collect import YFgetAll
from YFData_Process import YFprocessData
from YFData_FilterStock import YFSignal

PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/" 

if __name__ == '__main__':

    start = t.perf_counter()

    #get IndustryList from AA
    #getIndustryList()

    #get All number from AA
    #getStockListData()

    #get All History Data from YF
    #YFgetAll("L")
    #YFgetAll("M")
    #YFgetAll("S")

    
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')