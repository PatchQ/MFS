import pandas as pd
import time as t
import tushare as ts


from AA_GetIndustryList import getIndustryList
from AA_GetStockListData import getStockListData
from YFData_GetAll import YFgetAll
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

    
    ts.set_token('4228ed3bd2b53edd9c1ced494c8190da84cd05b2869822905b4a1bbd')
    pro = ts.pro_api()
    stock = ts.get_hist_data('0005.HK')
    print(stock)
    

    
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')