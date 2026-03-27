import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  
from HEX.AA_GetIndustryList import getIndustryList
from HEX.AA_GetStockListData import getStockListData
from UTIL.LW_Collect import YFgetAll
from UTIL.LW_FilterStock import YFSignal
from ProcessTA import ProcessTA
from ProcessAI import CalAI

if __name__ == '__main__':

    start = cc.t.perf_counter()
    
    #get IndustryList from AA
    #getIndustryList()

    #get All number from AA
    #getStockListData()

    #get All History Data from YF
    # YFgetAll("L")    
    # ProcessTA("L",ai="False")
    # CalAI()
    #YFgetDaily("L")

    #get History Data from YF   
    #YFgetAll("L",cc.DATADATE)

    #process Data
    #ProcessTA("L",ai="True")
 
    YFSignal("L","BOSS2~BOSSB~BOSSCL1","30")
    
    for taname in cc.TALIST:
        YFSignal("L",taname,"30")        

    for modelname in cc.MODELLIST:
        YFSignal("L",modelname,"30")
    
    finish = cc.t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
