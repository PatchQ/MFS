import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import glob
import os
import UTIL.CommonConfig as cc
from HEX.AA_GetIndustryList import getIndustryList
from HEX.AA_GetStockListData import getStockListData
from UTIL.LW_Collect import YFgetAll
from UTIL.LW_FilterStock import YFSignal
from ProcessTA import ProcessTA
from ProcessAI import CalAI

if __name__ == '__main__':

    # 清空上次的信號輸出
    for f in glob.glob("Data/Result/*.csv"):
        os.remove(f)

    start = cc.t.perf_counter()
    
    #get IndustryList from AA
    #getIndustryList()

    #get All number from AA
    #getStockListData()

    #get History Data from YF   
    YFgetAll("L",cc.DATADATE)
    YFgetAll("M",cc.DATADATE)

    #process Data
    ProcessTA("L",ai="True")
    ProcessTA("M",ai="True")
 
    YFSignal("L","BOSS2~BOSSB~BOSSCL1","30")
    YFSignal("M","BOSS2~BOSSB~BOSSCL1","30")
    
    for taname in cc.TALIST:
      YFSignal("L",taname,"2")
      YFSignal("M",taname,"2")

    for modelname in cc.MODELLIST:
      YFSignal("L",modelname,"2")
      YFSignal("M",modelname,"2")
    
    finish = cc.t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
