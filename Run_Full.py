import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc
from HEX.AA_GetIndustryList import getIndustryList
from HEX.AA_GetStockListData import getStockListData
from UTIL.LW_Collect import DS_YFgetAll  # 用 DataSource Registry
from UTIL.LW_FilterStock import YFSignal
from ProcessTA import ProcessTA
from ProcessAI import CalAI

if __name__ == '__main__':

    # 暫存原始路徑
    _orig_path = cc.PATH
    _orig_outpath = cc.OUTPATH

    # 切換到 Full 版本的路徑（不影響 Run_Daily2.py）
    cc.PATH = cc.FPATH      # /root/GitHub/SData/FYFData/
    cc.OUTPATH = cc.FOUTPATH # /root/GitHub/SData/FP_YFData/

    start = cc.t.perf_counter()

    #get IndustryList from AA
    #getIndustryList()

    #get All number from AA
    #getStockListData()

    #get All History Data from YF (使用 DataSource Registry)
    DS_YFgetAll("L", "1900-01-01")  # 全量歷史
    DS_YFgetAll("M", "1900-01-01")

    ProcessTA("L", ai="False")
    ProcessTA("M", ai="False")

    #CalAI("L")
    #CalAI("M")

    finish = cc.t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')

    # 恢復原始路徑
    cc.PATH = _orig_path
    cc.OUTPATH = _orig_outpath
