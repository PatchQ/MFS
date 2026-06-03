import time as t
import YFData_Process_sk 
from YFData_Process_sk import YFprocessData
from YFData_FilterStock import YFSignal

YFData_Process_sk.PATH = "../SData/USData/"
YFData_Process_sk.OUTPATH = "../SData/P_USData/" 

if __name__ == '__main__':

    start = t.perf_counter()
    
    #process All Data
    YFprocessData("XASE")
    YFprocessData("XNMS")
    YFprocessData("XNCM")
    YFprocessData("XNGS")
    YFprocessData("XNYS")

    # #get Signal to excelfile
    # YFSignal("L","T1_150&EMA2","250")
    # YFSignal("M","T1_150&EMA2","250")
    # YFSignal("S","T1_150&EMA2","250")

    # YFSignal("L","BOSS1~BOSSCL1","30")
    # YFSignal("M","BOSS1~BOSSCL1","30")
    # YFSignal("S","BOSS1~BOSSCL1","30")       
    
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
