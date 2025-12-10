import time as t

from Util.LW_Collect import YFgetDaily
from Util.LW_ProcessBOSS import YFprocessData
from Util.LW_FilterStock import YFSignal
import Util.LW_ProcessBOSS

Util.LW_ProcessBOSS.PATH = "../SData/YFData/"
Util.LW_ProcessBOSS.OUTPATH = "../SData/P_YFData/" 

if __name__ == '__main__':

    start = t.perf_counter()

    DAYS = "5"
    #get Daily Data from YF
    YFgetDaily("L")
    YFgetDaily("M")
    #YFgetDaily("S")

    #process All Data
    YFprocessData("L")
    YFprocessData("M")
    #YFprocessData("S")

    YFSignal("L","BOSS2~BOSSB~BOSSTP1~BOSSTP2~BOSSCL1~BOSSCL2",DAYS)
    YFSignal("M","BOSS2~BOSSB~BOSSTP1~BOSSTP2~BOSSCL1~BOSSCL2",DAYS)
    #YFSignal("S","BOSS2~BOSSB~BOSSTP1~BOSSTP2~BOSSCL1~BOSSCL2",DAYS)    

    # YFprocessDataW10("L")
    # YFprocessDataW10("M")
    # YFprocessDataW10("S")

    # YFSignal("L","BOSS2~BOSSB~BOSSTP1~BOSSTP2~BOSSCL1~BOSSCL2",DAYS)
    # YFSignal("M","BOSS2~BOSSB~BOSSTP1~BOSSTP2~BOSSCL1~BOSSCL2",DAYS)
    # YFSignal("S","BOSS2~BOSSB~BOSSTP1~BOSSTP2~BOSSCL1~BOSSCL2",DAYS)    


    # YFSignal("L","T1_100",DAYS)
    # YFSignal("M","T1_100",DAYS)
    # YFSignal("S","T1_100",DAYS)
    
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
