import time as t

from UTIL.LW_Collect import YFgetDaily
from UTIL.LW_ProcessTA import ProcessTA
from UTIL.LW_FilterStock import YFSignal

if __name__ == '__main__':

    start = t.perf_counter()

    #get Daily Data from YF
    YFgetDaily("L")
    YFgetDaily("M")
    
    #process All Data
    ProcessTA("L")
    ProcessTA("M")

    YFSignal("L","BOSS2~BOSSB~BOSSCL1","20")
    YFSignal("M","BOSS2~BOSSB~BOSSCL1","20")
    
    YFSignal("L","HHHL","5")
    YFSignal("M","HHHL","5")

    YFSignal("L","VCP","5")
    YFSignal("M","VCP","5")

    YFSignal("L","EMA1","1")
    YFSignal("M","EMA1","1")
   
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
