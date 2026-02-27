import time as t

from UTIL.LW_Collect import YFgetDaily
from UTIL.LW_FilterStock import YFSignal

from ProcessTA import ProcessTA

if __name__ == '__main__':

    DAYS="1"

    start = t.perf_counter()

    #get Daily Data from YF
    #YFgetDaily("L")
    #YFgetDaily("M")
    
    #process All Data
    ProcessTA("L")
    ProcessTA("M")

    # YFSignal("L","BOSS2~BOSSB~BOSSCL1","20")
    # YFSignal("M","BOSS2~BOSSB~BOSSCL1","20")

    # YFSignal("L","DT",DAYS)
    # YFSignal("M","DT",DAYS)
    
    # YFSignal("L","HHHL",DAYS)
    # YFSignal("M","HHHL",DAYS)

    # YFSignal("L","VCP","5")
    # YFSignal("M","VCP","5")

    #YFSignal("L","EMA1",DAYS)
    #YFSignal("M","EMA1",DAYS)
   
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
