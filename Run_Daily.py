import UTIL.CommonConfig as cc

from UTIL.LW_Collect import YFgetDaily
from UTIL.LW_FilterStock import YFSignal

from ProcessTA import ProcessTA

if __name__ == '__main__':

    DAYS="1"

    start = cc.t.perf_counter()

    #get Daily Data from YF
    #YFgetDaily("L")
    #YFgetDaily("M")
    
    #process All Data
    #ProcessTA("L")
    #ProcessTA("M")

    # YFSignal("L","BOSS2~BOSSB~BOSSCL1","30")
    # YFSignal("M","BOSS2~BOSSB~BOSSCL1","30")

    # for modelname in cc.MODELLIST:
    #     YFSignal("L",modelname,"5")
    #     YFSignal("M",modelname,"5")

    YFSignal("L","HHHL","5")
    YFSignal("M","HHHL","5")

    YFSignal("L","VCP","5")
    YFSignal("M","VCP","5")            

   
    finish = cc.t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
