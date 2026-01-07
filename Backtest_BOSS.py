import pandas as pd
import os
from tqdm import tqdm
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
from talib import abstract

stype="L"

#get stock excel file from path
OUTPATH = "../SData/P_YFData/" 
snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(OUTPATH+"/"+stype)))
snolist = snolist[:10]

class KdCross(Strategy):

    #sl_ratio = 99     # stop loss ratio, 99 means 1% loss

    def init(self):
        super().init()

    def next(self):
        if self.data.BOSSB:
            self.buy()
            print(self.data.index, self.trades, self.position.pl_pct , self.position.size)
        
        elif self.position.pl_pct < -.1:
            self.position.close()
            print(self.data.index, self.trades, self.position.pl_pct , self.position.size)
            
        elif self.position.pl_pct > .2:
            self.position.close()
            print(self.data.index, self.trades, self.position.pl_pct , self.position.size)


for sno in tqdm(snolist):
    print(sno)
    
    df = pd.read_csv(OUTPATH+"/"+stype+"/"+sno+".csv")

    ## 整理資料格式    
    df.set_index("index" , inplace=True)
    df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df.index)))

    bt = Backtest(df, KdCross,cash=100000,commission=.002)
    output = bt.run()
    #bt.plot()
    print(output)    
    





