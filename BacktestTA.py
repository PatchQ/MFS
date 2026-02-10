import pandas as pd
import numpy as np
import time as t
import os
from tqdm import tqdm
from backtesting import Backtest, Strategy

OUTPATH = "../SData/P_YFData/" 

class run(Strategy):

    signal = 'BOSSB'
    stype = "L"
    
    def init(self):        
        return

    def next(self):

        if self.signal in self.data.df.columns:                 
            if self.data[self.signal][-1]:
                self.buy(size=200000)
                #print(self.data.index[-1], self.trades, self.position.pl_pct , self.position.size)
            
            if self.position:
                if self.position.pl_pct < -10.0:
                    self.position.close()
                    #print(self.data.index[-1], self.trades, self.position.pl_pct , self.position.size)
                    
                if self.position.pl_pct > 20.0:
                    self.position.close()
                    #print(self.data.index[-1], self.trades, self.position.pl_pct , self.position.size)



def processBT(signal, stype):

    results_dict = {
        'sno': [],
        'returns': [],
        'sharpe_ratios': [],
        'max_drawdowns': [],
        'trades_counts': [],
        'win_rates': []    
    }

    snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(OUTPATH+"/"+stype)))
    snolist = snolist[:]

    for sno in tqdm(snolist):
        
        df = pd.read_csv(OUTPATH+"/"+stype+"/"+sno+".csv")
        
        df.set_index("index" , inplace=True)
        df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df.index)))

        bt = Backtest(
            df, run, cash=1000000,
            commission=0.002,
            margin=1.0,  #margin = 0.02 (1/50=0.02) 50倍槓杆
            trade_on_close=True, 
            hedging=False,
            exclusive_orders=True,
            finalize_trades=True
        )

        output = bt.run(signal=signal, stype=stype)
        
        # 收集主要指標
        if output['Return [%]']!=0:
            results_dict['sno'].append(sno)
            results_dict['returns'].append(output['Return [%]'])
            results_dict['sharpe_ratios'].append(output['Sharpe Ratio'])
            results_dict['max_drawdowns'].append(output['Max. Drawdown [%]'])
            results_dict['trades_counts'].append(output['# Trades'])
            results_dict['win_rates'].append(output['Win Rate [%]'])
        
    resultdf = pd.DataFrame(results_dict)
    resultdf.to_csv("Data/BT_"+stype+"_"+signal+".csv",index=False)

    # 計算總體統計
    print("\n=== 整體回測統計 ===")
    print(f"平均報酬率: {np.mean(resultdf['returns']):.2f}%")
    print(f"報酬率標準差: {np.std(resultdf['returns']):.2f}%")
    print(f"平均夏普比率: {np.mean(resultdf['sharpe_ratios']):.2f}")
    print(f"平均最大回撤: {np.mean(resultdf['max_drawdowns']):.2f}%")
    print(f"總交易次數: {sum(resultdf['trades_counts'])}")
    print(f"平均勝率: {np.mean(resultdf['win_rates']):.2f}%") 



if __name__ == '__main__':

    start = t.perf_counter()
    
    #processBT("BOSSB", "L")
    #processBT("BOSSB", "M")

    #processBT("HHHL", "L")
    #processBT("HHHL", "M")

    #processBT("VCP", "L")
    #processBT("VCP", "M")

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')


