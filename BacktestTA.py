import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from backtesting import Backtest, Strategy


class run(Strategy):
    
    def init(self):
        return

    def next(self):
        if "BOSSB" in df.columns:

            buy_condition = self.data.BOSSB
            #buy_condition = self.data.HHHL
            #buy_condition = self.data.VCP

            if buy_condition:
                self.buy(size=200000)
                #print(self.data.index[-1], self.trades, self.position.pl_pct , self.position.size)
            
            if self.position:
                if self.position.pl_pct < -10.0:
                    self.position.close()
                    #print(self.data.index[-1], self.trades, self.position.pl_pct , self.position.size)
                    
                if self.position.pl_pct > 20.0:
                    self.position.close()
                    #print(self.data.index[-1], self.trades, self.position.pl_pct , self.position.size)


results_dict = {
    'sno': [],
    'returns': [],
    'sharpe_ratios': [],
    'max_drawdowns': [],
    'trades_counts': [],
    'win_rates': []    
}



stype="L"
OUTPATH = "../SData/P_YFData/" 
snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(OUTPATH+"/"+stype)))
snolist = snolist[:]


for sno in tqdm(snolist):
    
    df = pd.read_csv(OUTPATH+"/"+stype+"/"+sno+".csv")
    
    df.set_index("index" , inplace=True)
    df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df.index)))

    bt = Backtest(df, run,cash=1000000,commission=.002, margin=1) #margin = 0.02 (1/50=0.02) 50倍槓杆
    output = bt.run()
    
    # 收集主要指標
    if output['Return [%]']!=0:
        results_dict['sno'].append(sno)
        results_dict['returns'].append(output['Return [%]'])
        results_dict['sharpe_ratios'].append(output['Sharpe Ratio'])
        results_dict['max_drawdowns'].append(output['Max. Drawdown [%]'])
        results_dict['trades_counts'].append(output['# Trades'])
        results_dict['win_rates'].append(output['Win Rate [%]'])
    
resultdf = pd.DataFrame(results_dict)
resultdf.to_csv("Data/BT_"+stype+"_VCP.csv",index=False)

# 計算總體統計
print("\n=== 整體回測統計 ===")
print(f"平均報酬率: {np.mean(resultdf['returns']):.2f}%")
print(f"報酬率標準差: {np.std(resultdf['returns']):.2f}%")
print(f"平均夏普比率: {np.mean(resultdf['sharpe_ratios']):.2f}")
print(f"平均最大回撤: {np.mean(resultdf['max_drawdowns']):.2f}%")
print(f"總交易次數: {sum(resultdf['trades_counts'])}")
print(f"平均勝率: {np.mean(resultdf['win_rates']):.2f}%") 






