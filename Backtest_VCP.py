import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm
from backtesting import Backtest, Strategy

#get stock excel file from path
stype="L"

#get stock excel file from path
OUTPATH = "../SData/P_YFData/" 
snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(OUTPATH+"/"+stype)))
snolist = snolist[:]


class VCP(Strategy):
    
    def init(self):
        return

    def next(self):
        if self.data.VCP:
            self.buy()
            #print(self.data.index, self.trades, self.position.pl_pct , self.position.size)
        
        if self.position:
            if self.position.pl_pct < -0.1:
                self.position.close()
                #print(self.data.index, self.trades, self.position.pl_pct , self.position.size)
                
            if self.position.pl_pct > 0.5:
                self.position.close()
                #print(self.data.index, self.trades, self.position.pl_pct , self.position.size)

results_dict = {
    'returns': [],
    'sharpe_ratios': [],
    'max_drawdowns': [],
    'trades_counts': [],
    'win_rates': [],
    'details': {}  # 存儲每檔股票的詳細結果
}

for sno in tqdm(snolist):
    
    df = pd.read_csv(OUTPATH+"/"+stype+"/"+sno+".csv")

    ## 整理資料格式    
    df.set_index("index" , inplace=True)
    df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df.index)))

    bt = Backtest(df, VCP,cash=1000000,commission=.002)
    output = bt.run()
    
    # 收集主要指標
    results_dict['returns'].append(output['Return [%]'])
    results_dict['sharpe_ratios'].append(output['Sharpe Ratio'])
    results_dict['max_drawdowns'].append(output['Max. Drawdown [%]'])
    results_dict['trades_counts'].append(output['# Trades'])
    results_dict['win_rates'].append(output['Win Rate [%]'])
    
    # 存儲詳細結果
    results_dict['details'][sno] = output

#resultdf = pd.DataFrame(results_dict)
#resultdf.to_csv("Data/"+stype+"_VCP_"+datetime.now().strftime("%Y%m%d")+".csv",index=False)

# 計算總體統計
print("\n=== 整體回測統計 ===")
print(f"平均報酬率: {np.mean(results_dict['returns']):.2f}%")
print(f"報酬率標準差: {np.std(results_dict['returns']):.2f}%")
print(f"平均夏普比率: {np.mean(results_dict['sharpe_ratios']):.2f}")
print(f"平均最大回撤: {np.mean(results_dict['max_drawdowns']):.2f}%")
print(f"總交易次數: {sum(results_dict['trades_counts'])}")
print(f"平均勝率: {np.mean(results_dict['win_rates']):.2f}%") 
