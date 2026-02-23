import pandas as pd
import numpy as np
import time as t
import os
from tqdm import tqdm
from backtesting import Backtest, Strategy
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


OUTPATH = "../SData/P_YFData/" 
#OUTPATH = "../SData/FP_YFData/"

class run(Strategy):

    signal=""
    stype=""
    max_holdbars=0
    sl=0
    tp=0
    dd=0
    
    def init(self):
        self.highest_profit = 0
        self.holdingbars = 0
        self.ishold = False
        self.tp2_price = 0
        self.cl_price = 0

    def next(self):

        if self.signal in self.data.df.columns:                 
            if self.data[self.signal][-1] :#& self.data.EMA1:
                #price = self.data.Close[-1]
                #bsize = int(5000 / (price * 0.10))
                self.buy()

                self.ishold = True
                self.holdingbars = 0                
                self.highest_profit = 0
                self.tp2_price = self.data.tp2_price[-1]
                self.cl_price = self.data.cl_price[-1]
                #print(self.data.index[-1], self.trades, self.position.pl_pct , self.position.size)
            
            if self.position:
                current_pl = self.position.pl_pct

                if self.ishold:
                    self.holdingbars += 1

                # 條件1：持倉時間止損
                if self.holdingbars >= self.max_holdbars:
                    self.position.close()
                    self.is_holding = False
                    self.holding_bars = 0                    
                    #print(self.data.index[-1], self.trades, self.position.pl_pct , self.position.size)
                    return
                

                 # 條件2：價格止損/止盈        
                if self.signal == "BOSSB":

                    if self.data.Close[-1] < self.cl_price:
                        self.position.close()
                        self.is_holding = False
                        self.holding_bars = 0                    
                        #print(self.data.index[-1], self.trades, self.position.pl_pct , self.position.size)
                        return
                    
                    if self.data.Close[-1] > self.tp2_price:
                        self.position.close()
                        self.is_holding = False
                        self.holding_bars = 0                    
                        #print(self.data.index[-1], self.trades, self.position.pl_pct , self.position.size)
                        return
                    
                else:
                    # 條件2：百分比止損/止盈      
                    if current_pl < self.sl:
                        self.position.close()
                        self.is_holding = False
                        self.holding_bars = 0                    
                        #print(self.data.index[-1], self.trades, self.position.pl_pct , self.position.size)
                        return
                    
                    if current_pl > self.tp:
                        self.position.close()
                        self.is_holding = False
                        self.holding_bars = 0                    
                        #print(self.data.index[-1], self.trades, self.position.pl_pct , self.position.size)
                        return

                                
                # 條件3：追蹤止損（從最高點回撤N%）
                self.highest_profit = max(self.highest_profit, current_pl)

                if self.highest_profit > self.dd and current_pl < (self.highest_profit - self.dd):                    
                    self.position.close()
                    self.is_holding = False
                    self.holding_bars = 0                    
                    #print(self.data.index[-1], self.trades, self.position.pl_pct , self.position.size)
                    return


def processBT(signal, stype, max_holdbars, sl, tp, dd):

    tempdf = pd.DataFrame()
    resultdf = pd.DataFrame()

    snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(OUTPATH+"/"+stype)))
    snolist = snolist[:]

    for sno in tqdm(snolist):
        
        df = pd.read_csv(OUTPATH+"/"+stype+"/"+sno+".csv")

        if len(df)!=0:
        
            df.set_index("index" , inplace=True)
            df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df.index)))

            bt = Backtest(
                df, run, cash=200000,
                commission=0.002,
                margin=1.0,  #margin = 0.02 (1/50=0.02) 50倍槓杆
                trade_on_close=False, 
                hedging=False,
                exclusive_orders=False #確保同時只有一個訂單
                #finalize_trades=True  #回測結束時平倉
            )

            output = bt.run(signal=signal, stype=stype, max_holdbars=max_holdbars, sl=sl, tp=tp, dd=dd)
            
            if output['# Trades'] != 0:
                bt.plot(filename=f'{OUTPATH}/BT/{signal}/{sno}_{signal}.html',open_browser=False)
                # 收集主要指標
                tempdf['sno'] = sno
                tempdf['returns'] = [output['Return [%]']] #總收益率
                tempdf['final'] = [output['Equity Final [$]']] #最終淨值
                tempdf['peak'] = [output['Equity Peak [$]']] #最高淨值
                tempdf['trades_counts'] = [output['# Trades']] 
                tempdf['win_rates'] = [output['Win Rate [%]']]

                tempdf['RR'] = [output['Profit Factor']] #盈虧比(獲利因子)
                tempdf['SQN'] = [output['SQN']] #策略表現綜合評分
                tempdf['sharpe_ratios'] = [output['Sharpe Ratio']] #夏普比率(風險調整收益)
                tempdf['sortino_ratios'] = [output['Sortino Ratio']] #索提諾比率(下行風調整收益)
                tempdf['calmar_ratios'] = [output['Calmar Ratio']] #卡爾瑪比率(收益與最大回撤之比)
                tempdf['avg_trade'] = [output['Avg. Trade [%]']]
                tempdf['best_trade'] = [output['Best Trade [%]']]
                tempdf['worst_trade'] = [output['Worst Trade [%]']]
                tempdf['max_tradeday'] = [output['Max. Trade Duration']]
                tempdf['avg_tradeday'] = [output['Avg. Trade Duration']]

                tempdf['max_drawdowns'] = [output['Max. Drawdown [%]']]
                tempdf['avg_drawdowns'] = [output['Avg. Drawdown [%]']]
                tempdf['max_drawdownday'] = [output['Max. Drawdown Duration']]
                tempdf['avg_drawdownday'] = [output['Avg. Drawdown Duration']]

                tempdf['buy_hold_return'] = [output['Buy & Hold Return [%]']] #買入持有策略收益率
                tempdf['ann_return'] = [output['Return (Ann.) [%]']] #年化收益率
                tempdf['volatility'] = [output['Volatility (Ann.) [%]']] #年化波動率
                                
                resultdf = pd.concat([resultdf, tempdf], ignore_index=True)
            
    resultdf.to_csv(f'{OUTPATH}/BT/BT_{stype}_{signal}.csv',index=False)

    # 計算總體統計
    print(f"\n=== {signal} : 整體回測統計 ({stype}) ===")
    print(f"平均報酬率: {np.mean(resultdf['returns']):.2f}%")
    print(f"報酬率標準差: {np.std(resultdf['returns']):.2f}%")
    print(f"平均最佳收益: {np.mean(resultdf['best_trade']):.2f}%")
    print(f"平均最差收益: {np.mean(resultdf['worst_trade']):.2f}%")    
    print(f"平均盈虧比: {np.mean(resultdf['RR']):.2f}")
    print(f"平均策略表現綜合評分: {np.mean(resultdf['SQN']):.2f}")
    print(f"平均夏普比率: {np.mean(resultdf['sharpe_ratios']):.2f}")
    print(f"平均索提諾比率: {np.mean(resultdf['sortino_ratios']):.2f}")
    print(f"平均卡爾瑪比率: {np.mean(resultdf['calmar_ratios']):.2f}") 
    print(f"平均交易次數: {np.mean(resultdf['trades_counts'])}")
    print(f"總交易次數: {sum(resultdf['trades_counts'])}")
    print(f"平均勝率: {np.mean(resultdf['win_rates']):.2f}%") 


if __name__ == '__main__':

    max_holdbars = 100  # 最大持倉K線數
    sl = -10.0      # 止損百分比
    tp = 20.0    # 止盈百分比
    dd = 5.0     # 回撤

    start = t.perf_counter()
    
    processBT("DT", "L", max_holdbars, sl, tp, dd)

    #processBT("BOSSB", "L", max_holdbars, sl, tp, dd)
    #processBT("BOSSB", "M", max_holdbars, sl, tp, dd)

    #processBT("HHHL", "L", max_holdbars, sl, tp, dd)
    #processBT("HHHL", "M", max_holdbars, sl, tp, dd)

    #processBT("VCP", "L", max_holdbars, sl, tp, dd)
    #processBT("VCP", "M", max_holdbars, sl, tp, dd)

    finish = t.perf_counter()
    
    print(f'It took {round(finish-start,2)} second(s) to finish.')


