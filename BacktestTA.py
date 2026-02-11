import pandas as pd
import numpy as np
import time as t
import os
from tqdm import tqdm
from backtesting import Backtest, Strategy

#OUTPATH = "../SData/P_YFData/" 
OUTPATH = "../SData/FP_YFData/"

class run(Strategy):

    signal = 'BOSSB'
    stype = "L"    

    max_holdbars = 100  # 最大持倉K線數
    sl = -10.0      # 止損百分比
    tp = 20.0    # 止盈百分比    

    
    def init(self):
        self.highest_profit = 0
        self.holdingbars = 0
        self.ishold = False

    def next(self):

        if self.signal in self.data.df.columns:                 
            if self.data[self.signal][-1]:
                self.buy()
                self.ishold = True
                self.holdingbars = 0                
                self.highest_profit = 0
                #print(self.data.index[-1], self.trades, self.position.pl_pct , self.position.size)
            
            if self.position:
                if self.ishold:
                    self.holdingbars += 1

                # 條件1：持倉時間止損
                if self.holdingbars >= self.max_holdbars:
                    self.position.close()
                    self.is_holding = False
                    self.holding_bars = 0                    
                    #print(self.data.index[-1], self.trades, self.position.pl_pct , self.position.size)
                    return

                 # 條件2：百分比止損
                current_pl = self.position.pl_pct

                if current_pl < self.sl:
                    self.position.close()
                    self.is_holding = False
                    self.holding_bars = 0                    
                    #print(self.data.index[-1], self.trades, self.position.pl_pct , self.position.size)
                    return
                    
                # 條件3：百分比止盈
                if current_pl > self.tp:
                    self.position.close()
                    self.is_holding = False
                    self.holding_bars = 0                    
                    #print(self.data.index[-1], self.trades, self.position.pl_pct , self.position.size)
                    return
                
                if self.signal=="BOSSB":
                    # 條件4：追蹤止損（從最高點回撤5%）
                    self.highest_profit = max(self.highest_profit, current_pl)

                    if self.highest_profit > 5.0 and current_pl < (self.highest_profit - 5.0):                    
                        self.position.close()
                        self.is_holding = False
                        self.holding_bars = 0                    
                        #print(self.data.index[-1], self.trades, self.position.pl_pct , self.position.size)
                        return



def processBT(signal, stype):

    results_dict = {
        'sno': [],
        'returns': [],
        'final': [],
        'peak': [],
        'trades_counts': [],
        'win_rates': [],

        'RR': [],
        'SQN': [],
        'sharpe_ratios': [],
        'sortino_ratios': [],
        'calmar_ratios': [],

        'avg_trade': [],
        'best_trade': [],
        'worst_trade': [],
        'max_tradeday': [],
        'avg_tradeday': [],
        'max_drawdowns': [],
        'avg_drawdowns': [],
        'max_drawdownday': [],
        'avg_drawdownday': [],

        'buy_hold_return': [],
        'ann_return': [],
        'volatility': []
    }

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
                exclusive_orders=True, #確保同時只有一個訂單
                finalize_trades=True  #回測結束時平倉
            )

            output = bt.run(signal=signal, stype=stype)
            bt.plot(filename=f'{OUTPATH}/BT/{signal}/{sno}_{signal}.html',open_browser=False)

            # 優化持倉時間參數
            # optimization = bt.optimize(
            #     max_holdbars=range(50, 80, 100),
            #     sl=[-5, -8, -10, -12],
            #     tp=[10, 15, 20, 25],
            #     maximize='Sharpe Ratio',
            #     constraint=lambda p: p.tp > abs(p.sl)
            # )
            # print("最佳參數:", optimization._strategy)

            
            # 收集主要指標
            if output['Return [%]']!=0:
                results_dict['sno'].append(sno)

                results_dict['returns'].append(output['Return [%]']) #總收益率
                results_dict['final'].append(output['Equity Final [$]']) #最終淨值
                results_dict['peak'].append(output['Equity Peak [$]']) #最高淨值
                results_dict['trades_counts'].append(output['# Trades'])
                results_dict['win_rates'].append(output['Win Rate [%]'])

                results_dict['RR'].append(output['Profit Factor']) #盈虧比
                results_dict['SQN'].append(output['SQN']) #策略表現綜合評分
                results_dict['sharpe_ratios'].append(output['Sharpe Ratio']) #夏普比率(風險調整收益)
                results_dict['sortino_ratios'].append(output['Sortino Ratio']) #索提諾比率(下行風調整收益)
                results_dict['calmar_ratios'].append(output['Calmar Ratio']) #卡爾瑪比率(收益與最大回撤之比)

                results_dict['avg_trade'].append(output['Avg. Trade [%]'])
                results_dict['best_trade'].append(output['Best Trade [%]'])
                results_dict['worst_trade'].append(output['Worst Trade [%]'])
                results_dict['max_tradeday'].append(output['Max. Trade Duration'])
                results_dict['avg_tradeday'].append(output['Avg. Trade Duration'])
                results_dict['max_drawdowns'].append(output['Max. Drawdown [%]'])
                results_dict['avg_drawdowns'].append(output['Avg. Drawdown [%]'])
                results_dict['max_drawdownday'].append(output['Max. Drawdown Duration'])
                results_dict['avg_drawdownday'].append(output['Avg. Drawdown Duration'])        

                results_dict['buy_hold_return'].append(output['Buy & Hold Return [%]']) #買入持有策略收益率
                results_dict['ann_return'].append(output['Return (Ann.) [%]']) #年化收益率
                results_dict['volatility'].append(output['Volatility (Ann.) [%]']) #年化波動率
                
        
    resultdf = pd.DataFrame(results_dict)
    resultdf.to_csv(f'{OUTPATH}/BT/BT_{stype}_{signal}.csv',index=False)

    # 計算總體統計
    print("\n=== 整體回測統計 ===")
    print(f"平均報酬率: {np.mean(resultdf['returns']):.2f}%")
    print(f"報酬率標準差: {np.std(resultdf['returns']):.2f}%")
    print(f"平均最佳收益: {np.mean(resultdf['best_trade']):.2f}%")
    print(f"平均盈虧比: {np.mean(resultdf['RR']):.2f}")
    print(f"平均夏普比率: {np.mean(resultdf['sharpe_ratios']):.2f}")
    print(f"平均最大回撤: {np.mean(resultdf['max_drawdowns']):.2f}%")
    print(f"總交易次數: {sum(resultdf['trades_counts'])}")
    print(f"平均勝率: {np.mean(resultdf['win_rates']):.2f}%") 



if __name__ == '__main__':

    start = t.perf_counter()
    
    #processBT("BOSSB", "L")
    #processBT("BOSSB", "M")

    processBT("HHHL", "L")
    #processBT("HHHL", "M")

    #processBT("VCP", "L")
    #processBT("VCP", "M")

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')


