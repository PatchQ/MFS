import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  

from backtesting import Backtest, Strategy
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

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
                if self.dd > 0:
                    self.highest_profit = max(self.highest_profit, current_pl)

                    if self.highest_profit > self.dd and current_pl < (self.highest_profit - self.dd):                    
                        self.position.close()
                        self.is_holding = False
                        self.holding_bars = 0                    
                        #print(self.data.index[-1], self.trades, self.position.pl_pct , self.position.size)
                        return


def runBacktest(sno, stype, signal, max_holdbars, sl, tp, dd):
    
    tempdf = cc.pd.DataFrame()    
        
    df = cc.pd.read_csv(cc.OUTPATH+"/"+stype+"/"+sno+".csv")
    #df = df.loc[df["index"]>"2024-12-31"]        

    if len(df)!=0:
    
        df.set_index("index" , inplace=True)
        df = df.set_index(cc.pd.DatetimeIndex(cc.pd.to_datetime(df.index)))

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

            if cc.IS_WINDOWS:
                 bt.plot(filename=f'{cc.OUTPATH}/BT/{signal}/{sno}.html',open_browser=False)
                        
            # 收集主要指標               
            tempdf['returns'] = [output['Return [%]']] #總收益率
            tempdf['sno'] = str(sno).replace('P_','')
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

    return tempdf  
                            

def processBT(stype, signal, max_holdbars, sl, tp, dd):

    resultdf = cc.pd.DataFrame()

    snolist = list(map(lambda s: s.replace(".csv", ""), cc.os.listdir(cc.OUTPATH+"/"+stype)))
    SLIST = cc.pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST.assign(signal=signal+"")
    SLIST = SLIST.assign(max_holdbars=max_holdbars)
    SLIST = SLIST.assign(sl=sl)
    SLIST = SLIST.assign(tp=tp)
    SLIST = SLIST.assign(dd=dd)    
    SLIST = SLIST[:]
    
    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        for tempdf in cc.tqdm(executor.map(runBacktest,SLIST["sno"],SLIST["stype"],SLIST["signal"],SLIST["max_holdbars"],
                                        SLIST["sl"],SLIST["tp"],SLIST["dd"],chunksize=1),total=len(SLIST)):            
            #tempdf = tempdf.dropna(axis=1, how="all")
            #print(tempdf)
            if len(tempdf)>0:
                resultdf = cc.pd.concat([tempdf, resultdf], ignore_index=True)
    
    resultdf.to_csv(f'{cc.OUTPATH}/BT/BT_{stype}_{signal}.csv', index=False)

    if len(resultdf)>0:
        # 計算總體統計
        print(f"\n=== {signal} : 整體回測統計 ({stype}) ===")
        print(f"平均報酬率: {cc.np.mean(resultdf['returns']):.2f}%")
        print(f"報酬率標準差: {cc.np.std(resultdf['returns']):.2f}%")
        print(f"平均最佳收益: {cc.np.mean(resultdf['best_trade']):.2f}%")
        print(f"平均最差收益: {cc.np.mean(resultdf['worst_trade']):.2f}%")    
        print(f"平均盈虧比: {cc.np.mean(resultdf['RR']):.2f}")
        print(f"平均策略表現綜合評分: {cc.np.mean(resultdf['SQN']):.2f}")
        print(f"平均夏普比率: {cc.np.mean(resultdf['sharpe_ratios']):.2f}")
        print(f"平均索提諾比率: {cc.np.mean(resultdf['sortino_ratios']):.2f}")
        print(f"平均卡爾瑪比率: {cc.np.mean(resultdf['calmar_ratios']):.2f}") 
        print(f"平均交易次數: {cc.np.mean(resultdf['trades_counts'])}")
        print(f"總交易次數: {sum(resultdf['trades_counts'])}")
        print(f"平均勝率: {cc.np.mean(resultdf['win_rates']):.2f}%") 



if __name__ == '__main__':

    max_holdbars = 100  # 最大持倉K線數
    sl = -10.0      # 止損百分比
    tp = 20.0    # 止盈百分比
    dd = 0.0     # 回撤

    start = cc.t.perf_counter()

    # for modelname in cc.MODELLIST:
    #     processBT("L", modelname, max_holdbars, sl, tp, dd)
    #     processBT("M", modelname, max_holdbars, sl, tp, dd)

    for taname in cc.TALIST:
        processBT("L", taname, max_holdbars, sl, tp, dd)
        processBT("M", taname, max_holdbars, sl, tp, dd)

    
    finish = cc.t.perf_counter()
    
    print(f'It took {round(finish-start,2)} second(s) to finish.')


