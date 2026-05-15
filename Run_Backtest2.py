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
    hsi_data=None
    hsi_trend_filter=False  # 是否啟用HSI趨勢過濾
    
    def init(self):
        self.highest_profit = 0
        self.holdingbars = 0
        self.ishold = False
        self.tp2_price = 0
        self.cl_price = 0
        
        # 載入HSI趨勢數據
        if run.hsi_data is None and run.hsi_trend_filter:
            run.hsi_data = cc.getHSIData()
    
    def next(self):

        if self.signal in self.data.df.columns:
            if self.data[self.signal][-1]:
                # HSI趨勢過濾：只有HSI在均線上方才做多
                allow_buy = True
                if run.hsi_trend_filter and run.hsi_data is not None:
                    current_date = self.data.index[-1].strftime('%Y-%m-%d')
                    try:
                        if current_date in run.hsi_data.index:
                            allow_buy = bool(run.hsi_data.loc[current_date, 'HSI_Uptrend'])
                        else:
                            # 嘗試找最接近的日期
                            hsi_dates = pd.to_datetime(run.hsi_data.index)
                            closest = hsi_dates[hsi_dates <= pd.Timestamp(current_date)]
                            if len(closest) > 0:
                                closest_date = closest[-1].strftime('%Y-%m-%d')
                                allow_buy = bool(run.hsi_data.loc[closest_date, 'HSI_Uptrend'])
                    except:
                        allow_buy = True  # 出錯時默認允許
                
                if allow_buy:
                    self.buy()
                    self.ishold = True
                    self.holdingbars = 0
                    self.highest_profit = 0
                    if self.signal == "BOSSB":
                        self.tp2_price = self.data.tp2_price[-1]
                        self.cl_price = self.data.cl_price[-1]

            if self.position:
                current_pl = self.position.pl_pct

                if self.ishold:
                    self.holdingbars += 1

                # 條件1：持倉時間止損
                if self.holdingbars >= self.max_holdbars:
                    self.position.close()
                    self.is_holding = False
                    self.holding_bars = 0                    
                    return
                

                 # 條件2：價格止損/止盈        
                if self.signal == "BOSSB":

                    if self.data.Close[-1] < self.cl_price:
                        self.position.close()
                        self.is_holding = False
                        self.holding_bars = 0                    
                        return
                    
                    if self.data.Close[-1] > self.tp2_price:
                        self.position.close()
                        self.is_holding = False
                        self.holding_bars = 0                    
                        return
                    
                else:
                    # 條件2：百分比止損/止盈      
                    if current_pl < self.sl:
                        self.position.close()
                        self.is_holding = False
                        self.holding_bars = 0                    
                        return
                    
                    if current_pl > self.tp:
                        self.position.close()
                        self.is_holding = False
                        self.holding_bars = 0                    
                        return

                                
                # 條件3：追蹤止損（從最高點回撤N%）
                if self.dd > 0:
                    self.highest_profit = max(self.highest_profit, current_pl)

                    if self.highest_profit > self.dd and current_pl < (self.highest_profit - self.dd):                    
                        self.position.close()
                        self.is_holding = False
                        self.holding_bars = 0                    
                        return


def runBacktest(sno, stype, signal, max_holdbars, sl, tp, dd, hsi_trend_filter=False):
    
    tempdf = cc.pd.DataFrame()    
        
    df = cc.pd.read_csv(cc.FOUTPATH+"/"+stype+"/"+sno+".csv")

    if len(df)!=0:
    
        df.set_index("index" , inplace=True)
        df = df.set_index(cc.pd.DatetimeIndex(cc.pd.to_datetime(df.index)))

        bt = Backtest(
            df, run, cash=200000,
            commission=0.002,
            margin=1.0,
            trade_on_close=False, 
            hedging=False,
            exclusive_orders=False
        )

        output = bt.run(signal=signal, stype=stype, max_holdbars=max_holdbars, sl=sl, tp=tp, dd=dd, hsi_trend_filter=hsi_trend_filter)

        if output['# Trades'] != 0:

            if cc.IS_WINDOWS:
                 bt.plot(filename=f'{cc.FOUTPATH}/BT/{signal}/{sno}.html',open_browser=False)
                       
            # 收集主要指標               
            tempdf['returns'] = [output['Return [%]']]
            tempdf['sno'] = str(sno).replace('P_','')
            tempdf['final'] = [output['Equity Final [$]']]
            tempdf['peak'] = [output['Equity Peak [$]']]
            tempdf['trades_counts'] = [output['# Trades']] 
            tempdf['win_rates'] = [output['Win Rate [%]']]

            tempdf['RR'] = [output['Profit Factor']]
            tempdf['SQN'] = [output['SQN']]
            tempdf['sharpe_ratios'] = [output['Sharpe Ratio']]
            tempdf['sortino_ratios'] = [output['Sortino Ratio']]
            tempdf['calmar_ratios'] = [output['Calmar Ratio']]
            tempdf['avg_trade'] = [output['Avg. Trade [%]']]
            tempdf['best_trade'] = [output['Best Trade [%]']]
            tempdf['worst_trade'] = [output['Worst Trade [%]']]
            tempdf['max_tradeday'] = [output['Max. Trade Duration']]
            tempdf['avg_tradeday'] = [output['Avg. Trade Duration']]

            tempdf['max_drawdowns'] = [output['Max. Drawdown [%]']]
            tempdf['avg_drawdowns'] = [output['Avg. Drawdown [%]']]
            tempdf['max_drawdownday'] = [output['Max. Drawdown Duration']]
            tempdf['avg_drawdownday'] = [output['Avg. Drawdown Duration']]

            tempdf['buy_hold_return'] = [output['Buy & Hold Return [%]']]
            tempdf['ann_return'] = [output['Return (Ann.) [%]']]
            tempdf['volatility'] = [output['Volatility (Ann.) [%]']]

    return tempdf                         


def processBT(stype, signal, max_holdbars, sl, tp, dd, hsi_trend_filter=False):

    resultdf = cc.pd.DataFrame()

    snolist = list(map(lambda s: s.replace(".csv", ""), cc.os.listdir(cc.FOUTPATH+"/"+stype)))
    SLIST = cc.pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST.assign(signal=signal+"")
    SLIST = SLIST.assign(max_holdbars=max_holdbars)
    SLIST = SLIST.assign(sl=sl)
    SLIST = SLIST.assign(tp=tp)
    SLIST = SLIST.assign(dd=dd)
    SLIST = SLIST.assign(hsi_trend_filter=hsi_trend_filter)
    SLIST = SLIST[:]
    
    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        for tempdf in cc.tqdm(executor.map(runBacktest,SLIST["sno"],SLIST["stype"],SLIST["signal"],SLIST["max_holdbars"],
                                        SLIST["sl"],SLIST["tp"],SLIST["dd"],SLIST["hsi_trend_filter"],chunksize=1),total=len(SLIST)):            
            if len(tempdf)>0:
                resultdf = cc.pd.concat([tempdf, resultdf], ignore_index=True)
    
    resultdf.to_csv(f'{cc.FOUTPATH}/BT/BT_{stype}_{signal}_hsi{int(hsi_trend_filter)}.csv', index=False)

    if len(resultdf)>0:
        # 計算總體統計
        print(f"\n=== {signal} : 整體回測統計 ({stype}) HSI過濾={hsi_trend_filter} ===")
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
    sl = 3.0        # 止損百分比（窄止損）
    tp = 3.0        # 止盈百分比（1:1快進快出）
    dd = 0.0        # 回撤
    hsi_trend_filter = False  # 關閉HSI過濾

    start = cc.t.perf_counter()

    # for modelname in cc.MODELLIST:
    #     processBT("L", modelname, max_holdbars, sl, tp, dd)
    #     processBT("M", modelname, max_holdbars, sl, tp, dd)

    processBT("H", "ICHIMOKU", max_holdbars, sl, tp, dd, hsi_trend_filter)

    
    finish = cc.t.perf_counter()
    
    print(f'It took {round(finish-start,2)} second(s) to finish.')

