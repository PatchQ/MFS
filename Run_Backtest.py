import sys
import os
import warnings

# 加入專案根目錄到系統路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  
from backtesting import Backtest, Strategy

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class ModernStrategy(Strategy):
    # --- 1. 宣告策略參數 (讓 backtesting 庫能自動識別，方便未來做最佳化) ---
    signal = ""
    stype = ""
    max_holdbars = 0
    sl = 0.0     # 預期傳入百分比，例如 -10.0 代表 -10%
    tp = 0.0     # 預期傳入百分比，例如 20.0 代表 20%
    dd = 0.0     # 預期傳入百分比，例如 5.0 代表回撤 5%

    def init(self):
        # --- 2. 初始化階段：綁定資料欄位，提升 next() 的執行效能 ---
        self.has_signal = self.signal in self.data.df.columns
        if self.has_signal:
            # 取得訊號陣列
            self.entry_signal = self.data[self.signal]
            
        # 針對 BOSSB 的特殊判斷
        self.is_bossb = (self.signal == "BOSSB")
        if self.is_bossb:
            self.tp2_price = self.data.tp2_price if 'tp2_price' in self.data.df.columns else None
            self.cl_price = self.data.cl_price if 'cl_price' in self.data.df.columns else None

        # 初始化自定義追蹤狀態
        self.holdingbars = 0
        self.highest_profit_pct = 0.0

    def next(self):
        # 如果資料中沒有我們指定的訊號欄位，直接跳過
        if not self.has_signal:
            return

        current_close = self.data.Close[-1]

        # --- 3. 已持倉狀態下的平倉邏輯 ---
        if self.position:
            self.holdingbars += 1
            # 將 backtesting 的小數轉為百分比 (例如 0.05 轉為 5.0)，方便與輸入的參數比較
            current_pl_pct = self.position.pl_pct * 100 

            # 條件 A：持倉時間到達上限 (Time-stop)
            if self.max_holdbars > 0 and self.holdingbars >= self.max_holdbars:
                self.position.close()
                self.holdingbars = 0
                return
            
            # 條件 B：BOSSB 專用價格止損/止盈
            if self.is_bossb:
                if current_close < self.cl_price[-1] or current_close > self.tp2_price[-1]:
                    self.position.close()
                    self.holdingbars = 0
                    return
            
            # 條件 C：追蹤止損 (Trailing Stop - 回撤超過 dd%)
            if self.dd > 0:
                self.highest_profit_pct = max(self.highest_profit_pct, current_pl_pct)
                # 如果最高獲利已經超過我們設定的回撤值，且當前獲利從最高點跌落超過 dd
                if self.highest_profit_pct > self.dd and current_pl_pct < (self.highest_profit_pct - self.dd):
                    self.position.close()
                    self.holdingbars = 0
                    return

        # --- 4. 空倉狀態下的進場邏輯 ---
        elif self.entry_signal[-1]:
            # 計算精確的止損(sl)與止盈(tp)「絕對價格」
            sl_price = None
            tp_price = None
            
            # 使用內建的 sl/tp 參數進場，能讓回測引擎自動捕捉 K 線內的極值 (High/Low)
            if not self.is_bossb:
                if self.sl < 0:
                    sl_price = current_close * (1 + self.sl / 100) # 計算止損價
                if self.tp > 0:
                    tp_price = current_close * (1 + self.tp / 100) # 計算止盈價
            
            # 執行買入
            self.buy(sl=sl_price, tp=tp_price)
            
            # 重置計算變數
            self.holdingbars = 0
            self.highest_profit_pct = 0.0


def runBacktest(sno, stype, signal, max_holdbars, sl, tp, dd):
    tempdf = cc.pd.DataFrame()    
    
    file_path = f"{cc.OUTPATH}/{stype}/{sno}.csv"
    if not os.path.exists(file_path):
        return tempdf
        
    df = cc.pd.read_csv(file_path)

    if len(df) != 0:
        df.set_index("index" , inplace=True)
        df.index = cc.pd.to_datetime(df.index)

        # 傳入更新後的 ModernStrategy
        bt = Backtest(
            df, ModernStrategy, cash=200000,
            commission=0.002,
            margin=1.0, 
            trade_on_close=False, 
            hedging=False,
            exclusive_orders=True # 改為 True 確保同一時間只有一張單，符合原本邏輯
        )

        output = bt.run(signal=signal, stype=stype, max_holdbars=max_holdbars, sl=sl, tp=tp, dd=dd)

        if output['# Trades'] != 0:
            if cc.IS_WINDOWS:
                 bt.plot(filename=f'{cc.OUTPATH}/BT/{signal}/{sno}.html', open_browser=False)
                        
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


