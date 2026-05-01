"""
Fisher + Ichimoku 組合策略回測
只使用 Fisher 和 Ichimoku 兩個策略，只有當兩者同時發出買入信號時才進場
"""

import sys
import os
import warnings

# 加入專案根目錄到系統路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc
from backtesting import Backtest, Strategy
import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class FisherIchimokuStrategy(Strategy):
    """
    組合策略：只有當 Fisher 和 Ichimoku 兩個策略同時發出買入信號時才進場
    """
    # --- 宣告策略參數 ---
    max_holdbars = 0
    sl = 0.0     # 預期傳入百分比，例如 -10.0 代表 -10%
    tp = 0.0     # 預期傳入百分比，例如 20.0 代表 20%
    dd = 0.0     # 預期傳入百分比，例如 5.0 代表回撤 5%

    def init(self):
        # --- 初始化階段：檢查 Fisher 和 Ichimoku 的信號欄位 ---
        self.fisher_signal = 'FISHER' in self.data.df.columns
        self.ichimoku_signal = 'ICHIMOKU' in self.data.df.columns
        
        # 初始化自定義追蹤狀態
        self.holdingbars = 0
        self.highest_profit_pct = 0.0

    def next(self):
        # 如果資料中沒有我們指定的訊號欄位，直接跳過
        if not (self.fisher_signal and self.ichimoku_signal):
            return

        current_close = self.data.Close[-1]

        # --- 已持倉狀態下的平倉邏輯 ---
        if self.position:
            self.holdingbars += 1
            # 將 backtesting 的小數轉為百分比 (例如 0.05 轉為 5.0)，方便與輸入的參數比較
            current_pl_pct = self.position.pl_pct * 100 

            # 條件 A：持倉時間到達上限 (Time-stop)
            if self.max_holdbars > 0 and self.holdingbars >= self.max_holdbars:
                self.position.close()
                self.holdingbars = 0
                return
            
            # 條件 B：追蹤止損 (Trailing Stop - 回撤超過 dd%)
            if self.dd > 0:
                self.highest_profit_pct = max(self.highest_profit_pct, current_pl_pct)
                # 如果最高獲利已經超過我們設定的回撤值，且當前獲利從最高點跌落超過 dd
                if self.highest_profit_pct > self.dd and current_pl_pct < (self.highest_profit_pct - self.dd):
                    self.position.close()
                    self.holdingbars = 0
                    return

        # --- 空倉狀態下的進場邏輯 ---
        else:
            fisher_bull = self.data['FISHER'][-1] if self.fisher_signal else False
            ichimoku_bull = self.data['ICHIMOKU'][-1] if self.ichimoku_signal else False
            
            # 只有當 Fisher 和 Ichimoku 同時發出買入信號時才進場
            if fisher_bull and ichimoku_bull:
                # 計算精確的止損(sl)與止盈(tp)「絕對價格」
                sl_price = None
                tp_price = None
                
                if self.sl < 0:
                    sl_price = current_close * (1 + self.sl / 100) # 計算止損價
                if self.tp > 0:
                    tp_price = current_close * (1 + self.tp / 100) # 計算止盈價
                
                # 執行買入
                self.buy(sl=sl_price, tp=tp_price)
                
                # 重置計算變數
                self.holdingbars = 0
                self.highest_profit_pct = 0.0


def runBacktest_FisherIchimoku(sno, stype, max_holdbars, sl, tp, dd):
    tempdf = cc.pd.DataFrame()    
    
    # 資料路徑 - 從 OUTPATH 讀取處理後的資料
    file_path = f"{cc.OUTPATH}/{stype}/P_{sno}.csv"
    if not os.path.exists(file_path):
        return tempdf
        
    df = cc.pd.read_csv(file_path)

    if len(df) != 0:
        df.set_index("index" , inplace=True)
        df.index = cc.pd.to_datetime(df.index)
        
        # Debug: check columns
        has_fisher = 'FISHER' in df.columns
        has_ichimoku = 'ICHIMOKU' in df.columns
        fisher_count = df['FISHER'].sum() if has_fisher else 0
        ichimoku_count = df['ICHIMOKU'].sum() if has_ichimoku else 0
        both_count = ((df['FISHER'] == True) & (df['ICHIMOKU'] == True)).sum() if (has_fisher and has_ichimoku) else 0
        print(f'Stock {sno}: FISHER={fisher_count}, ICHIMOKU={ichimoku_count}, BOTH={both_count}', flush=True)

        # 傳入更新後的 FisherIchimokuStrategy
        bt = Backtest(
            df, FisherIchimokuStrategy, cash=200000,
            commission=0.002,
            margin=1.0, 
            trade_on_close=False, 
            hedging=False,
            exclusive_orders=True
        )

        output = bt.run(max_holdbars=max_holdbars, sl=sl, tp=tp, dd=dd)

        if output['# Trades'] != 0:
            # 確保輸出目錄存在
            bt_dir = f'{cc.OUTPATH}/BT/FisherIchimoku'
            if not os.path.exists(bt_dir):
                os.makedirs(bt_dir, exist_ok=True)
            
            if cc.IS_WINDOWS:
                 bt.plot(filename=f'{bt_dir}/{sno}.html', open_browser=False)
                          
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
            
            # 存檔
            result_file = f'{bt_dir}/{sno}_result.csv'
            tempdf.to_csv(result_file, index=False)
            
    return tempdf


def processBT_FisherIchimoku(stype, max_holdbars, sl, tp, dd):
    resultdf = cc.pd.DataFrame()
    
    print(f'processBT_FisherIchimoku started for {stype}', flush=True)

    # 從 OUTPATH 目錄讀取已處理的股票列表
    snolist = list(map(lambda s: s.replace("P_", "").replace(".csv", ""), 
                       [f for f in cc.os.listdir(cc.OUTPATH+"/"+stype) if f.startswith('P_')]))
    print(f'Found {len(snolist)} stocks', flush=True)
    
    SLIST = cc.pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST.assign(max_holdbars=max_holdbars)
    SLIST = SLIST.assign(sl=sl)
    SLIST = SLIST.assign(tp=tp)
    SLIST = SLIST.assign(dd=dd)    
    
    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        for tempdf in cc.tqdm(executor.map(runBacktest_FisherIchimoku, SLIST["sno"], SLIST["stype"],
                                           SLIST["max_holdbars"], SLIST["sl"], SLIST["tp"], SLIST["dd"],
                                           chunksize=1), total=len(SLIST)):            
            if len(tempdf)>0:
                resultdf = cc.pd.concat([tempdf, resultdf], ignore_index=True)
    
    print(f'Collected {len(resultdf)} results', flush=True)
    
    resultdf.to_csv(f'{cc.OUTPATH}/BT/BT_{stype}_FisherIchimoku.csv', index=False)

    if len(resultdf)>0:
        # 計算總體統計
        print(f"\n=== Fisher + Ichimoku 策略 : 整體回測統計 ({stype}) ===")
        print(f"平均報酬率: {cc.np.mean(resultdf['returns']):.2f}%")
        print(f"報酬率標準差: {cc.np.std(resultdf['returns']):.2f}%")
        print(f"平均最佳收益: {cc.np.mean(resultdf['best_trade']):.2f}%")
        print(f"平均最差收益: {cc.np.mean(resultdf['worst_trade']):.2f}%")
        print(f"平均盈虧比: {cc.np.mean(resultdf['RR']):.2f}")
        print(f"平均策略表現綜合評分: {cc.np.mean(resultdf['SQN']):.2f}")
        print(f"平均夏普比率: {cc.np.mean(resultdf['sharpe_ratios']):.2f}")
        print(f"平均索提諾比率: {cc.np.mean(resultdf['sortino_ratios']):.2f}")
        print(f"平均卡爾瑪比率: {cc.np.mean(resultdf['calmar_ratios']):.2f}")
        print(f"平均交易次數: {cc.np.mean(resultdf['trades_counts']):.1f}")
        print(f"總交易次數: {sum(resultdf['trades_counts'])}")
        print(f"平均勝率: {cc.np.mean(resultdf['win_rates']):.2f}%")
        print(f"總股票數: {len(resultdf)}")
        print(f"進場成功率: {len(resultdf[resultdf['trades_counts'] > 0]) / len(resultdf) * 100:.2f}%")
    
    return resultdf
    # 回測參數 - 測試不同配置
    max_holdbars = 100   # 持倉時間上限
    sl = -10.0          # 止損 -10%
    tp = 20.0          # 止盈 20%
    dd = 5.0           # 回撤 5%
    
    start = cc.t.perf_counter()
    
    print("=" * 60)
    print("Fisher + Ichimoku 策略 Backtest Starting...")
    print("進場條件: Fisher + Ichimoku 兩策略同時買入")
    print("=" * 60)
    
    # 回測組合策略
    print("\nProcessing: FisherIchimoku (L)", flush=True)
    result_l = processBT_FisherIchimoku("L", max_holdbars, sl, tp, dd)
    print(f"Result L: {len(result_l) if result_l is not None else 0} rows", flush=True)
    
    print("\nProcessing: FisherIchimoku (M)")
    processBT_FisherIchimoku("M", max_holdbars, sl, tp, dd)
    
    finish = cc.t.perf_counter()
    
    print(f"\nIt took {round(finish - start, 2)} second(s) to finish.")