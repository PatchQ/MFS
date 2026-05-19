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

class ModernStrategy(Strategy):
    # --- 1. 宣告策略參數 ---
    signal = ""
    stype = ""
    max_holdbars = 0
    min_trend_bars = 20  # 趨勢判斷：至少要在均線上多少天
    sl = 0.0
    tp = 0.0
    dd = 0.0

    def init(self):
        # 初始化訊號
        self.has_signal = self.signal in self.data.df.columns
        if self.has_signal:
            self.entry_signal = self.data[self.signal]
        
        # BOSSB 特殊判斷
        self.is_bossb = (self.signal == "BOSSB")
        if self.is_bossb:
            self.tp2_price = self.data.tp2_price if 'tp2_price' in self.data.df.columns else None
            self.cl_price = self.data.cl_price if 'cl_price' in self.data.df.columns else None

        # 初始化自定義追蹤狀態
        self.holdingbars = 0
        self.highest_profit_pct = 0.0
        self._entry_relative_strength = None  # 進場時記錄相對強度
        
        # === 1. 計算 20 日均線 (趨勢持有用) ===
        # 使用 self.data.df 來訪問底層 DataFrame
        self.ema20 = self.data.df['Close'].rolling(20).mean()
        
        # === 3. HSI 過濾器：計算個股 vs HSI 相對強度 ===
        hsi_df = cc.getHSIData()
        if hsi_df is not None:
            # 對齊 HSI 到個股時間軸
            self.hsi_aligned = hsi_df.reindex(self.data.df.index, method='ffill')
            self.hsi_ema20 = self.hsi_aligned['Close'].rolling(20).mean()
            self.hsi_uptrend = self.hsi_aligned['Close'] > self.hsi_ema20
            # === 軟過濾：計算 HSI ATR (20日高低差) ===
            self.hsi_atr = self.hsi_aligned['Close'].rolling(20).std()
        else:
            self.hsi_aligned = None
            self.hsi_uptrend = True  # 沒有 HSI data 就預設允許
            self.hsi_atr = None
        
        # 記錄倉位調整係數（用於事後分析）
        self.position_adjustment_log = []

    def is_strong_uptrend(self):
        """判斷是否處於強勢上升趨勢（用於趨勢持有）"""
        if len(self.data.df) < self.min_trend_bars:
            return False
        
        current_price = self.data.Close[-1]
        ema_value = self.ema20.iloc[-1]
        
        # 計算有多少天在均線之上
        days_above = 0
        for i in range(-1, -self.min_trend_bars - 1, -1):
            if self.data.df['Close'].iloc[i] > self.ema20.iloc[i]:
                days_above += 1
            else:
                break
        
        # 80% 時間在均線上視為強勢
        return days_above >= self.min_trend_bars * 0.8

    def is_market_confirmed(self):
        """判斷大盤是否確認上升（HSI 在 EMA20 之上）"""
        if self.hsi_aligned is None:
            return True  # 沒有 HSI data 就預設允許
        
        current_idx = self.data.df.index[-1]
        if current_idx not in self.hsi_aligned.index:
            return True  # 沒有對應的 HSI 數據就預設允許
        
        # 直接計算當前 bar 的 HSI EMA20
        current_hsi_close = self.hsi_aligned.loc[current_idx, 'Close']
        # 計算最近 20 天的 HSI EMA20
        hsi_series = self.hsi_aligned['Close'].loc[:current_idx]
        ema20_value = hsi_series.ewm(span=20, min_periods=1).mean().iloc[-1]
        
        return current_hsi_close > ema20_value

    def calc_entry_relative_strength(self):
        """方案 B：完全移除 HSI 過濾，不做任何相對強度限制"""
        return True

    def calc_position_size_adjustment(self):
        """
        根據 HSI 信號強度計算倉位調整係數
        hsi_signal = (HSI收盤 - EMA20) / ATR
        position_ratio = 1 + hsi_signal * 0.5
        範圍：0.5 ~ 1.5（最低半倉，最高1.5倍倉）
        """
        if self.hsi_aligned is None or self.hsi_atr is None:
            return 1.0  # 沒有 HSI data 就返回預設值
        
        current_idx = self.data.df.index[-1]
        if current_idx not in self.hsi_aligned.index:
            return 1.0
        
        hsi_close = self.hsi_aligned.loc[current_idx, 'Close']
        hsi_ema = self.hsi_ema20.iloc[-1] if len(self.hsi_ema20) > 0 else hsi_close
        atr = self.hsi_atr.iloc[-1] if len(self.hsi_atr) > 0 else 1.0
        
        # 避免 ATR 為 0 或過小
        if atr <= 0:
            return 1.0
        
        hsi_signal = (hsi_close - hsi_ema) / atr
        position_ratio = 1 + hsi_signal * 0.5
        
        # 限制在 0.5 ~ 1.5 範圍內
        return max(0.5, min(1.5, position_ratio))

    def next(self):
        # 如果資料中沒有指定的訊號欄位，直接跳過
        if not self.has_signal:
            return

        current_close = self.data.Close[-1]

        # --- 已持倉狀態下的平倉邏輯 ---
        if self.position:
            self.holdingbars += 1
            current_pl_pct = self.position.pl_pct * 100 

            # 條件 A：持倉時間到達上限 (Time-stop)
            if self.max_holdbars > 0 and self.holdingbars >= self.max_holdbars:
                # 【優化 1】趨勢持有：強勢上升就繼續持有
                if self.is_strong_uptrend():
                    return  # 繼續持有，不平倉
                else:
                    self.position.close()
                    self.holdingbars = 0
                    return
            
            # 條件 B：BOSSB 專用價格止損/止盈
            if self.is_bossb:
                if current_close < self.cl_price[-1] or current_close > self.tp2_price[-1]:
                    self.position.close()
                    self.holdingbars = 0
                    return
            
            # 條件 C：追蹤止損 (Trailing Stop)
            if self.dd > 0:
                self.highest_profit_pct = max(self.highest_profit_pct, current_pl_pct)
                if self.highest_profit_pct > self.dd and current_pl_pct < (self.highest_profit_pct - self.dd):
                    self.position.close()
                    self.holdingbars = 0
                    return

        # --- 空倉狀態下的進場邏輯 ---
        elif self.data[self.signal][-1]:
            # 【優化 3】HSI 過濾：只有在 HSI 上升且個股強於大盤時才能入場
            if not (self.is_market_confirmed() and self.calc_entry_relative_strength()):
                return
            
            # 計算止損止盈價格
            sl_price = None
            tp_price = None
            
            if not self.is_bossb:
                if self.sl < 0:
                    sl_price = current_close * (1 + self.sl / 100)
                if self.tp > 0:
                    tp_price = current_close * (1 + self.tp / 100)
            
            # 執行買入
            self.buy(sl=sl_price, tp=tp_price)
            
            # 記錄倉位調整係數（軟過濾 - 不減少交易次數，僅調整曝險）
            adjustment = self.calc_position_size_adjustment()
            self.position_adjustment_log.append({
                'date': str(self.data.df.index[-1]),
                'adjustment': adjustment
            })
            
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

        bt = Backtest(
            df, ModernStrategy, cash=200000,
            commission=0.002,
            margin=1.0, 
            trade_on_close=False, 
            hedging=False,
            exclusive_orders=True
        )

        output = bt.run(signal=signal, stype=stype, max_holdbars=max_holdbars, sl=sl, tp=tp, dd=dd)

        if output['# Trades'] != 0:
            if cc.IS_WINDOWS:
                 bt.plot(filename=f'{cc.OUTPATH}/BT/{signal}/{sno}.html', open_browser=False)
                       
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
            if len(tempdf)>0:
                resultdf = cc.pd.concat([tempdf, resultdf], ignore_index=True)
    
    resultdf.to_csv(f'{cc.OUTPATH}/BT/BT_{stype}_{signal}.csv', index=False)

    if len(resultdf)>0:
        print(f"\n=== {signal} : 整體回測統計 ({stype}) ===")
        print(f"平均報酬率: {cc.np.mean(resultdf['returns']):.2f}%")
        print(f"平均勝率: {cc.np.mean(resultdf['win_rates']):.2f}%")
        print(f"平均盈虧比: {cc.np.mean(resultdf['RR']):.2f}")
        print(f"平均策略評分: {cc.np.mean(resultdf['SQN']):.2f}")
        print(f"總交易次數: {sum(resultdf['trades_counts'])}")
    
    return resultdf


def run_param_sweep(stype, signal, param_name, param_values, base_params):
    results = []
    
    for val in param_values:
        params = base_params.copy()
        params[param_name] = val
        
        resultdf = processBT(
            stype=stype,
            signal=signal,
            max_holdbars=params['max_holdbars'],
            sl=params['sl'],
            tp=params['tp'],
            dd=params['dd']
        )
        
        if len(resultdf) > 0:
            results.append({
                'param_value': val,
                'mean_return': np.mean(resultdf['returns']),
                'mean_sqn': np.mean(resultdf['SQN']),
                'mean_winrate': np.mean(resultdf['win_rates']),
                'total_trades': sum(resultdf['trades_counts'])
            })
    
    return cc.pd.DataFrame(results)


if __name__ == '__main__':
    # 暫存原始路徑
    _orig_path = cc.PATH
    _orig_outpath = cc.OUTPATH

    # 切換到 Full 版本的路徑
    cc.PATH = cc.FPATH
    cc.OUTPATH = cc.FOUTPATH
    
    # 回測參數
    max_holdbars = 100
    sl = -10.0
    tp = 20.0
    dd = 0.0
    
    start = cc.t.perf_counter()
    
    print("=" * 60)
    print("VectorBT Backtest2 Starting...")
    print("=" * 60)
    print(f"Using FPATH: {cc.FPATH}")
    print(f"Using FOUTPATH: {cc.OUTPATH}")
    print(f"Trend Holding: enabled (min_trend_bars=20)")
    print(f"HSI Filter: enabled (HSI uptrend + Stock stronger than market)")
    
    for taname in cc.TALIST:
        print(f"\nProcessing: {taname}")
        processBT("L", taname, max_holdbars, sl, tp, dd)
        processBT("M", taname, max_holdbars, sl, tp, dd)
    
    finish = cc.t.perf_counter()
    
    print(f"\nIt took {round(finish - start, 2)} second(s) to finish.")

    # 恢復原始路徑
    cc.PATH = _orig_path
    cc.OUTPATH = _orig_outpath
    print("=" * 60)
    print("VectorBT Backtest2 Completed!")
    print("=" * 60)
