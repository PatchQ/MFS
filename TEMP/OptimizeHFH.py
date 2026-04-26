"""
HFH 參數優化腳本
使用 Grid Search 方式找尋最佳參數
"""
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  
import numpy as np
import pandas as pd
from itertools import product
from backtesting import Backtest, Strategy
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class HFHStrategy(Strategy):
    """HFH 參數優化用策略"""
    signal = "HFH"
    max_holdbars = 0
    sl = -10.0
    tp = 20.0
    dd = 0.0
    
    def init(self):
        self.entry_signal = self.data.HFH
        self.holdingbars = 0
        self.highest_profit_pct = 0.0
        
    def next(self):
        if not hasattr(self.data, 'HFH') or self.data.HFH is None:
            return
            
        current_close = self.data.Close[-1]
        
        if self.position:
            self.holdingbars += 1
            current_pl_pct = self.position.pl_pct * 100
            
            if self.max_holdbars > 0 and self.holdingbars >= self.max_holdbars:
                self.position.close()
                self.holdingbars = 0
                return
                
            if self.dd > 0:
                self.highest_profit_pct = max(self.highest_profit_pct, current_pl_pct)
                if self.highest_profit_pct > self.dd and current_pl_pct < (self.highest_profit_pct - self.dd):
                    self.position.close()
                    self.holdingbars = 0
                    return
        else:
            if self.data.HFH[-1]:
                sl_price = None
                tp_price = None
                if self.sl < 0:
                    sl_price = current_close * (1 + self.sl / 100)
                if self.tp > 0:
                    tp_price = current_close * (1 + self.tp / 100)
                self.buy(sl=sl_price, tp=tp_price)
                self.holdingbars = 0
                self.highest_profit_pct = 0.0


def run_single_backtest(df, params):
    """對單一參數組合進行回測"""
    try:
        bt = Backtest(df, HFHStrategy, cash=200000, commission=0.002,
                      margin=1.0, trade_on_close=False, hedging=False,
                      exclusive_orders=True)
        
        output = bt.run(sl=params['sl'], tp=params['tp'], dd=params['dd'])
        
        return {
            'return_pct': output['Return [%]'] if output['# Trades'] > 0 else 0,
            'trades': output['# Trades'],
            'win_rate': output['Win Rate [%]'] if output['# Trades'] > 0 else 0,
            'profit_factor': output['Profit Factor'] if output['# Trades'] > 0 else 0,
            'sqn': output['SQN'] if output['# Trades'] > 0 else 0,
            'max_dd': output['Max. Drawdown [%]'] if output['# Trades'] > 0 else 0,
        }
    except Exception as e:
        return {
            'return_pct': 0, 'trades': 0, 'win_rate': 0,
            'profit_factor': 0, 'sqn': 0, 'max_dd': 0
        }


def optimize_hfh_params(stype="L", sample_count=20):
    """
    HFH 參數優化
    stype: 股票類型 (L/M)
    sample_count: 抽樣股票數量
    """
    print(f"=== HFH 參數優化 ({stype}) ===")
    
    # 讀取股票列表
    snolist = [s.replace(".csv", "") for s in os.listdir(cc.OUTPATH+"/"+stype)]
    sample_stocks = snolist[:min(sample_count, len(snolist))]
    
    # 參數網格
    param_grid = {
        'min_strong_bullish': [2, 3, 4, 5],
        'body_ratio': [0.4, 0.5, 0.6],
        'min_flat_length': [5, 7, 10],
        'max_flat_pct': [0.08, 0.10, 0.12],
        'min_close_strength': [0.5, 0.6, 0.7],
        'min_volume_ratio': [1.0, 1.2, 1.5],
    }
    
    results = []
    total_combinations = 1
    for v in param_grid.values():
        total_combinations *= len(v)
    
    print(f"總共 {total_combinations} 種參數組合")
    print(f"測試 {len(sample_stocks)} 檔股票...")
    
    # 讀取所有股票數據
    all_dfs = {}
    for sno in sample_stocks:
        try:
            df = pd.read_csv(f"{cc.OUTPATH}/{stype}/P_{sno}.csv", index_col=0)
            df.index = pd.to_datetime(df.index)
            all_dfs[sno] = df
        except:
            continue
    
    # 生成所有參數組合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for combo in product(*param_values):
        params = dict(zip(param_names, combo))
        params['sl'] = -10.0
        params['tp'] = 20.0
        params['dd'] = 0.0
        
        total_return = 0
        total_trades = 0
        total_winrate = 0
        valid_stocks = 0
        
        for sno, df in all_dfs.items():
            try:
                # 先用新參數重新計算 HFH
                df_copy = df.copy()
                df_copy = cc.calHFH(df_copy, 
                                    min_strong_bullish=params['min_strong_bullish'],
                                    body_ratio=params['body_ratio'],
                                    min_flat_length=params['min_flat_length'],
                                    max_flat_pct=params['max_flat_pct'],
                                    min_close_strength=params['min_close_strength'],
                                    min_volume_ratio=params['min_volume_ratio'],
                                    require_consecutive_higher=True,
                                    max_body_deviation=0.30,
                                    min_flat_body_ratio=0.30,
                                    max_upper_wick=0.2,
                                    next_day_confirm=True,
                                    next_day_max_drop=0.03)
                
                result = run_single_backtest(df_copy, params)
                
                if result['trades'] > 0:
                    total_return += result['return_pct']
                    total_trades += result['trades']
                    total_winrate += result['win_rate']
                    valid_stocks += 1
            except:
                continue
        
        if valid_stocks > 0:
            avg_return = total_return / valid_stocks
            avg_trades = total_trades / valid_stocks
            avg_winrate = total_winrate / valid_stocks
            
            results.append({
                'params': params,
                'avg_return': avg_return,
                'avg_trades': avg_trades,
                'avg_winrate': avg_winrate,
                'valid_stocks': valid_stocks
            })
            
            print(f"Params: {params}")
            print(f"  Avg Return: {avg_return:.2f}%, Avg Trades: {avg_trades:.1f}, Win Rate: {avg_winrate:.1f}%")
    
    # 排序結果
    results.sort(key=lambda x: x['avg_return'], reverse=True)
    
    print("\n=== Top 5 參數組合 ===")
    for i, r in enumerate(results[:5]):
        print(f"{i+1}. Return: {r['avg_return']:.2f}%, Trades: {r['avg_trades']:.1f}, WinRate: {r['avg_winrate']:.1f}%")
        print(f"   Params: {r['params']}")
    
    # 保存結果
    result_df = pd.DataFrame([{
        'return': r['avg_return'],
        'trades': r['avg_trades'],
        'winrate': r['avg_winrate'],
        **{k: v for k, v in r['params'].items()}
    } for r in results])
    
    result_df.to_csv(f"{cc.OUTPATH}/HFH_Optimization_Results_{stype}.csv", index=False)
    print(f"\n結果已保存到 {cc.OUTPATH}/HFH_Optimization_Results_{stype}.csv")
    
    return results


if __name__ == '__main__':
    start = cc.t.perf_counter()
    
    print("=" * 60)
    print("HFH 參數優化開始")
    print("=" * 60)
    
    results = optimize_hfh_params("L", sample_count=30)
    
    finish = cc.t.perf_counter()
    print(f"\n總耗時: {round(finish-start, 2)} 秒")
    print("=" * 60)