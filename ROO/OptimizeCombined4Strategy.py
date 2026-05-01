"""
Combined4Strategy 參數優化器
目標：找到最佳參數組合
- 交易次數約 3000 筆
- 勝率 55% 以上
- 平均報酬率 10% 以上
"""

import sys
import os
import warnings
import itertools

# 加入專案根目錄到系統路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc
from backtesting import Backtest, Strategy
import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class Combined4StrategyOptimizer(Strategy):
    """可調整參數的組合策略"""
    max_holdbars = 0
    sl = 0.0
    tp = 0.0
    dd = 0.0
    
    # 可優化的參數
    min_bull_count = 2  # 預設最少2個策略同意
    
    def init(self):
        self.fisher_signal = 'FISHER' in self.data.df.columns
        self.breakout200_signal = 'BREAKOUT200' in self.data.df.columns
        self.gbs22c_signal = 'GBS22C' in self.data.df.columns
        self.ichimoku_signal = 'ICHIMOKU' in self.data.df.columns
        
        self.holdingbars = 0
        self.highest_profit_pct = 0.0

    def next(self):
        if not (self.fisher_signal and self.breakout200_signal and self.gbs22c_signal and self.ichimoku_signal):
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
            fisher_bull = self.data['FISHER'][-1] if self.fisher_signal else False
            breakout_bull = self.data['BREAKOUT200'][-1] if self.breakout200_signal else False
            gbs22c_bull = self.data['GBS22C'][-1] if self.gbs22c_signal else False
            ichimoku_bull = self.data['ICHIMOKU'][-1] if self.ichimoku_signal else False
            
            bull_count = sum([fisher_bull, breakout_bull, gbs22c_bull, ichimoku_bull])
            
            # 動態調整進場條件
            if bull_count >= self.min_bull_count:
                sl_price = None
                tp_price = None
                
                if self.sl < 0:
                    sl_price = current_close * (1 + self.sl / 100)
                if self.tp > 0:
                    tp_price = current_close * (1 + self.tp / 100)
                
                self.buy(sl=sl_price, tp=tp_price)
                self.holdingbars = 0
                self.highest_profit_pct = 0.0


def run_single_backtest(sno, stype, min_bull_count, max_holdbars, sl, tp, dd):
    """執行單次回測"""
    file_path = f"{cc.OUTPATH}/{stype}/P_{sno}.csv"
    if not os.path.exists(file_path):
        return None
        
    df = cc.pd.read_csv(file_path)
    if len(df) == 0:
        return None
        
    df.set_index("index", inplace=True)
    df.index = cc.pd.to_datetime(df.index)

    bt = Backtest(
        df, Combined4StrategyOptimizer, cash=200000,
        commission=0.002,
        margin=1.0, 
        trade_on_close=False, 
        hedging=False,
        exclusive_orders=True
    )
    
    output = bt.run(min_bull_count=min_bull_count, max_holdbars=max_holdbars, sl=sl, tp=tp, dd=dd)
    
    if output['# Trades'] == 0:
        return None
        
    return {
        'sno': sno,
        'returns': output['Return [%]'],
        'trades': output['# Trades'],
        'win_rate': output['Win Rate [%]'],
        'RR': output['Profit Factor'],
        'SQN': output['SQN']
    }


def optimize_parameter(min_bull_count, max_holdbars, sl, tp, dd, stype):
    """測試特定參數組合"""
    snolist = [f.replace("P_", "").replace(".csv", "") 
               for f in os.listdir(f"{cc.OUTPATH}/{stype}") if f.startswith('P_')]
    
    results = []
    for sno in snolist:
        result = run_single_backtest(sno, stype, min_bull_count, max_holdbars, sl, tp, dd)
        if result:
            results.append(result)
    
    if not results:
        return None
        
    df = cc.pd.DataFrame(results)
    
    total_trades = df['trades'].sum()
    avg_return = df['returns'].mean()
    avg_winrate = df['win_rates'].mean() if 'win_rates' in df.columns else 0
    avg_RR = df['RR'].mean() if 'RR' in df.columns else 0
    
    return {
        'min_bull_count': min_bull_count,
        'total_trades': total_trades,
        'stocks_traded': len(df),
        'avg_return': avg_return,
        'avg_winrate': avg_winrate,
        'avg_RR': avg_RR
    }


def run_optimization():
    """執行完整優化流程"""
    print("=" * 70)
    print("Combined4Strategy 參數優化")
    print("目標: 交易次數~3000, 勝率>55%, 平均報酬率>10%")
    print("=" * 70)
    
    # 測試不同的 min_bull_count
    bull_count_options = [2, 3, 4]
    
    # 測試不同的止損止盈參數
    sl_options = [-5.0, -8.0, -10.0]
    tp_options = [15.0, 20.0, 30.0]
    
    results = []
    
    # 先測試不同的 bull_count
    print("\n=== 測試不同的進場策略數量 ===")
    for bull_count in bull_count_options:
        print(f"\n測試 bull_count >= {bull_count}...")
        result = optimize_parameter(
            min_bull_count=bull_count,
            max_holdbars=100,
            sl=-8.0,
            tp=20.0,
            dd=5.0,
            stype="L"
        )
        if result:
            results.append(result)
            print(f"  交易次數: {result['total_trades']}, 勝率: {result['avg_winrate']:.1f}%, 報酬率: {result['avg_return']:.2f}%")
    
    # 找出最佳 bull_count
    best_result = None
    for r in results:
        if r['avg_winrate'] >= 55 and r['avg_return'] >= 10 and abs(r['total_trades'] - 3000) < 1000:
            if best_result is None or r['total_trades'] > best_result['total_trades']:
                best_result = r
    
    if best_result:
        print(f"\n最佳 bull_count: {best_result['min_bull_count']}")
        print(f"  交易次數: {best_result['total_trades']}")
        print(f"  勝率: {best_result['avg_winrate']:.1f}%")
        print(f"  報酬率: {best_result['avg_return']:.2f}%")
    else:
        # 找不到完美匹配，找最接近目標的
        print("\n找不到完美匹配，顯示所有結果:")
        for r in results:
            print(f"  bull_count={r['min_bull_count']}: trades={r['total_trades']}, winrate={r['avg_winrate']:.1f}%, return={r['avg_return']:.2f}%")
        
        # 找勝率最高的
        best_result = max(results, key=lambda x: x['avg_winrate'])
        print(f"\n勝率最高的設定: bull_count={best_result['min_bull_count']}")
        print(f"  交易次數: {best_result['total_trades']}")
        print(f"  勝率: {best_result['avg_winrate']:.1f}%")
        print(f"  報酬率: {best_result['avg_return']:.2f}%")
    
    print("\n" + "=" * 70)
    print("優化完成")
    print("=" * 70)


if __name__ == '__main__':
    run_optimization()