"""
測試不同策略組合的表現
目標：找出產生正報酬的策略組合
"""

import sys
import os
import warnings

# 加入專案根目錄到系統路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc
from backtesting import Backtest, Strategy
from TA.LW_CheckFisher import checkFisher, FisherParams
from TA.LW_CheckBreakout200 import checkBreakout200, Breakout200Params
from TA.LW_CheckGBS22C import checkGBS22C, GBS22CParams
from TA.LW_CheckIchimoku import checkIchimoku, IchimokuParams
import numpy as np

class ComboStrategy(Strategy):
    """測試不同策略組合"""
    
    def __init__(self, broker, data, params):
        super().__init__(broker, data)
        self.fisher_weight = params.fisher_weight
        self.breakout_weight = params.breakout_weight
        self.gbs22c_weight = params.gbs22c_weight
        self.ichimoku_weight = params.ichimoku_weight
        self.threshold = params.threshold  # 總權重門檻
        self.max_holdbars = params.max_holdbars
        self.sl = params.sl
        self.tp = params.tp
        self.dd = params.dd
        
    def init(self):
        # 計算各策略信號
        self.fisher_params = FisherParams()
        self.breakout_params = Breakout200Params()
        self.gbs22c_params = GBS22CParams()
        self.ichimoku_params = IchimokuParams()
        
    def next(self):
        # 已經持倉，檢查退出條件
        if self.position:
            # 檢查持倉時間
            bars_held = len(self.data) - self.entry_bar
            if bars_held >= self.max_holdbars:
                self.position.close()
                return
                
            # 檢查止損/止盈
            current_return = (self.data.Close[-1] / self.entry_price - 1) * 100
            if current_return <= self.sl:
                self.position.close()
                return
            if current_return >= self.tp:
                self.position.close()
                return
            return
            
        # 計算各策略信號
        idx = len(self.data) - 1
        if idx < 50:
            return
            
        # Fisher 信號
        try:
            fisher_result = checkFisher(self.data.df, self.data.sno, 'L', self.fisher_params)
            fisher_signal = fisher_result['bullish'] if fisher_result else False
        except:
            fisher_signal = False
            
        # Breakout200 信號
        try:
            breakout_result = checkBreakout200(self.data.df, self.data.sno, 'L', self.breakout_params)
            breakout_signal = breakout_result['bullish'] if breakout_result else False
        except:
            breakout_signal = False
            
        # GBS22C 信號
        try:
            gbs22c_result = checkGBS22C(self.data.df, self.data.sno, 'L', self.gbs22c_params)
            gbs22c_signal = gbs22c_result['bullish'] if gbs22c_result else False
        except:
            gbs22c_signal = False
            
        # Ichimoku 信號
        try:
            ichimoku_result = checkIchimoku(self.data.df, self.data.sno, 'L', self.ichimoku_params)
            ichimoku_signal = ichimoku_result['bullish'] if ichimoku_result else False
        except:
            ichimoku_signal = False
        
        # 計算總權重
        total_weight = (
            fisher_signal * self.fisher_weight +
            breakout_signal * self.breakout_weight +
            gbs22c_signal * self.gbs22c_weight +
            ichimoku_signal * self.ichimoku_weight
        )
        
        # 檢查是否達到門檻
        if total_weight >= self.threshold:
            self.buy()


class ComboParams:
    def __init__(self, fisher_weight, breakout_weight, gbs22c_weight, ichimoku_weight, threshold, max_holdbars, sl, tp, dd):
        self.fisher_weight = fisher_weight
        self.breakout_weight = breakout_weight
        self.gbs22c_weight = gbs22c_weight
        self.ichimoku_weight = ichimoku_weight
        self.threshold = threshold
        self.max_holdbars = max_holdbars
        self.sl = sl
        self.tp = tp
        self.dd = dd


def test_combo(fisher_w, breakout_w, gbs22c_w, ichimoku_w, threshold, max_holdbars, sl, tp, dd, stype='L'):
    """測試特定組合"""
    params = ComboParams(fisher_w, breakout_w, gbs22c_w, ichimoku_w, threshold, max_holdbars, sl, tp, dd)
    
    try:
        bt = Backtest(cc.getSlist(stype), ComboStrategy, commission=.005, hedge=False, cash=1000000)
        result = bt.run(fisher_weight=fisher_w, breakout_weight=breakout_w, 
                       gbs22c_weight=gbs22c_w, ichimoku_weight=ichimoku_w,
                       threshold=threshold, max_holdbars=max_holdbars, sl=sl, tp=tp, dd=dd)
        
        trades = result._trades
        if len(trades) > 0:
            win_rate = (trades['Return'] > 0).sum() / len(trades) * 100
            avg_return = trades['Return'].mean()
            return {
                'trades': len(trades),
                'win_rate': win_rate,
                'avg_return': avg_return,
                'equity': result['Equity Final [$]']
            }
        else:
            return {'trades': 0, 'win_rate': 0, 'avg_return': 0, 'equity': 0}
    except Exception as e:
        return {'trades': 0, 'win_rate': 0, 'avg_return': 0, 'equity': 0, 'error': str(e)}


def run_combo_test():
    """運行組合測試"""
    results = []
    
    # 測試不同組合權重
    # 每個策略默認權重為1，門檻設為2（表示至少需要2個策略同意）
    test_cases = [
        # 測試2/4共識（門檻2）
        {'fisher_w': 1, 'breakout_w': 1, 'gbs22c_w': 1, 'ichimoku_w': 1, 'threshold': 2, 'name': '2/4共識(門檻2)'},
        # 測試3/4共識（門檻3）
        {'fisher_w': 1, 'breakout_w': 1, 'gbs22c_w': 1, 'ichimoku_w': 1, 'threshold': 3, 'name': '3/4共識(門檻3)'},
        # 測試4/4共識（門檻4）
        {'fisher_w': 1, 'breakout_w': 1, 'gbs22c_w': 1, 'ichimoku_w': 1, 'threshold': 4, 'name': '4/4共識(門檻4)'},
        # 測試2/4但更嚴格的止損止盈
        {'fisher_w': 1, 'breakout_w': 1, 'gbs22c_w': 1, 'ichimoku_w': 1, 'threshold': 2, 'name': '2/4+嚴格止損'},
    ]
    
    # 測試參數組合
    test_params = [
        # (max_holdbars, sl, tp, dd)
        (50, -5.0, 30.0, 3.0),   # 當前參數
        (100, -10.0, 20.0, 5.0), # 原始參數
        (30, -3.0, 15.0, 2.0),   # 極度嚴格
        (75, -8.0, 25.0, 4.0),   # 中等
    ]
    
    print("=" * 80)
    print("策略組合測試")
    print("=" * 80)
    
    for test in test_cases:
        for params in test_params:
            max_holdbars, sl, tp, dd = params
            result = test_combo(
                test['fisher_w'], test['breakout_w'], test['gbs22c_w'], test['ichimoku_w'],
                test['threshold'], max_holdbars, sl, tp, dd, 'L'
            )
            
            results.append({
                'name': test['name'],
                'params': f"hold={max_holdbars}, sl={sl}, tp={tp}, dd={dd}",
                **result
            })
            
            print(f"{test['name']} | hold={max_holdbars}, sl={sl}, tp={tp}, dd={dd}")
            print(f"  交易次數: {result.get('trades', 0)}, 勝率: {result.get('win_rate', 0):.1f}%, 平均報酬: {result.get('avg_return', 0):.2f}%, 資金: {result.get('equity', 0):.0f}")
    
    # 排序並顯示最佳組合
    print("\n" + "=" * 80)
    print("結果排名（按平均報酬率）")
    print("=" * 80)
    
    valid_results = [r for r in results if r.get('trades', 0) >= 100]  # 至少100筆交易
    valid_results.sort(key=lambda x: x.get('avg_return', 0), reverse=True)
    
    for i, r in enumerate(valid_results[:10]):
        print(f"{i+1}. {r['name']} | {r['params']}")
        print(f"   交易: {r.get('trades', 0)}, 勝率: {r.get('win_rate', 0):.1f}%, 報酬: {r.get('avg_return', 0):.2f}%")


if __name__ == '__main__':
    run_combo_test()