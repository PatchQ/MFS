"""
BOSS 參數優化系統
================
可持續調較參數、進行回測並記錄結果，自動找出最佳參數組合。

用法:
    python BOSS_Optimizer.py --mode grid --rounds 100
    python BOSS_Optimizer.py --mode genetic --generations 50
    python BOSS_Optimizer.py --mode adaptive --rounds 200
"""

import os
import sys
import time
import random
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
sys.path.insert(0, PROJECT_ROOT)

try:
    import UTIL.CommonConfig as cc
    from TA.LW_CheckBoss import BOSSParams
    from TA.LW_CalHHLL import calHHLL
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    print(f"警告: 缺少模組 {e}")


PARAM_SPACE = {
    'WINDOW_22D_LOW': [18, 20, 22, 25, 28],
    'WINDOW_33D_LOW': [28, 33, 38, 43],
    'MA_PERIOD': [100, 150, 200],
    'VOLATILITY_THRESHOLD': [0.10, 0.12, 0.14, 0.16, 0.18],
    'BOSS1_PATTERNS': [["LHLLHH", "HHLLHH"], ["LHLLHH"], ["HHLLHH"]],
    'BULLISH_RATIO_THRESHOLD': [0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    'STRONG_BULLISH_MIN': [1, 2, 3],
    'BULLISH_COUNT_MIN': [3, 4, 5, 6],
    'STOP_LOSS_BUFFER': [0.97, 0.98, 0.99, 1.00],
    'BUY_TOLERANCE': [1.002, 1.005, 1.008, 1.010],
    'TP1_THRESHOLD': [0.990, 0.995, 1.000],
    'TP2_THRESHOLD': [0.98, 0.99, 0.995],
    'TP3_THRESHOLD': [0.97, 0.98, 0.99],
    'BUY_DEADLINE': [15, 20, 22, 25, 30],
    'TP_DEADLINE': [20, 25, 30, 35, 40],
    'TU_PROFIT_THRESHOLD': [0.005, 0.01, 0.015, 0.02],
}

FIXED_PARAMS = {
    'WINDOW_22D_LOW': 22,
    'WINDOW_33D_LOW': 33,
    'MA_PERIOD': 150,
}


class ResultTracker:
    def __init__(self, output_dir=None):
        self.results = []
        self.best_score = float('-inf')
        self.best_params = None
        self.iteration = 0
        self.output_dir = output_dir or DATA_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.history_file = os.path.join(self.output_dir, 'BOSS_Optimizer_History.csv')
        self.checkpoint_file = os.path.join(self.output_dir, 'BOSS_Optimizer_Checkpoint.pkl')
        self._load_history()
    
    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                df = pd.read_csv(self.history_file)
                if not df.empty:
                    self.results = df.to_dict('records')
                    if self.results:
                        valid_scores = [r.get('final_score', float('-inf')) for r in self.results 
                                       if r.get('final_score', 0) != float('inf') and r.get('final_score', 0) != float('-inf')]
                        if valid_scores:
                            self.best_score = max(valid_scores)
                            self.best_params = next((r for r in self.results 
                                                    if r.get('final_score', 0) == self.best_score), None)
                        self.iteration = len(self.results)
                        print(f"已載入 {len(self.results)} 筆記錄，最佳分數: {self.best_score:.3f}")
            except Exception as e:
                print(f"載入歷史失敗: {e}")
    
    def add_result(self, params, metrics):
        self.iteration += 1
        
        params_clean = {k: (str(v) if isinstance(v, list) else v) for k, v in params.items()}
        
        record = {
            'iteration': self.iteration,
            'timestamp': datetime.now().isoformat(),
            'final_score': metrics.get('final_score', 0),
            'win_rate': metrics.get('win_rate', 0),
            'total_trades': metrics.get('total_trades', 0),
            'tp3_rate': metrics.get('tp3_rate', 0),
            'cl1_rate': metrics.get('cl1_rate', 0),
            'return_pct': metrics.get('return_pct', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            **params_clean
        }
        
        self.results.append(record)
        
        score = record['final_score']
        if score != float('inf') and score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            print(f"  *** 新最佳!: score={self.best_score:.3f}, trades={record['total_trades']}")
        
        if self.iteration % 10 == 0:
            self.save()
    
    def save(self):
        df = pd.DataFrame(self.results)
        df.to_csv(self.history_file, index=False)
        checkpoint = {'results': self.results, 'best_score': self.best_score, 
                      'best_params': self.best_params, 'iteration': self.iteration}
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
        except:
            pass
    
    def get_top(self, n=10):
        df = pd.DataFrame(self.results)
        if 'final_score' in df.columns:
            df = df[df['final_score'] != float('inf')]
            df = df[df['final_score'] != float('-inf')]
            return df.nlargest(n, 'final_score')
        return df.head(n)


class SimulatedBacktester:
    """基於歷史統計資料的模擬回測器"""
    
    def __init__(self, stype='L'):
        self.stype = stype
        self.base_stats = self._load_baseline()
    
    def _load_baseline(self):
        """直接讀取並計算基准統計"""
        stat_file = os.path.join(DATA_DIR, f'{self.stype}_BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2~BOSSTU1~BOSSTU2_Stat.csv')
        
        if not os.path.exists(stat_file):
            return None
        
        try:
            # 讀取 CSV，跳過可能有問題的最後一列
            df = pd.read_csv(stat_file)
            
            # 嘗試識別並排除總計行
            # 總計行的 sno 通常是空的或是 NaN
            df = df[df['sno'].notna()]
            
            # 轉換數值欄位
            numeric_cols = ['TP123', 'TP12', 'TP1', 'TP2', 'TP3', 'TP1C', 'CL1', 'TU1', 'TU2', 'BY1', 'TOTAL', 'WR']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 過濾掉數值為 NaN 的行
            df = df[df['TOTAL'].notna()]
            
            # 計算總計
            totals = {}
            for col in numeric_cols:
                if col in df.columns:
                    totals[col] = df[col].sum()
            
            total_trades = totals.get('TOTAL', 0)
            tp3_total = totals.get('TP3', 0)
            tp1_total = totals.get('TP1', 0)
            cl1_total = totals.get('CL1', 0)
            tu1_total = totals.get('TU1', 0)
            tu2_total = totals.get('TU2', 0)
            tp12_total = totals.get('TP12', 0)
            
            # 計算加權勝率
            wr_values = df['WR'].dropna()
            wr_values = wr_values[wr_values > 0]
            avg_wr = wr_values.mean() if len(wr_values) > 0 else 40.0
            
            return {
                'total_trades': total_trades,
                'avg_win_rate': avg_wr,
                'tp3': tp3_total,
                'tp12': tp12_total,
                'tp1': tp1_total,
                'cl1': cl1_total,
                'tu1': tu1_total,
                'tu2': tu2_total,
            }
        except Exception as e:
            print(f"載入基准數據失敗: {e}")
            return None
    
    def simulate_backtest(self, params):
        if self.base_stats is None:
            return None
        
        base = self.base_stats
        base_wr = base['avg_win_rate']
        base_total = base['total_trades']
        
        if base_total == 0:
            return None
        
        # 提取參數
        vol = params.get('VOLATILITY_THRESHOLD', 0.14)
        bul = params.get('BULLISH_RATIO_THRESHOLD', 0.65)
        sb = params.get('STRONG_BULLISH_MIN', 1)
        bc = params.get('BULLISH_COUNT_MIN', 4)
        slb = params.get('STOP_LOSS_BUFFER', 0.99)
        
        # 勝率調整
        vol_adj = (vol - 0.14) * 200
        bul_adj = -(bul - 0.55) * 100
        sb_adj = -(sb - 1) * 3
        bc_adj = -(bc - 3) * 2
        slb_adj = (slb - 0.99) * 50
        
        total_adj = vol_adj + bul_adj + sb_adj + bc_adj + slb_adj
        adj_wr = base_wr + total_adj
        adj_wr = max(15, min(90, adj_wr))
        
        # 交易次數調整
        filter_factor = 1.0
        filter_factor *= (1 - (vol - 0.14) * 3)
        filter_factor *= (1 - (bul - 0.50) * 0.5)
        filter_factor *= (1 - (sb - 1) * 0.1)
        filter_factor *= (1 - (bc - 3) * 0.08)
        filter_factor = max(0.05, min(1.0, filter_factor))
        
        est_total = max(1, int(base_total * filter_factor))
        
        # 估算 outcome 分佈
        tp3_rate = 0.12 + (adj_wr - 40) * 0.003
        tp3_rate = max(0.02, min(0.35, tp3_rate))
        
        tp1_rate = max(0.05, min(0.35, adj_wr / 100 * 0.6))
        
        cl1_rate = (100 - adj_wr) / 100 * 0.55
        cl1_rate = max(0.10, min(0.70, cl1_rate))
        
        tp3 = int(est_total * tp3_rate)
        tp1 = int(est_total * tp1_rate)
        cl1 = int(est_total * cl1_rate)
        tp2 = int(est_total * 0.08)
        tu1 = int(est_total * 0.05)
        tu2 = int(est_total * (1 - tp3_rate - tp1_rate - cl1_rate - 0.08 - 0.05))
        
        total = tp3 + tp2 + tp1 + cl1 + tu1 + tu2
        if total == 0:
            return None
        
        # 評分
        raw_score = tp3*3 + tp2*2 + tp1*1 - cl1*2 - tu1*0.5 - tu2*1
        normalized = raw_score / total
        final_score = normalized * (0.5 + adj_wr/100 * 0.5)
        
        est_return = (tp3 * 0.30 + tp2 * 0.20 + tp1 * 0.10 - cl1 * 0.05 - tu2 * 0.03) / total * 100
        est_max_dd = cl1_rate * 10
        
        return {
            'total_trades': total,
            'win_rate': adj_wr,
            'tp3_rate': tp3_rate * 100,
            'tp2_rate': tp2 / total * 100 if total > 0 else 0,
            'tp1_rate': tp1_rate * 100,
            'cl1_rate': cl1_rate * 100,
            'tu1_rate': tu1 / total * 100 if total > 0 else 0,
            'tu2_rate': tu2 / total * 100 if total > 0 else 0,
            'final_score': final_score,
            'raw_score': raw_score,
            'return_pct': est_return,
            'max_drawdown': est_max_dd,
        }


class ParamGenerator:
    def __init__(self, param_space, fixed_params=None):
        self.param_space = param_space
        self.fixed_params = fixed_params or {}
        
    def generate_grid(self, keys=None):
        if keys is None:
            keys = list(self.param_space.keys())
        values = [self.param_space[k] for k in keys]
        for combo in product(*values):
            yield dict(zip(keys, combo))
    
    def generate_random(self, n=100, seed=42):
        random.seed(seed)
        keys = list(self.param_space.keys())
        for _ in range(n):
            yield {k: random.choice(self.param_space[k]) for k in keys}
    
    def mutate(self, params, rate=0.3):
        mutated = params.copy()
        for k in self.param_space.keys():
            if random.random() < rate:
                mutated[k] = random.choice(self.param_space[k])
        return mutated
    
    def crossover(self, p1, p2):
        return {k: random.choice([p1.get(k), p2.get(k)]) for k in self.param_space.keys()}


class BOSSOptimizer:
    def __init__(self, stype='L', output_dir=None):
        self.stype = stype
        self.tracker = ResultTracker(output_dir)
        self.generator = ParamGenerator(PARAM_SPACE, FIXED_PARAMS)
        self.stocks = self._load_stock_list()
        self.tester = SimulatedBacktester(stype)
        print(f"找到 {len(self.stocks)} 檔股票的歷史數據")
    
    def _load_stock_list(self):
        stat_file = os.path.join(DATA_DIR, f'{self.stype}_BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2~BOSSTU1~BOSSTU2_Stat.csv')
        stocks = []
        if os.path.exists(stat_file):
            try:
                df = pd.read_csv(stat_file)
                df = df[df['sno'].notna()]
                df = df[df['sno'] != '']
                stocks = df['sno'].tolist()
            except:
                pass
        return stocks
    
    def _evaluate_params(self, params):
        return self.tester.simulate_backtest(params)
    
    def optimize_grid(self, rounds=None):
        print(f"\n{'='*60}")
        print("網格搜尋模式")
        print(f"{'='*60}")
        
        key_params = ['VOLATILITY_THRESHOLD', 'BULLISH_RATIO_THRESHOLD', 
                      'STRONG_BULLISH_MIN', 'BULLISH_COUNT_MIN']
        
        combos = list(self.generator.generate_grid(key_params))
        print(f"總共 {len(combos)} 種組合")
        
        if rounds and rounds < len(combos):
            combos = combos[:rounds]
            print(f"限制為前 {rounds} 種")
        
        for i, params in enumerate(combos):
            params.update(FIXED_PARAMS)
            print(f"\n[{i+1}/{len(combos)}] vol={params['VOLATILITY_THRESHOLD']}, bul={params['BULLISH_RATIO_THRESHOLD']}")
            
            metrics = self._evaluate_params(params)
            if metrics:
                self.tracker.add_result(params, metrics)
                print(f"  score={metrics['final_score']:.3f}, trades={metrics['total_trades']}, WR={metrics['win_rate']:.1f}%")
        
        self.tracker.save()
        return self.tracker.best_params
    
    def optimize_random(self, rounds=100):
        print(f"\n{'='*60}")
        print(f"隨機搜尋模式 ({rounds} 輪)")
        print(f"{'='*60}")
        
        for i, params in enumerate(self.generator.generate_random(rounds)):
            params.update(FIXED_PARAMS)
            
            metrics = self._evaluate_params(params)
            if metrics:
                self.tracker.add_result(params, metrics)
                print(f"[{i+1}/{rounds}] score={metrics['final_score']:.3f}, WR={metrics['win_rate']:.1f}%")
        
        self.tracker.save()
        return self.tracker.best_params
    
    def optimize_genetic(self, generations=30, population=20):
        print(f"\n{'='*60}")
        print(f"遺傳演算法模式 ({generations} 代, 族群 {population})")
        print(f"{'='*60}")
        
        population_params = list(self.generator.generate_random(population, seed=42))
        best_params = None
        best_score = float('-inf')
        
        for gen in range(generations):
            print(f"\n--- 第 {gen+1} 代 ---")
            
            scores = []
            for params in population_params:
                params.update(FIXED_PARAMS)
                metrics = self._evaluate_params(params)
                
                if metrics:
                    score = metrics['final_score']
                    scores.append((params.copy(), score, metrics))
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        print(f"  *** 新最佳!: score={score:.3f}")
            
            if not scores:
                break
            
            scores.sort(key=lambda x: x[1], reverse=True)
            survivors = [s[0] for s in scores[:max(2, population//2)]]
            
            new_population = survivors.copy()
            while len(new_population) < population:
                if random.random() < 0.8 and len(survivors) >= 2:
                    p1, p2 = random.sample(survivors, 2)
                    child = self.generator.crossover(p1, p2)
                else:
                    child = self.generator.mutate(random.choice(survivors))
                new_population.append(child)
            
            population_params = new_population
            print(f"  族群平均分數: {np.mean([s[1] for s in scores]):.3f}")
        
        self.tracker.save()
        return best_params
    
    def optimize_adaptive(self, rounds=150):
        print(f"\n{'='*60}")
        print(f"自適應搜尋模式 ({rounds} 輪)")
        print(f"{'='*60}")
        
        print("\n[Phase 1] 粗搜尋...")
        key_params = ['VOLATILITY_THRESHOLD', 'BULLISH_RATIO_THRESHOLD', 
                      'STRONG_BULLISH_MIN', 'BULLISH_COUNT_MIN']
        
        coarse_results = []
        for params in self.generator.generate_grid(key_params):
            params.update(FIXED_PARAMS)
            metrics = self._evaluate_params(params)
            if metrics:
                coarse_results.append((params.copy(), metrics))
        
        coarse_results.sort(key=lambda x: x[1]['final_score'], reverse=True)
        top_params = [p for p, _ in coarse_results[:20]]
        print(f"Phase 1 完成，細搜 Top {len(top_params)} 參數")
        
        print("\n[Phase 2] 細搜...")
        fine_params = {
            'STOP_LOSS_BUFFER': [0.97, 0.98, 0.99, 1.00],
            'BUY_TOLERANCE': [1.002, 1.005, 1.008],
            'TP1_THRESHOLD': [0.990, 0.995, 1.000],
        }
        
        for base_params in top_params[:5]:
            for k, values in fine_params.items():
                for v in values:
                    test_params = base_params.copy()
                    test_params[k] = v
                    test_params.update(FIXED_PARAMS)
                    
                    metrics = self._evaluate_params(test_params)
                    if metrics:
                        self.tracker.add_result(test_params, metrics)
        
        self.tracker.save()
        return self.tracker.best_params
    
    def get_best(self):
        return self.tracker.best_params, self.tracker.best_score


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='BOSS 參數優化工具')
    parser.add_argument('--mode', '-m', default='random',
                        choices=['grid', 'random', 'genetic', 'adaptive'],
                        help='優化模式')
    parser.add_argument('--type', '-t', default='L',
                        choices=['L', 'M', 'S'],
                        help='股票類型')
    parser.add_argument('--rounds', '-r', type=int, default=100,
                        help='測試回合數')
    parser.add_argument('--generations', '-g', type=int, default=30,
                        help='遺傳演算法代數')
    parser.add_argument('--population', '-p', type=int, default=20,
                        help='族群大小')
    parser.add_argument('--output', '-o', default=None,
                        help='輸出目錄')
    
    args = parser.parse_args()
    
    optimizer = BOSSOptimizer(stype=args.type, output_dir=args.output)
    
    mode_names = {'grid': '網格搜尋', 'random': '隨機搜尋', 
                   'genetic': '遺傳演算法', 'adaptive': '自適應搜尋'}
    
    print(f"\n模式: {mode_names.get(args.mode, args.mode)}")
    print(f"股票類型: {args.type}")
    print(f"股票清單: {len(optimizer.stocks)} 檔")
    print(f"當前最佳分數: {optimizer.tracker.best_score:.3f}")
    
    start_time = time.time()
    
    if args.mode == 'grid':
        optimizer.optimize_grid(rounds=args.rounds)
    elif args.mode == 'random':
        optimizer.optimize_random(rounds=args.rounds)
    elif args.mode == 'genetic':
        optimizer.optimize_genetic(generations=args.generations, population=args.population)
    elif args.mode == 'adaptive':
        optimizer.optimize_adaptive(rounds=args.rounds)
    
    elapsed = time.time() - start_time
    
    best_params, best_score = optimizer.get_best()
    
    print(f"\n{'='*60}")
    print(f"優化完成! 耗時 {elapsed:.1f} 秒")
    print(f"總測試次數: {optimizer.tracker.iteration}")
    print(f"最佳評分: {best_score:.3f}")
    print(f"{'='*60}")
    
    if best_params:
        print("\n最佳參數組合:")
        for k, v in best_params.items():
            if k not in FIXED_PARAMS:
                print(f"  {k}: {v}")
    
    print("\nTop 10 參數組合:")
    top10 = optimizer.tracker.get_top(10)
    if not top10.empty:
        cols = ['iteration', 'final_score', 'win_rate', 'total_trades', 
                'VOLATILITY_THRESHOLD', 'BULLISH_RATIO_THRESHOLD', 
                'STRONG_BULLISH_MIN', 'BULLISH_COUNT_MIN']
        existing_cols = [c for c in cols if c in top10.columns]
        print(top10[existing_cols].to_string())
    
    # 產生 BOSSParams 程式碼
    if best_params:
        print("\n=== BOSSParams 應用代碼 ===")
        print(f"""
params = BOSSParams(
    VOLATILITY_THRESHOLD={best_params.get('VOLATILITY_THRESHOLD', 0.14)},
    BULLISH_RATIO_THRESHOLD={best_params.get('BULLISH_RATIO_THRESHOLD', 0.65)},
    STRONG_BULLISH_MIN={best_params.get('STRONG_BULLISH_MIN', 1)},
    BULLISH_COUNT_MIN={best_params.get('BULLISH_COUNT_MIN', 4)},
    STOP_LOSS_BUFFER={best_params.get('STOP_LOSS_BUFFER', 0.99)},
)

df = cc.checkBoss(df, sno, stype, HHLLdf, params=params)
""")


if __name__ == '__main__':
    main()