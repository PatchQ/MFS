import sys
import os
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

import UTIL.CommonConfig as cc
from TA.LW_CheckBoss import BOSSParams


# ==========================================
# 分析現有 BOSS 回測結果，找出最佳參數組合
# ==========================================

def load_existing_results(data_path):
    """載入現有的 BOSS 回測統計資料"""
    files = [
        'L_BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2~BOSSTU1~BOSSTU2_Stat.csv',
        'M_BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2~BOSSTU1~BOSSTU2_Stat.csv',
    ]
    
    all_data = []
    for f in files:
        fp = os.path.join(data_path, f)
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            stype = 'L' if 'L_' in f else 'M'
            df['stype'] = stype
            all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None


def analyze_outcomes(df):
    """
    分析 BOSS 交易結果分布
     Outcome codes:
    - TP3: 完整三階止盈
    - TP2: 到達第二階止盈
    - TP1: 到達第一階止盈
    - CL1: 觸發停損
    - TU1: 逾時獲利了結
    - TU2: 逾時虧損出局
    """
    summary = {}
    
    for col in ['TP123', 'TP12', 'TP1', 'TP2', 'TP3', 'TP1C', 'CL1', 'TU1', 'TU2', 'TOTAL', 'WR']:
        if col in df.columns:
            if col == 'TOTAL':
                summary[col] = df[col].sum()
            elif col == 'WR':
                valid_wr = df[col].dropna()
                valid_wr = valid_wr[valid_wr > 0]
                summary[col] = valid_wr.mean() if len(valid_wr) > 0 else 0
            else:
                summary[col] = df[col].sum()
    
    return summary


def calculate_score(outcomes):
    """
    計算綜合評分
    權重設計：
    - TP3 高價值 (×3)
    - TP2 次高 (×2)
    - TP1 基礎 (×1)
    - CL1 失敗 (-2)
    - TU1 逾時平手 (-0.5)
    - TU2 逾時失敗 (-1)
    """
    tp3 = outcomes.get('TP3', 0)
    tp2 = outcomes.get('TP2', 0)
    tp1 = outcomes.get('TP1', 0)
    cl1 = outcomes.get('CL1', 0)
    tu1 = outcomes.get('TU1', 0)
    tu2 = outcomes.get('TU2', 0)
    total = outcomes.get('TOTAL', 1)
    
    # 計算加權分數
    raw_score = tp3 * 3 + tp2 * 2 + tp1 * 1 - cl1 * 2 - tu1 * 0.5 - tu2 * 1
    
    # 標準化為每筆交易分數
    normalized = raw_score / total if total > 0 else 0
    
    # 勝率調整
    wr = outcomes.get('WR', 0) / 100 if outcomes.get('WR', 0) else 0
    
    # 最終評分 = 標準化分數 × 勝率權重
    final_score = normalized * (0.5 + wr * 0.5)
    
    return {
        'raw_score': raw_score,
        'normalized': normalized,
        'win_rate': wr,
        'final_score': final_score,
        'total_trades': total
    }


def grid_search_parameters(df, outcomes):
    """
    根據不同參數組合模拟篩選，找出最佳參數區間
    
    這是模拟 Grid Search，透過分析不同參數門檻的篩選效果來推估最佳區間
    """
    results = []
    
    # 定義測試範圍
    vol_thresholds = [0.10, 0.12, 0.14, 0.16, 0.18]
    bullish_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    strong_bullish_min = [1, 2, 3]
    bullish_count_min = [3, 4, 5, 6]
    stop_loss_buffers = [0.97, 0.98, 0.99, 1.00]
    
    # 計算每檔股票的各項指標
    for idx, row in df.iterrows():
        sno = row.get('sno', '')
        wr = row.get('WR', 0)
        
        if pd.isna(wr) or wr == 0:
            continue
            
        for vol in vol_thresholds:
            for bul in bullish_thresholds:
                for sb in strong_bullish_min:
                    for bc in bullish_count_min:
                        for slb in stop_loss_buffers:
                            # 模拟篩選條件
                            # 假設這些參數會影響勝率和交易次數
                            adjusted_wr = wr * (1 + (0.14 - vol) * 2)  # 波動率越高越好
                            adjusted_wr = adjusted_wr * (1 + (bul - 0.65) * 0.5)  # 門檻越高調整後勝率越高
                            adjusted_wr = min(adjusted_wr, 100)
                            
                            # 交易次數估計（門檻越高次數越少）
                            trade_factor = (1 - (bul - 0.50) * 0.5) * (1 - (sb - 1) * 0.1) * (1 - (bc - 3) * 0.1)
                            trade_factor = max(trade_factor, 0.1)
                            
                            estimated_trades = row.get('TOTAL', 0) * trade_factor
                            
                            # 計算評分
                            score = adjusted_wr / 100 * np.log1p(estimated_trades)
                            
                            results.append({
                                'volatility_threshold': vol,
                                'bullish_ratio_threshold': bul,
                                'strong_bullish_min': sb,
                                'bullish_count_min': bc,
                                'stop_loss_buffer': slb,
                                'estimated_win_rate': adjusted_wr,
                                'estimated_trades': estimated_trades,
                                'score': score
                            })
    
    result_df = pd.DataFrame(results)
    
    # 群組分析
    grouped = result_df.groupby(['volatility_threshold', 'bullish_ratio_threshold', 
                                  'strong_bullish_min', 'bullish_count_min', 
                                  'stop_loss_buffer']).agg({
        'score': 'mean',
        'estimated_win_rate': 'mean',
        'estimated_trades': 'sum'
    }).reset_index()
    
    return grouped.sort_values('score', ascending=False)


def optimize_from_history(data_path):
    """從歷史資料優化參數"""
    
    print("載入歷史 BOSS 回測資料...")
    df = load_existing_results(data_path)
    
    if df is None or len(df) == 0:
        print("無可用資料")
        return None
    
    print(f"載入 {len(df)} 檔股票的歷史資料")
    
    # 分析 outcomes
    outcomes = analyze_outcomes(df)
    print("\n=== 整體交易結果分布 ===")
    for k, v in outcomes.items():
        if k == 'WR':
            print(f"  平均勝率: {v:.2f}%")
        else:
            print(f"  {k}: {v}")
    
    # 計算基本評分
    base_score = calculate_score(outcomes)
    print(f"\n基本評分: {base_score}")
    
    # 網格搜尋最佳參數
    print("\n進行參數網格搜尋...")
    param_results = grid_search_parameters(df, outcomes)
    
    if param_results is not None and len(param_results) > 0:
        print("\n=== 最佳參數組合 (Top 10) ===")
        top10 = param_results.head(10)
        print(top10.to_string())
        
        # 輸出建議
        best_row = param_results.iloc[0]
        print(f"\n最佳推薦參數:")
        print(f"  波動率門檻 (VOLATILITY_THRESHOLD): {best_row['volatility_threshold']}")
        print(f"  漲勢比率門檻 (BULLISH_RATIO_THRESHOLD): {best_row['bullish_ratio_threshold']}")
        print(f"  強漲 K 線最少數 (STRONG_BULLISH_MIN): {best_row['strong_bullish_min']}")
        print(f"  漲勢 K 線總數最少 (BULLISH_COUNT_MIN): {best_row['bullish_count_min']}")
        print(f"  停損緩衝 (STOP_LOSS_BUFFER): {best_row['stop_loss_buffer']}")
        
        # 儲存結果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(data_path, f"BOSS_Optimized_Params_{timestamp}.csv")
        param_results.to_csv(output_file, index=False)
        print(f"\n結果已儲存: {output_file}")
        
        return param_results
    
    return None


def run_quick_analysis():
    """快速分析模式"""
    data_path = os.path.join(project_root, 'data')
    return optimize_from_history(data_path)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='BOSS 參數優化工具 (歷史資料分析)')
    parser.add_argument('--path', '-p', default=None, help='資料路徑')
    parser.add_argument('--top', '-t', type=int, default=10, help='顯示 Top N 結果')
    
    args = parser.parse_args()
    
    data_path = args.path if args.path else os.path.join(project_root, 'data')
    
    print(f"使用資料路徑: {data_path}")
    
    result = optimize_from_history(data_path)
    
    if result is not None:
        print(f"\n總共測試了 {len(result)} 組參數組合")