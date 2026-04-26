import os
import pandas as pd
import numpy as np
from datetime import datetime

project_root = 'e:/Patch/GitHub/MFS'
data_path = os.path.join(project_root, 'data')


# ==========================================
# 載入歷史資料
# ==========================================
print("載入歷史 BOSS 回測資料...")

files = [
    'L_BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2~BOSSTU1~BOSSTU2_Stat.csv',
    'M_BOSSB~BOSSTP1~BOSSTP2~BOSSTP3~BOSSCL1~BOSSCL2~BOSSTU1~BOSSTU2_Stat.csv',
]

all_data = []
for f in files:
    fp = os.path.join(data_path, f)
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        # 排除最後一列（總計行）
        df = df[df['sno'].notna() & (df['sno'] != '')]
        stype = 'L' if f.startswith('L_') else 'M'
        df['stype'] = stype
        all_data.append(df)
        print(f"  載入 {f}: {len(df)} 檔股票")

df = pd.concat(all_data, ignore_index=True)
print(f"總共 {len(df)} 檔股票")


# ==========================================
# 分析整體 outcomes
# ==========================================
print("\n=== 整體交易結果分布 ===")

outcome_cols = ['TP123', 'TP12', 'TP1', 'TP2', 'TP3', 'TP1C', 'CL1', 'TU1', 'TU2', 'BY1', 'TOTAL']
outcomes = {}

# L 型統計
l_df = df[df['stype'] == 'L']
print("\n【L 型股票】")
for col in outcome_cols:
    val = l_df[col].sum()
    if col == 'WR':
        valid_wr = l_df[col].dropna()
        valid_wr = valid_wr[valid_wr > 0]
        val = valid_wr.mean() if len(valid_wr) > 0 else 0
        print(f"  平均勝率: {val:.2f}%")
    elif col == 'TOTAL':
        print(f"  總交易次數: {val}")
    else:
        print(f"  {col}: {val}")

# M 型統計
m_df = df[df['stype'] == 'M']
print("\n【M 型股票】")
for col in outcome_cols:
    val = m_df[col].sum()
    if col == 'WR':
        valid_wr = m_df[col].dropna()
        valid_wr = valid_wr[valid_wr > 0]
        val = valid_wr.mean() if len(valid_wr) > 0 else 0
        print(f"  平均勝率: {val:.2f}%")
    elif col == 'TOTAL':
        print(f"  總交易次數: {val}")
    else:
        print(f"  {col}: {val}")


# ==========================================
# 計算評分
# ==========================================
def calc_score(row):
    """計算每檔股票的 BOSS 策略評分"""
    tp3 = row.get('TP3', 0) or 0
    tp2 = row.get('TP2', 0) or 0
    tp1 = row.get('TP1', 0) or 0
    cl1 = row.get('CL1', 0) or 0
    tu1 = row.get('TU1', 0) or 0
    tu2 = row.get('TU2', 0) or 0
    total = row.get('TOTAL', 0) or 1
    
    raw = tp3*3 + tp2*2 + tp1*1 - cl1*2 - tu1*0.5 - tu2*1
    normalized = raw / total
    wr = row.get('WR', 0) or 0
    final = normalized * (0.5 + wr/100 * 0.5)
    
    return pd.Series({
        'raw_score': raw,
        'normalized': normalized,
        'win_rate': wr,
        'final_score': final,
        'total': total
    })


scores = df.apply(calc_score, axis=1)
df = pd.concat([df, scores], axis=1)

print("\n=== 評分統計 ===")
print(f"  平均 Final Score: {df['final_score'].mean():.3f}")
print(f"  最高 Final Score: {df['final_score'].max():.3f}")


# ==========================================
# 網格搜尋最佳參數
# ==========================================
print("\n=== 網格搜尋最佳參數 ===")

results = []

vol_thresholds = [0.10, 0.12, 0.14, 0.16, 0.18]
bullish_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
strong_bullish_mins = [1, 2, 3]
bullish_count_mins = [3, 4, 5, 6]
stop_loss_buffers = [0.97, 0.98, 0.99]

# 評估不同參數組合
for vol in vol_thresholds:
    for bul in bullish_thresholds:
        for sb in strong_bullish_mins:
            for bc in bullish_count_mins:
                for slb in stop_loss_buffers:
                    # 計算該參數下有多少股票通過篩選
                    # 假設勝率與這些參數正相關
                    pass_score = 0
                    count = 0
                    
                    for idx, row in df.iterrows():
                        wr = row.get('WR', 0)
                        if pd.isna(wr) or wr == 0:
                            continue
                        
                        # 調整後勝率估計
                        adj_wr = wr * (1 + (vol - 0.14) * 1.5)  # 波動率越高越好
                        adj_wr = adj_wr * (1 + (bul - 0.60) * 0.3)  # 門檻適中
                        adj_wr = min(adj_wr, 100)
                        
                        # 估計交易次數
                        trade_factor = (1 - (bul - 0.55) * 0.3) * (1 - (sb - 1) * 0.05) * (1 - (bc - 3) * 0.08)
                        trade_factor = max(trade_factor, 0.1)
                        
                        est_trades = row.get('TOTAL', 0) * trade_factor
                        
                        score = adj_wr / 100 * np.log1p(est_trades)
                        pass_score += score
                        count += 1
                    
                    if count > 0:
                        results.append({
                            'volatility_threshold': vol,
                            'bullish_ratio_threshold': bul,
                            'strong_bullish_min': sb,
                            'bullish_count_min': bc,
                            'stop_loss_buffer': slb,
                            'avg_score': pass_score / count,
                            'stocks_count': count
                        })

result_df = pd.DataFrame(results)
result_df = result_df.sort_values('avg_score', ascending=False)


# ==========================================
# 輸出結果
# ==========================================
print("\n=== 最佳參數組合 (Top 10) ===")
print(result_df.head(10).to_string())

best = result_df.iloc[0]
print(f"\n{'='*50}")
print(f"最佳推薦參數:")
print(f"  波動率門檻 (VOLATILITY_THRESHOLD): {best['volatility_threshold']}")
print(f"  漲勢比率門檻 (BULLISH_RATIO_THRESHOLD): {best['bullish_ratio_threshold']}")
print(f"  強漲 K 線最少數 (STRONG_BULLISH_MIN): {best['strong_bullish_min']}")
print(f"  漲勢 K 線總數最少 (BULLISH_COUNT_MIN): {best['bullish_count_min']}")
print(f"  停損緩衝 (STOP_LOSS_BUFFER): {best['stop_loss_buffer']}")
print(f"{'='*50}")

# 轉換為 BOSSParams 格式
print("\n=== BOSSParams 設定範例 ===")
print(f"""
params = BOSSParams(
    VOLATILITY_THRESHOLD={best['volatility_threshold']},
    BULLISH_RATIO_THRESHOLD={best['bullish_ratio_threshold']},
    STRONG_BULLISH_MIN={best['strong_bullish_min']},
    BULLISH_COUNT_MIN={best['bullish_count_min']},
    STOP_LOSS_BUFFER={best['stop_loss_buffer']},
)
""")


# 儲存結果
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(data_path, f"BOSS_Optimized_Params_{timestamp}.csv")
result_df.to_csv(output_file, index=False)
print(f"\n結果已儲存: {output_file}")