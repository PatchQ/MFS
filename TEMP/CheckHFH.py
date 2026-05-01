"""Quick check HFH signals after ProcessTA"""
import os
import pandas as pd

total = 0
for f in os.listdir('../SData/P_YFdata/L/')[:50]:
    if f.startswith('P_'):
        df = pd.read_csv(f'../SData/P_YFdata/L/{f}', index_col=0)
        hfh_count = df['HFH'].sum()
        if hfh_count > 0:
            print(f'{f}: {hfh_count} HFH')
        total += hfh_count

print(f'\nTotal HFH signals (first 50 stocks): {total}')