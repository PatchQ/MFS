import sys
sys.path.append('.')
import UTIL.CommonConfig as cc
import pandas as pd

# Read Combined4Strategy results
bt_file = cc.OUTPATH + '/BT/BT_L_Combined4Strategy.csv'
print('Reading:', bt_file)
df = pd.read_csv(bt_file)
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
if 'trades_counts' in df.columns:
    print('Total trades:', df['trades_counts'].sum())
else:
    print('No trades_counts column')
if 'win_rates' in df.columns:
    print('Avg win rate:', df['win_rates'].mean())
if 'returns' in df.columns:
    print('Avg return:', df['returns'].mean())
print(df.head(3).to_string())