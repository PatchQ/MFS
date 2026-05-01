import sys
sys.path.append('.')
import UTIL.CommonConfig as cc
import os
import pandas as pd

print('Starting debug test', flush=True)

# Read one stock and run backtest
file_path = f'{cc.OUTPATH}/L/P_0001.HK.csv'
print(f'Reading: {file_path}', flush=True)

df = pd.read_csv(file_path)
df.set_index('index', inplace=True)
df.index = pd.to_datetime(df.index)
print(f'Data shape: {df.shape}', flush=True)

# Check if FISHER and ICHIMOKU columns exist
has_fisher = 'FISHER' in df.columns
has_ichimoku = 'ICHIMOKU' in df.columns
print(f'Has FISHER: {has_fisher}, Has ICHIMOKU: {has_ichimoku}', flush=True)

if has_fisher:
    print(f'FISHER True count: {df["FISHER"].sum()}', flush=True)
if has_ichimoku:
    print(f'ICHIMOKU True count: {df["ICHIMOKU"].sum()}', flush=True)

# Check the result of Combined4Strategy
combined_file = cc.OUTPATH + '/BT/BT_L_Combined4Strategy.csv'
print(f'\nReading Combined4Strategy results...', flush=True)
combined_df = pd.read_csv(combined_file)
print(f'Combined4Strategy shape: {combined_df.shape}', flush=True)
print(f'Total trades: {combined_df["trades_counts"].sum()}', flush=True)
print(f'Average win rate: {combined_df["win_rates"].mean():.2f}%', flush=True)
print(f'Average return: {combined_df["returns"].mean():.2f}%', flush=True)

# Check if individual Fisher and Ichimoku results exist
fisher_file = cc.OUTPATH + '/BT/BT_L_FISHER.csv'
ichimoku_file = cc.OUTPATH + '/BT/BT_L_ICHIMOKU.csv'

if os.path.exists(fisher_file):
    fisher_df = pd.read_csv(fisher_file)
    print(f'\nFisher: {fisher_df["trades_counts"].sum()} trades, win rate: {fisher_df["win_rates"].mean():.2f}%, return: {fisher_df["returns"].mean():.2f}%', flush=True)

if os.path.exists(ichimoku_file):
    ichimoku_df = pd.read_csv(ichimoku_file)
    print(f'Ichimoku: {ichimoku_df["trades_counts"].sum()} trades, win rate: {ichimoku_df["win_rates"].mean():.2f}%, return: {ichimoku_df["returns"].mean():.2f}%', flush=True)

print('\nDone', flush=True)