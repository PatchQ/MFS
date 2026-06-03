import sys
sys.path.append('.')
import UTIL.CommonConfig as cc
import os
import pandas as pd

print('=== Analysis of Combined4Strategy Results ===\n', flush=True)

# Read Combined4Strategy results
combined_file = cc.OUTPATH + '/BT/BT_L_Combined4Strategy.csv'
combined_df = pd.read_csv(combined_file)

total_trades = combined_df['trades_counts'].sum()
avg_win_rate = combined_df['win_rates'].mean()
avg_return = combined_df['returns'].mean()
stocks_with_trades = len(combined_df[combined_df['trades_counts'] > 0])

print(f'L Type Results (3/4 consensus):')
print(f'  Total trades: {total_trades}')
print(f'  Win rate: {avg_win_rate:.2f}%')
print(f'  Return: {avg_return:.2f}%')
print(f'  Stocks with trades: {stocks_with_trades}')

# Check what threshold would give us 3000 trades
# Current: bull_count >= 3
# Let's see what value of bull_count would give ~3000 trades

# Read all individual strategy results
strategies = ['FISHER', 'ICHIMOKU', 'BREAKOUT200', 'GBS22C']

print('\n=== Individual Strategy Results ===\n', flush=True)
for strat in strategies:
    strat_file = cc.OUTPATH + f'/BT/BT_L_{strat}.csv'
    if os.path.exists(strat_file):
        try:
            strat_df = pd.read_csv(strat_file)
            strat_trades = strat_df['trades_counts'].sum()
            strat_wr = strat_df['win_rates'].mean()
            strat_ret = strat_df['returns'].mean()
            print(f'{strat}: {strat_trades} trades, WR: {strat_wr:.2f}%, Return: {strat_ret:.2f}%')
        except Exception as e:
            print(f'{strat}: Error - {e}')

# Read M type combined
combined_m_file = cc.OUTPATH + '/BT/BT_M_Combined4Strategy.csv'
if os.path.exists(combined_m_file):
    combined_m_df = pd.read_csv(combined_m_file)
    m_trades = combined_m_df['trades_counts'].sum()
    m_wr = combined_m_df['win_rates'].mean()
    m_ret = combined_m_df['returns'].mean()
    print(f'\nM Type Combined: {m_trades} trades, WR: {m_wr:.2f}%, Return: {m_ret:.2f}%')

print('\n=== Summary ===\n', flush=True)
print(f'Current 3/4 consensus L+M total trades: {total_trades + m_trades}')
print(f'Target: 3000 trades, 55% win rate, 10% return')
print(f'Problem: 3/4 consensus produces negative return')
print(f'Solution needed: Find consensus level or parameter change that produces positive return with ~3000 trades')