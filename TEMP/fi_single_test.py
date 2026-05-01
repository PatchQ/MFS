import sys
sys.path.append('.')
import UTIL.CommonConfig as cc
from backtesting import Backtest, Strategy
from TA.LW_CheckFisher import checkFisher, FisherParams
from TA.LW_CheckIchimoku import checkIchimoku, IchimokuParams
import pandas as pd
import numpy as np

print('Starting test', flush=True)

# Test on one stock
file_path = f'{cc.OUTPATH}/L/P_0001.HK.csv'
print(f'Reading: {file_path}', flush=True)
df = pd.read_csv(file_path)
df.set_index('index', inplace=True)
df.index = pd.to_datetime(df.index)

print('Running TA...', flush=True)
# Run TA
fisher_params = FisherParams()
ichimoku_params = IchimokuParams()
df = checkFisher(df, '0001.HK', 'L', fisher_params)
df = checkIchimoku(df, '0001.HK', 'L', ichimoku_params)

# Check signals
both_true = ((df['FISHER'] == True) & (df['ICHIMOKU'] == True)).sum()
print(f'Stock 0001.HK: Both True count = {both_true}', flush=True)

# Run backtest on this stock
class FIStrategy(Strategy):
    max_holdbars = 100
    sl = -10.0
    tp = 20.0
    dd = 5.0
    
    def init(self):
        self.fisher_signal = 'FISHER' in self.data.df.columns
        self.ichimoku_signal = 'ICHIMOKU' in self.data.df.columns
        
    def next(self):
        if not (self.fisher_signal and self.ichimoku_signal):
            return
        fisher_bull = self.data['FISHER'][-1] if self.fisher_signal else False
        ichimoku_bull = self.data['ICHIMOKU'][-1] if self.ichimoku_signal else False
        if fisher_bull and ichimoku_bull:
            self.buy()

print('Running backtest...', flush=True)
bt = Backtest(df, FIStrategy, cash=200000, commission=0.002, margin=1.0, trade_on_close=False, hedging=False, exclusive_orders=True)
result = bt.run()
print(f'Result: # Trades = {result["# Trades"]}, Return = {result["Return [%]"]:.2f}%', flush=True)

# Save result
with open('TEMP/fi_test_result.txt', 'w') as f:
    f.write(f'Trades: {result["# Trades"]}\n')
    f.write(f'Return: {result["Return [%]"]:.2f}%\n')
    f.write(f'Win Rate: {result["Win Rate [%]"]:.2f}%\n')