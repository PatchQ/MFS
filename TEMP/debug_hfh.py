import pandas as pd
import sys

df = pd.read_csv('../Sdata/P_YFdata/L/P_0001.HK.csv', index_col=0)
print(f'Read {len(df)} rows')
print(f'Index dtype before: {df.index.dtype}')
df.index = pd.to_datetime(df.index)
print(f'Index dtype after: {df.index.dtype}')
df.index = pd.to_datetime(df.index)
print(f'Index dtype after conversion: {df.index.dtype}')
print(f'HFH dtype: {df["HFH"].dtype}')
print(f'HFH True count: {(df["HFH"]==True).sum()}')

# Check HFH signal distribution
hfhs = df[df['HFH']==True]
print(f'HFH dates: {hfhs.index.tolist()}')

# Check specific dates
for date in hfhs.index:
    idx = df.index.get_loc(date)
    print(f'\nDate {date}:')
    print(f'  Index position: {idx}')
    print(f'  HFH at position: {df.iloc[idx]["HFH"]}')
    print(f'  Close at position: {df.iloc[idx]["Close"]}')
    # Check surrounding values
    for i in range(max(0, idx-2), min(len(df), idx+3)):
        print(f'  [{i}] HFH={df.iloc[i]["HFH"]}, Close={df.iloc[i]["Close"]}')

# Now test backtest
from backtesting import Backtest, Strategy

class TestStrat(Strategy):
    def init(self):
        # Store reference to original data
        pass
        
    def next(self):
        bar_num = len(self.data)
        idx = self.data.index[-1]
        hfh_val = self.data.HFH[-1]
        close_val = self.data.Close[-1]
        
        # Only print around the first HFH signal date (2004-01-20)
        if bar_num >= 982 and bar_num <= 990:
            print(f'Bar {bar_num}: index={idx.date()}, HFH={hfh_val}, Close={close_val:.2f}')
        
        if hfh_val:
            print(f'*** BUY SIGNAL at Bar {bar_num}, index={idx.date()}, price={close_val}')
            self.buy()

bt = Backtest(df, TestStrat, cash=200000, commission=0.002, margin=1.0, exclusive_orders=True)
result = bt.run()
print(f'\nBacktest result:')
print(f'# Trades: {result["# Trades"]}')