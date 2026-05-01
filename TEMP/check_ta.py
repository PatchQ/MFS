import sys
sys.path.append('.')
import UTIL.CommonConfig as cc
from TA.LW_CheckFisher import checkFisher, FisherParams
from TA.LW_CheckIchimoku import checkIchimoku, IchimokuParams
import pandas as pd

# Read one sample stock
file_path = f'{cc.OUTPATH}/L/P_0001.HK.csv'
df = pd.read_csv(file_path)
df.set_index('index', inplace=True)
df.index = pd.to_datetime(df.index)

# Run TA functions
fisher_params = FisherParams()
ichimoku_params = IchimokuParams()

df = checkFisher(df, '0001.HK', 'L', fisher_params)
df = checkIchimoku(df, '0001.HK', 'L', ichimoku_params)

print('Columns after TA:', list(df.columns[-15:]))
print('FISHER True count:', df['FISHER'].sum() if 'FISHER' in df.columns else 'N/A')
print('ICHIMOKU True count:', df['ICHIMOKU'].sum() if 'ICHIMOKU' in df.columns else 'N/A')
if 'FISHER' in df.columns and 'ICHIMOKU' in df.columns:
    both_true = ((df['FISHER'] == True) & (df['ICHIMOKU'] == True)).sum()
    print('Both True count:', both_true)
    print('\nLast 10 FISHER values:', df['FISHER'].tail(10).tolist())
    print('Last 10 ICHIMOKU values:', df['ICHIMOKU'].tail(10).tolist())