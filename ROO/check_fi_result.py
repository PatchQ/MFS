import sys
sys.path.append('.')
import UTIL.CommonConfig as cc
import pandas as pd
import os

# Check if FisherIchimoku file exists
fi_file = cc.OUTPATH + '/BT/BT_L_FisherIchimoku.csv'
print(f'FisherIchimoku file exists: {os.path.exists(fi_file)}')

if os.path.exists(fi_file):
    df = pd.read_csv(fi_file)
    print(f'Shape: {df.shape}')
    print(df.head())
else:
    print('FisherIchimoku file not found')

# Also check individual stock results in FisherIchimoku folder
fi_dir = cc.OUTPATH + '/BT/FisherIchimoku'
print(f'\nFisherIchimoku dir exists: {os.path.exists(fi_dir)}')
if os.path.exists(fi_dir):
    files = os.listdir(fi_dir)
    print(f'File count: {len(files)}')
    print(f'First 5: {files[:5]}')