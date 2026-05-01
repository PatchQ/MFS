import sys
sys.path.append('.')
print('Step 1: Starting', flush=True)

import UTIL.CommonConfig as cc
import os

print('Step 2: Import done', flush=True)
print(f'OUTPATH: {cc.OUTPATH}', flush=True)

# Check BT directory
bt_path = cc.OUTPATH + '/BT'
print(f'BT path: {bt_path}', flush=True)
print(f'BT exists: {os.path.exists(bt_path)}', flush=True)

# List BT contents
if os.path.exists(bt_path):
    contents = os.listdir(bt_path)
    print(f'BT contents ({len(contents)}):', contents[:10], flush=True)

# Check FisherIchimoku file
fi_csv = cc.OUTPATH + '/BT/BT_L_FisherIchimoku.csv'
print(f'FI CSV exists: {os.path.exists(fi_csv)}', flush=True)

# Check FisherIchimoku dir
fi_dir = cc.OUTPATH + '/BT/FisherIchimoku'
print(f'FI dir exists: {os.path.exists(fi_dir)}', flush=True)
if os.path.exists(fi_dir):
    files = os.listdir(fi_dir)
    print(f'FI dir files: {len(files)}', flush=True)

print('Step 3: Done', flush=True)