import sys
sys.path.append('.')
import UTIL.CommonConfig as cc
import os

# Debug script
print('DEBUG: Starting script', flush=True)

bt_dir = cc.OUTPATH + '/BT'
print(f'DEBUG: BT base dir: {bt_dir}', flush=True)
print(f'DEBUG: BT base exists: {os.path.exists(bt_dir)}', flush=True)

target_dir = bt_dir + '/FisherIchimoku'
print(f'DEBUG: Target dir: {target_dir}', flush=True)
print(f'DEBUG: Target exists: {os.path.exists(target_dir)}', flush=True)

if not os.path.exists(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    print(f'DEBUG: Created dir, now exists: {os.path.exists(target_dir)}', flush=True)

if os.path.exists(bt_dir):
    contents = os.listdir(bt_dir)
    print(f'DEBUG: Contents of BT dir: {contents}', flush=True)
else:
    print('DEBUG: BT dir does not exist', flush=True)

# Check L directory
L_dir = cc.OUTPATH + '/L'
print(f'DEBUG: L dir: {L_dir}', flush=True)
print(f'DEBUG: L exists: {os.path.exists(L_dir)}', flush=True)
if os.path.exists(L_dir):
    files = [f for f in os.listdir(L_dir) if f.startswith('P_')]
    print(f'DEBUG: P_ files count: {len(files)}', flush=True)

print('DEBUG: Script completed', flush=True)