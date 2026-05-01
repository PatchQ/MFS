import sys
sys.path.append('e:/Patch/GitHub/MFS')
import UTIL.CommonConfig as cc
import os

print('OUTPATH:', cc.OUTPATH)
L_path = cc.OUTPATH + '/L'
print('L exists:', os.path.exists(L_path))

if os.path.exists(L_path):
    files = [f for f in os.listdir(L_path) if f.startswith('P_')]
    print('P_ files count:', len(files))
    print('First 5:', files[:5] if files else 'None')
else:
    print('L directory not found')
    # Check parent
    parent = os.path.dirname(cc.OUTPATH)
    print('Parent:', parent)
    if os.path.exists(parent):
        print('Contents:', os.listdir(parent)[:10])