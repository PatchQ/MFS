import os
import sys

# Debug paths
log = []

log.append(f'CWD: {os.getcwd()}')

import UTIL.CommonConfig as cc
log.append(f'cc.OUTPATH: {cc.OUTPATH}')
log.append(f'cc.PATH: {cc.PATH}')

# Test paths
test_paths = [
    cc.OUTPATH,
    '../Sdata/P_YFdata',
    '../../Sdata/P_YFdata',
]

for p in test_paths:
    full = os.path.abspath(p)
    exists = os.path.exists(full)
    log.append(f'{p}: exists={exists}')
    if exists:
        try:
            items = os.listdir(full)
            log.append(f'  items: {items[:5]}')
            if 'L' in items:
                lpath = os.path.join(full, 'L')
                lfiles = os.listdir(lpath)
                log.append(f'  L files: {len(lfiles)}, first: {lfiles[0] if lfiles else "empty"}')
        except Exception as e:
            log.append(f'  error: {e}')

with open('debug_paths.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(log))