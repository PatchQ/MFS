import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from TA.LW_CheckFisher import FisherParams, checkFisher

print('FisherParams.TRIGGER_THRESHOLD =', FisherParams.TRIGGER_THRESHOLD)
print('FisherParams.STRONG_THRESHOLD =', FisherParams.STRONG_THRESHOLD)

df = pd.read_csv('e:/Patch/GitHub/Sdata/P_YFdata/L/P_0001.HK.csv', index_col=0, low_memory=False)
print('Before checkFisher - FISHER sum:', df['FISHER'].sum())

df_result = checkFisher(df.head(200).copy(), '0001.HK', 'L')
print('After checkFisher - FISHER sum:', df_result['FISHER'].sum())
print('FISHER_SIGNAL:', df_result['FISHER_SIGNAL'].value_counts().to_dict())
print('FISHER_STRENGTH:', df_result['FISHER_STRENGTH'].value_counts().to_dict())

# 測試簡單的計算
print('Test calculation:', 1+1)
