import pandas as pd
import numpy as np
import time as t
import concurrent.futures as cf
import yfinance as yf
import os
import sys
import platform

from collections import defaultdict
from datetime import datetime, timedelta
from tqdm import tqdm

from TA.LW_CalHHLL import *
from TA.LW_Calindicator import *
from TA.LW_CheckWave import *
from TA.LW_CheckBoss import *
from TA.LW_CheckT1 import *
from TA.LW_CheckVCP import *

from AI.DecisionTree import *
from AI.XGBoost import *
from AI.LightGBM import *
from AI.LogisticRegression import *
from AI.MLP import *
from AI.RandomForest import *
from AI.SVM import *
from AI.VOTING import *
import AI.ZPrediction as zp

PROD = False

PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/"

#PATH = "../SData/FYFData/"
#OUTPATH = "../SData/FP_YFData/"

MODELLIST = ["DT","XGB","LGBM","LR","MLP","RF","SVM","VOTING"]

IS_WINDOWS = platform.system() == "Windows"
IS_IOS = platform.system() == "Darwin"

# 預設的 max_workers（可覆寫）
DEFAULT_MAX_WORKERS = 5 if IS_WINDOWS else 1

ExecutorType = cf.ProcessPoolExecutor if IS_WINDOWS else cf.ThreadPoolExecutor

    
   
