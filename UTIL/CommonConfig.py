import pandas as pd
import numpy as np
import time as t
import concurrent.futures as cf
import yfinance as yf
import os
import sys
import platform
import zipfile
import csv
import requests
from pathlib import Path

from collections import defaultdict
from datetime import datetime, timedelta
from tqdm import tqdm

from TA.LW_CalHHLL import *
from TA.LW_Calindicator import *
from TA.LW_CheckWave import *
from TA.LW_CheckBoss import *
from TA.LW_CheckT1 import *
from TA.LW_CheckVCP import *

from AI.MLP import *    #總交易次數: 257 平均勝率: 69.38%
from AI.SVM import *    #總交易次數: 102 平均勝率: 67.74%
from AI.RandomForest import *   #總交易次數: 21 平均勝率: 66.36%
from AI.LogisticRegression import * #總交易次數: 2422 平均勝率: 64.28%
from AI.DecisionTree import *   #總交易次數: 3109 平均勝率: 57.42%

PROD = True

IOPATH = "../Sdata/HKEX/IO/"
SOPATH = "../Sdata/HKEX/SO/"

DATADATE = "2023-01-01"
PATH = "../Sdata/YFdata/"
OUTPATH = "../Sdata/P_YFdata/"

#PATH = "../Sdata/FYFdata/"
#OUTPATH = "../Sdata/FP_YFdata/"

TALIST = ["BOSSB","HHHL","VCP"]
MODELLIST = ["SVM","MLP","RF"]

IS_WINDOWS = platform.system() == "Windows"
IS_IOS = platform.system() == "Darwin"

# 預設的 max_workers（可覆寫）
DEFAULT_MAX_WORKERS = 5 if IS_WINDOWS else 1

ExecutorType = cf.ProcessPoolExecutor if IS_WINDOWS else cf.ThreadPoolExecutor

    
   
