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
import holidays
import calendar
from pathlib import Path

from collections import defaultdict
from datetime import date, datetime, timedelta
from tqdm import tqdm

from TA.LW_Calindicator import *
from TA.LW_CalHHLL import *

from TA.LW_CheckBoss import *
from TA.LW_CheckWave import *
from TA.LW_CheckHFH import *
from TA.LW_CheckT1 import *
from TA.LW_CheckVCP import *
from TA.LW_CheckIchimoku import *
from TA.LW_CheckGBS22C import *
from TA.LW_CheckBreakout200 import *
from TA.LW_CheckFisher import *

from AI.MLP import *    #總交易次數: 257 平均勝率: 69.38%
from AI.SVM import *    #總交易次數: 102 平均勝率: 67.74%
from AI.RandomForest import *   #總交易次數: 21 平均勝率: 66.36%
from AI.LogisticRegression import * #總交易次數: 2422 平均勝率: 64.28%
from AI.DecisionTree import *   #總交易次數: 3109 平均勝率: 57.42%
from AI.ZPrediction import * 

PROD = True

IFPATH = "../SData/HKEX/IF/"
IOPATH = "../SData/HKEX/IO/"
SOPATH = "../SData/HKEX/SO/"

DATADATE = "2024-01-01"
PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/"

#PATH = "../SData/FYFdata/"
#OUTPATH = "../SData/FP_YFdata/"

#TALIST = ["BOSSB","HHHL","VCP","HFH","ICHIMOKU","GBS22C","BREAKOUT200","FISHER"]
TALIST = ["BOSSB","HHHL","VCP","HFH","ICHIMOKU"]
#TALIST = ["ICHIMOKU","GBS22C","BREAKOUT200","FISHER"]
#TALIST = ["BOSSB"]
MODELLIST = ["SVM","MLP","RF"]

IS_WINDOWS = platform.system() == "Windows"
IS_IOS = platform.system() == "Darwin"

# 預設的 max_workers（可覆寫）
DEFAULT_MAX_WORKERS = 5 if IS_IOS else 1

ExecutorType = cf.ThreadPoolExecutor if IS_IOS else cf.ProcessPoolExecutor

def getLastTradeDay(oyear, omonth):
    
    # 計算每月最後一日的前一個工作日（不包括香港公眾假期）
    year = 2000 + oyear
    month = list(calendar.month_abbr).index(omonth.capitalize())

    last_day_num = calendar.monthrange(year, month)[1]
    last_day = date(year, month, last_day_num)

    hk_holidays = holidays.HK(years=2000+oyear)
    prev_workday = last_day

    while prev_workday.weekday() >= 5 or prev_workday in hk_holidays:
        prev_workday -= timedelta(days=1)
    
    return getLastWorkday(prev_workday)

def getLastWorkday(sdate):
    """
    計算今日的上一個工作日（不包括香港公眾假期）
    :return: 上一個工作日的日期字串，格式 yyyymmdd
    """
    hk_holidays = holidays.HK(years=date.today().year)
    prev_workday = sdate - timedelta(days=1)
    while prev_workday.weekday() >= 5 or prev_workday in hk_holidays:
        prev_workday -= timedelta(days=1)
    return prev_workday


   
