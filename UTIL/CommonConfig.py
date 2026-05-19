import yfinance as yf
import pandas as pd
import numpy as np
import time as t
import concurrent.futures as cf
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

IFPATH = "/root/GitHub/SData/HKEX/IF/"
IOPATH = "/root/GitHub/SData/HKEX/IO/"
SOPATH = "/root/GitHub/SData/HKEX/SO/"

DATADATE = "2024-01-01"
#PATH = "/root/GitHub/SData/YFData/"
#OUTPATH = "/root/GitHub/SData/P_YFData/"

PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/"

FPATH = "/root/GitHub/SData/FYFData/"
FOUTPATH = "/root/GitHub/SData/FP_YFData/"

# HSI 趨勢數據（用於市場趨勢過濾）
_HSI_CACHE = None

def getHSIData():
    """獲取HSI歷史數據並計算EMA20"""
    global _HSI_CACHE
    if _HSI_CACHE is not None:
        return _HSI_CACHE
    
    try:
        # 嘗試從Yahoo Finance API獲取
        import requests
        url = 'https://query1.finance.yahoo.com/v8/finance/chart/^HSI'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        params = {'period1': '946684800', 'period2': '1735689600', 'interval': '1d'}
        r = requests.get(url, headers=headers, params=params, timeout=10)
        data = r.json()
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quotes = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'Date': pd.to_datetime(timestamps, unit='s').strftime('%Y-%m-%d'),
            'Close': quotes['close']
        })
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
        
        # 計算EMA20
        df['EMA20'] = df['Close'].ewm(span=20, min_periods=1).mean()
        df['HSI_Uptrend'] = df['Close'] > df['EMA20']
        
        _HSI_CACHE = df
        print(f"HSI數據已載入: {len(df)} 行, {df.index[0].date()} 至 {df.index[-1].date()}")
        return df
    except Exception as e:
        print(f"HSI數據獲取失敗: {e}")
        return None

# 市場趨勢過濾參數
HSI_TREND_FILTER = True  # 是否啟用HSI趨勢過濾
HSI_MIN_TREND_SCORE = 0  # HSI趨勢評分閾值（0=無要求，正值=必須漲勢）

#TALIST = ["BOSSB","HHHL","VCP","HFH","ICHIMOKU","GBS22C","BREAKOUT200","FISHER"]
TALIST = ["BOSSB","HHHL","VCP","HFH","ICHIMOKU"]
#TALIST = ["ICHIMOKU","GBS22C","BREAKOUT200","FISHER"]
#TALIST = ["BOSSB"]
MODELLIST = ["SVM","MLP","RF"]

IS_WINDOWS = platform.system() == "Windows"
IS_IOS = platform.system() == "Darwin"

# 預設的 max_workers（可覆寫）
DEFAULT_MAX_WORKERS = 5 if IS_IOS else (os.cpu_count() or 4)

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


   
