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
import workdays
from pathlib import Path

from collections import defaultdict
from datetime import date, datetime, timedelta
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


# ---------- 1. 定義2026年澳門公眾假期及補假日 ----------
# 資料來源：澳門特別行政區政府入口網站 [citation:1][citation:2][citation:3]
# 公眾假期（含補假）列表，格式為 (月, 日)
public_holidays_2026 = [
    (1, 1),   # 元旦
    (2, 17),  # 農曆正月初一
    (2, 18),  # 農曆正月初二
    (2, 19),  # 農曆正月初三
    (4, 3),   # 耶穌受難日
    (4, 4),   # 復活節前日
    (4, 5),   # 清明節
    (4, 6),   # 復活節前日補假 [citation:6]
    (4, 7),   # 清明節補假 [citation:6]
    (5, 1),   # 勞動節
    (5, 24),  # 佛誕節
    (5, 25),  # 佛誕節補假 [citation:6]
    (6, 19),  # 端午節
    (9, 26),  # 中秋節翌日
    (9, 28),  # 中秋節翌日補假 [citation:6]
    (10, 1),  # 中華人民共和國國慶日
    (10, 2),  # 中華人民共和國國慶日翌日
    (10, 18), # 重陽節
    (10, 19), # 重陽節補假 [citation:6]
    (11, 2),  # 追思節
    (12, 8),  # 聖母無原罪瞻禮
    (12, 20), # 澳門特別行政區成立紀念日
    (12, 21), # 澳門特別行政區成立紀念日補假 [citation:6]
    (12, 22), # 冬至
    (12, 24), # 聖誕節前日
    (12, 25), # 聖誕節
    # 除夕下午豁免上班不計入全天假期，故不列入
]

# 轉換為 date 物件集合
holiday_dates = {date(2026, m, d) for m, d in public_holidays_2026}


def is_workday(check_date: date) -> bool:
    """
    判斷某一天在澳門是否為工作日
    :param check_date: 需要判斷的日期
    :return: True = 工作日, False = 非工作日（週末或公眾假期）
    """
    # 星期六或星期日 -> 非工作日
    if check_date.weekday() >= 5:  # 5=星期六, 6=星期日
        return False
    # 公眾假期 -> 非工作日
    if check_date in holiday_dates:
        return False
    # 其餘為工作日
    return True

def is_yesterday_workday() -> bool:
    """
    判斷昨天是否為澳門的工作日
    :return: True = 昨天是工作日, False = 昨天不是工作日
    """
    yesterday = date.today() - timedelta(days=1)
    return is_workday(yesterday)

def previous_workday(from_date: date = None) -> str:
    """
    找出 from_date 的前一個工作日
    :param from_date: 基準日期，預設為今天
    :return: 前一個工作日的日期字串，格式 yyyymmdd
    """
    if from_date is None:
        from_date = date.today()

    check_date = from_date - timedelta(days=1)
    while not is_workday(check_date):
        check_date -= timedelta(days=1)
    return check_date.strftime("%Y%m%d")    
   
