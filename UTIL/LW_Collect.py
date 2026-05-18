import sys
import os
from functools import partial

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc

import requests
import time
from datetime import datetime, timedelta, timezone

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# DataSource Registry for pluggable data sources
from DataSource import get_ohlcv, list_sources


def _DS_worker(sno, stype, period):
    """
    Worker function for DS_YFgetAll (module-level for multiprocessing pickling)
    """
    try:
        # 構造 ticker：直接用 sno（stocklist 已帶 .HK 後綴）
        ticker = sno

        # 用 DataSource 統一接口（自動選擇 Yahoo Finance）
        df = get_ohlcv(ticker, start=period)

        if df is None or len(df) == 0:
            print(f"Get {sno} No data from DataSource")
            return

        # 標準化輸出格式（與 YFgetAll 一致）
        outputlist = cc.pd.DataFrame({
            'Date': cc.pd.to_datetime(df['Date']),
            'Open': df['Open'],
            'High': df['High'],
            'Low': df['Low'],
            'Close': df['Close'],
            'Volume': df['Volume']
        })
        outputlist.index = cc.pd.to_datetime(outputlist['Date'].dt.strftime('%Y%m%d'))
        outputlist.drop(columns=['Date'], inplace=True)
        outputlist = outputlist[outputlist['Volume'] > 0]
        outputlist.insert(0, "sno", sno)

        if len(outputlist) > 0:
            outputlist.to_csv(cc.PATH + "/" + stype + "/" + sno + ".csv")

    except Exception as e:
        print(f"Get {sno} Error: {str(e)}")


def DS_YFgetAll(stype, period="2000-01-01"):
    """
    用 DataSource Registry 下載歷史數據（統一接口，可插拔數據源）
    功能同 YFgetAll，但底層調用 DataSource.get_ohlcv
    """
    STOCKLIST = cc.pd.read_csv("Data/stocklist_" + stype + ".csv", dtype=str)
    SLIST = STOCKLIST[["sno"]]
    SLIST = SLIST.assign(stype=stype + "")
    SLIST = SLIST.assign(period=period + "")
    SLIST = SLIST[:]

    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        _ds_fn = partial(_DS_worker, stype=stype, period=period)
        list(cc.tqdm(executor.map(_ds_fn, SLIST["sno"], chunksize=1), total=len(SLIST)))

def _get_with_retry(url, max_retries=3, base_timeout=30):
    """
    帶重試機制的 GET 請求。
    - timeout 增加到 30 秒
    - 失敗時最多重試 3 次，等 5/10/15 秒
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=base_timeout)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError) as e:
            last_err = e
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 5
                time.sleep(wait)
            else:
                raise
    raise last_err

def _YFgetAll_worker(sno, stype, period):
    """
    用 requests 直接調用 Yahoo Finance API，全量下載。
    period: 'max' or 'Xd' (e.g., '2y')
    """
    try:
        if '.' not in sno:
            ticker = f"{int(sno):04d}.HK"
        else:
            ticker = sno


        # 計算 HKT 00:00 → UTC timestamp
        # HKT = UTC+8，所以 HKT 00:00 = UTC 前一日 16:00
        dt_hkt = datetime.strptime(period, "%Y-%m-%d")
        utc_dt = dt_hkt - timedelta(hours=8)
        period1 = int(utc_dt.replace(tzinfo=timezone.utc).timestamp())

        today = cc.datetime.now()
        utc_today = today - timedelta(hours=8)
        period2 = int(utc_today.replace(tzinfo=timezone.utc).timestamp())

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&period1={period1}&period2={period2}"
        resp = _get_with_retry(url)

        if resp.status_code != 200:
            print(f"Get {sno} HTTP Error: {resp.status_code}")
            return

        data = resp.json()
        result = data.get("chart", {}).get("result")
        if not result:
            return

        timestamps = result[0].get("timestamp", [])
        quote = result[0]["indicators"]["quote"][0]

        if not timestamps:
            return

        outputlist = cc.pd.DataFrame({
            'Date': cc.pd.to_datetime(timestamps, unit='s'),
            'Open': quote['open'],
            'High': quote['high'],
            'Low': quote['low'],
            'Close': quote['close'],
            'Volume': quote['volume']            
        })
        outputlist.index = cc.pd.to_datetime(outputlist['Date'].dt.strftime('%Y%m%d'))
        outputlist.drop(columns=['Date'], inplace=True)        
        outputlist = outputlist[outputlist['Volume'] > 0]
        outputlist.insert(0, "sno", sno)

        if len(outputlist) > 0:
            outputlist.to_csv(cc.PATH + "/" + stype + "/" + sno + ".csv")

    except Exception as e:
        print(f"Get {sno} Error: {str(e)}")


def YFgetAll(stype, period="2000-01-01"):
    STOCKLIST = cc.pd.read_csv("Data/stocklist_" + stype + ".csv", dtype=str)
    SLIST = STOCKLIST[["sno"]]
    SLIST = SLIST.assign(stype=stype + "")
    SLIST = SLIST.assign(period=period + "")
    SLIST = SLIST[:]

    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        _YFgetAll = partial(_YFgetAll_worker, stype=stype, period=period)
        list(cc.tqdm(executor.map(_YFgetAll, SLIST["sno"], chunksize=1), total=len(SLIST)))

if __name__ == '__main__':
    start = cc.t.perf_counter()

    #YFgetAll("L")
    #YFgetAll("M")

    # YFgetAll("L","2y")
    # YFgetAll("M","2y")

    finish = cc.t.perf_counter()
    print(f'It took {round(finish - start, 2)} second(s) to finish.')
