import sys
import os
from functools import partial

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc

import requests
from datetime import datetime, timedelta, timezone

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

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
        resp = requests.get(url, headers=HEADERS, timeout=10)

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


def YFgetAll(stype, period="20000101"):
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
