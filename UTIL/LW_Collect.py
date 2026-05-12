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


def getYFDaily(sno, sdate):
    """
    用 requests 直接調用 Yahoo Finance API，按指定日期範圍下載。
    sdate: str 'YYYY-MM-DD'
    """
    try:
        # 轉換股票代碼格式：00700 → 0700.HK
        if '.' not in sno:
            ticker = f"{int(sno):04d}.HK"
        else:
            ticker = sno

        # 計算 HKT 00:00 → UTC timestamp
        # HKT = UTC+8，所以 HKT 00:00 = UTC 前一日 16:00
        dt_hkt = datetime.strptime(sdate, "%Y-%m-%d")
        utc_dt = dt_hkt - timedelta(hours=8)
        period1 = int(utc_dt.replace(tzinfo=timezone.utc).timestamp())
        # period2 = 下一日 00:00 HKT，確保涵蓋完整目標日期
        period2 = period1 + 86400

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&period1={period1}&period2={period2}"

        resp = requests.get(url, headers=HEADERS, timeout=10)

        if resp.status_code != 200:
            print(f"Get {sno} HTTP Error: {resp.status_code}")
            return None

        data = resp.json()
        result = data.get("chart", {}).get("result")
        if not result:
            return None

        timestamps = result[0].get("timestamp", [])
        quote = result[0]["indicators"]["quote"][0]

        if not timestamps:
            return None

        # 構建 DataFrame
        df = cc.pd.DataFrame({
            'Date': cc.pd.to_datetime(timestamps, unit='s'),
            'Open': quote['open'],
            'High': quote['high'],
            'Low': quote['low'],
            'Close': quote['close'],
            'Volume': quote['volume'],
        })
        df['Date'] = cc.pd.to_datetime(df['Date']).dt.date

        # 過濾指定日期之後的數據（避免 UTC+HKT 時區誤差）
        sdate_dt = datetime.strptime(sdate, '%Y-%m-%d').date()
        df = df[df['Date'] >= sdate_dt]
        df = df[df['Volume'] > 0]
        df.insert(0, "sno", sno.replace('.HK', '').lstrip('0') or '0')

        if df.empty:
            return None

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"Get {sno} Data Error: {str(e)}")
        return None


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

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range={period}"
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
            'Volume': quote['volume'],
            'Adj Close': quote.get('adjclose', [None] * len(timestamps)),
        })
        outputlist.index = cc.pd.to_datetime(outputlist['Date'].dt.strftime('%Y%m%d'))
        outputlist = outputlist[outputlist['Volume'] > 0]
        outputlist.insert(0, "sno", sno)

        if len(outputlist) > 0:
            outputlist.to_csv(cc.PATH + "/" + stype + "/" + sno + ".csv")

    except Exception as e:
        print(f"Get {sno} Error: {str(e)}")


def getDataDaily(sno, stype):
    """
    增量更新：讀取現有 CSV，計算最新日期，只下載新數據。
    """
    try:
        csv_path = cc.PATH + "/" + stype + "/" + sno + ".csv"
        if not os.path.exists(csv_path):
            print(f"{sno} CSV 不存在，請先運行 YFgetAll")
            return

        existing_data = cc.pd.read_csv(csv_path)
        existing_data['Date'] = cc.pd.to_datetime(existing_data['Date']).dt.date

        if len(existing_data) == 0:
            last_date = datetime(1900, 1, 1).date()
        else:
            last_date = existing_data['Date'].max()
            if isinstance(last_date, str):
                last_date = datetime.strptime(last_date, '%Y-%m-%d').date()

        start_date = last_date - timedelta(days=1)
        today = datetime.now().date()

        if start_date > today:
            print(f"{sno} 已是最新資料，無需更新")
            return

        new_data = getYFDaily(sno, start_date.strftime('%Y-%m-%d'))

        if new_data is None or new_data.empty:
            print(f"{sno} No New Data")
            return

        updated_data = cc.pd.concat([existing_data, new_data], ignore_index=True)
        updated_data = updated_data.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
        updated_data.to_csv(csv_path, index=False)
        print(f"{sno} 已更新 {len(new_data)} 行新數據")

    except Exception as e:
        print(f"{sno} getDataDaily Error: {str(e)}")


def YFgetAll(stype, period="max"):
    STOCKLIST = cc.pd.read_csv("Data/stocklist_" + stype + ".csv", dtype=str)
    SLIST = STOCKLIST[["sno"]]
    SLIST = SLIST.assign(stype=stype + "")
    SLIST = SLIST.assign(period=period + "")
    SLIST = SLIST[:]

    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        _YFgetAll = partial(_YFgetAll_worker, stype=stype, period=period)
        list(cc.tqdm(executor.map(_YFgetAll, SLIST["sno"], chunksize=1), total=len(SLIST)))


def YFgetDaily(stype):
    snolist = list(map(lambda s: s.replace(".csv", ""), cc.os.listdir(cc.PATH + "/" + stype + "/")))
    SLIST = cc.pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype + "")
    SLIST = SLIST[:]

    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        list(cc.tqdm(executor.map(getDataDaily, SLIST["sno"], SLIST["stype"], chunksize=1), total=len(SLIST)))


if __name__ == '__main__':
    start = cc.t.perf_counter()

    YFgetAll("L")
    YFgetAll("M")

    # YFgetAll("L","2y")
    # YFgetAll("M","2y")

    finish = cc.t.perf_counter()
    print(f'It took {round(finish - start, 2)} second(s) to finish.')
