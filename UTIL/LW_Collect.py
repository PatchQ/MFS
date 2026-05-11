import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  

import requests
from datetime import datetime, timedelta

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def getYFDaily(sno, sdate):
    try:
        # 轉換股票代碼格式：00700.HK
        if '.' not in sno:
            ticker = f"{int(sno):04d}.HK"
        else:
            ticker = sno

        # 用 Yahoo Finance API 直接拉取（避免 yfinance 被 429）
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=60d"
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
        import pandas as pd
        df = pd.DataFrame({
            'Date': pd.to_datetime(timestamps, unit='s'),
            'Open': quote['open'],
            'High': quote['high'],
            'Low': quote['low'],
            'Close': quote['close'],
            'Volume': quote['volume'],
        })
        df['Date'] = df['Date'].dt.date

        # 過濾指定日期之後的數據
        sdate_dt = datetime.strptime(sdate, '%Y-%m-%d').date()
        df = df[df['Date'] >= sdate_dt]
        df = df[df['Volume'] > 0]
        df.insert(0, "sno", sno.replace('.HK','').lstrip('0') or '0')

        if df.empty:
            return None

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"Get {sno} Data Error: {str(e)}")
        return None


def getYFAll(sno, stype, period):
    # 轉換股票代碼格式
    if '.' not in sno:
        ticker = f"{int(sno):04d}.HK"
    else:
        ticker = sno

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=max"
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

    import pandas as pd
    outputlist = pd.DataFrame({
        'Date': pd.to_datetime(timestamps, unit='s'),
        'Open': quote['open'],
        'High': quote['high'],
        'Low': quote['low'],
        'Close': quote['close'],
        'Volume': quote['volume'],
        'Adj Close': quote.get('adjclose', [None]*len(timestamps)),
    })
    outputlist.index = pd.to_datetime(outputlist['Date'].dt.strftime('%Y%m%d'))
    outputlist = outputlist[outputlist['Volume'] > 0]
    outputlist.insert(0, "sno", sno)

    if len(outputlist) > 0:
        outputlist.to_csv(cc.PATH+"/"+stype+"/"+sno+".csv")

def getDataDaily(sno,stype):        
    
    existing_data = cc.pd.read_csv(cc.PATH+"/"+stype+"/"+sno+".csv")
    existing_data['Date'] = cc.pd.to_datetime(existing_data['Date']).dt.date
        
    if len(existing_data) == 0:
        last_date = cc.datetime(1900, 1, 1).date()
    else:        
        last_date = existing_data['Date'].max()
        if isinstance(last_date, str):
            last_date = cc.datetime.strptime(last_date, '%Y-%m-%d').date()

    start_date = last_date - cc.timedelta(days=1)
    today = cc.datetime.now().date()

    if start_date > today:
        print(f"{sno} 已是最新資料，無需更新")
        return
        
    new_data = getYFDaily(sno, start_date.strftime('%Y-%m-%d'))
    
    if new_data is None or new_data.empty:
        print(f"{sno} No New Data")
        return

    updated_data = cc.pd.concat([existing_data, new_data], ignore_index=True)        
    updated_data = updated_data.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')        
    updated_data.to_csv(cc.PATH+"/"+stype+"/"+sno+".csv", index=False)
        

def YFgetAll(stype,period="max"):
    STOCKLIST = cc.pd.read_csv("Data/stocklist_"+stype+".csv",dtype=str)
    #INDEXLIST = pd.Series(["^HSI","^DJI","^IXIC","^GSPC","^N225","^FTSE","^GDAXI","^FCHI","000001.SS","399001.SZ"])
    SLIST = STOCKLIST[["sno"]]
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST.assign(period=period+"")
    SLIST = SLIST[:]

    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        list(cc.tqdm(executor.map(getYFAll,SLIST["sno"],SLIST["stype"],SLIST["period"],chunksize=1),total=len(SLIST)))

def YFgetDaily(stype):

    snolist = list(map(lambda s: s.replace(".csv", ""), cc.os.listdir(cc.PATH+"/"+stype+"/")))
    SLIST = cc.pd.DataFrame(snolist, columns=["sno"])
    #INDEXLIST = pd.Series(["^HSI","^DJI","^IXIC","^GSPC","^N225","^FTSE","^GDAXI","^FCHI","000001.SS","399001.SZ"])    
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST[:]

    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        list(cc.tqdm(executor.map(getDataDaily,SLIST["sno"],SLIST["stype"],chunksize=1),total=len(SLIST)))        


if __name__ == '__main__':
    start = cc.t.perf_counter()

    YFgetAll("L")
    YFgetAll("M")

    #YFgetAll("L","2y")
    #YFgetAll("M","2y")

    finish = cc.t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')





    
   
