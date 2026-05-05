"""
HKEX 股票現貨 OHLCV 下載器
數據源：Yahoo Finance (yfinance)，通過 threading.Lock 序列化調用
每隻股票獨立一個 CSV：{SPPATH}/{code}/{code}_{date}.csv
"""
import sys
import os
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import UTIL.CommonConfig as cc
import threading
import time as t

HK_TZ = ZoneInfo('Asia/Hong_Kong')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPPATH = PROJECT_ROOT / "SData" / "HKEX" / "SP"

# yfinance 全局鎖 — 確保同時只有一個綫程使用 yfinance
_yf_lock = threading.Lock()

# ============================================================
# 股票列表（代碼, 名稱）
# ============================================================
STOCK_LIST = [
    ("00001", "CKH"),
    ("00700", "TCH"),
    ("00939", "CCB"),
    ("00941", "CHM"),
    ("00981", "GSMC"),
    ("01038", "VSC"),
    ("01109", "CRL"),
    ("01359", "CISC"),
    ("01658", "PAB"),
    ("01810", "XMI"),
    ("02318", "PING"),
    ("02382", "SML"),
    ("02388", "BOC"),
    ("02688", "ENM"),
    ("02888", "STC"),
    ("02913", "GZIS"),
    ("02999", "DPH"),
    ("03328", "COM"),
    ("03690", "MEIT"),
    ("03888", "WJG"),
    ("03968", "CMBC"),
    ("03988", "BOCHK"),
    ("06690", "AEON"),
    ("06808", "CMA"),
    ("06888", "TIH"),
    ("09660", "ZBJS"),
    ("09888", "BIDU"),
    ("09939", "SMART"),
    ("09987", "ALIBABA"),
    ("09999", "NTES"),
]


def getLastWorkday(sdate):
    """計算上一個工作日（排除香港公眾假期）"""
    hk_holidays = cc.HK_holidays
    prev = sdate - timedelta(days=1)
    while prev.weekday() >= 5 or prev in hk_holidays:
        prev -= timedelta(days=1)
    return prev


def download_one(code: str, odate: str) -> tuple[str, bool]:
    """下載單隻股票單日數據，回傳 (code, success)"""
    code5 = code.zfill(5)
    sno = f"{code5}.HK"

    with _yf_lock:
        try:
            stock = cc.yf.Ticker(sno)
            dt_start = datetime.strptime(odate, "%Y%m%d")
            # yfinance 需要 'YYYY-MM-DD' 格式
            start_str = dt_start.strftime("%Y-%m-%d")
            data = stock.history(start=start_str, auto_adjust=False)

            if data.empty:
                return code5, False

            data = data[data['Volume'] > 0]
            if data.empty:
                return code5, False

            data = data.reset_index()
            data['Date'] = cc.pd.to_datetime(data['Date']).dt.date
            data.insert(0, 'sno', code5)

            out_dir = SPPATH / code5
            out_dir.mkdir(parents=True, exist_ok=True)
            data.to_csv(out_dir / f"{code5}_{odate}.csv", index=False)
            return code5, True

        except Exception as e:
            print(f"  Error {code5}: {e}")
            return code5, False


def download_file(odate: str) -> bool:
    """下載單日所有股票"""
    success, failed = 0, []

    for code, name in STOCK_LIST:
        code5, ok = download_one(code, odate)
        if ok:
            success += 1
        else:
            failed.append(code5)

    if failed:
        print(f"  ⚠️  {odate}: {success} 成功，{len(failed)} 失敗（{failed[:3]}{'...' if len(failed)>3 else ''}）")
    else:
        print(f"  ✅ {odate}: {success} 隻股票")
    return True


def ProcessDownlaod(sdate, edate):
    start_date = datetime.strptime(sdate, "%Y%m%d")
    end_date = datetime.strptime(edate, "%Y%m%d")
    alldates = pd.date_range(start_date, end_date, freq='D').strftime("%Y%m%d").tolist()

    for odate in alldates:
        download_file(odate)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HKEX 股票現貨 OHLCV 下載（Yahoo Finance）")
    parser.add_argument("--sdate", type=str, help="起始日期 (YYYYMMDD)，預設為前一工作日")
    parser.add_argument("--edate", type=str, help="結束日期 (YYYYMMDD)，預設為前一工作日")
    args = parser.parse_args()

    start = t.perf_counter()

    if args.sdate and args.edate:
        sdate = args.sdate
        edate = args.edate
    else:
        now = datetime.now(HK_TZ)
        last_wd = getLastWorkday(date.today())
        if now.hour < 20:
            sdate = last_wd.strftime("%Y%m%d")
            edate = last_wd.strftime("%Y%m%d")
        else:
            sdate = now.strftime("%Y%m%d")
            edate = now.strftime("%Y%m%d")

    print("=" * 60)
    print(f"  Stock Price Download（Yahoo Finance）")
    print(f"  日期區間：{sdate} → {edate}")
    print(f"  輸出目錄：{SPPATH}")
    print("=" * 60)

    ProcessDownlaod(sdate, edate)

    finish = t.perf_counter()
    print(f"\n完成，耗時 {round(finish - start, 2)} 秒")
