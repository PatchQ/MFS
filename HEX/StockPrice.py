"""
HKEX 股票現貨 OHLCV 下載器
數據源：Yahoo Finance (yfinance)
股票列表：從 solist.csv 動態讀取 SNO 代碼

策略：每隻股票一次過拿完整區間（sdate→edate），大幅減少 API 請求次數
輸出：{SPPATH}/{code}/{code}_{date}.csv（每股票每日一個 CSV）
"""
import sys
import os
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import UTIL.CommonConfig as cc
import threading
import time as t

HK_TZ = ZoneInfo('Asia/Hong_Kong')

SPPATH = Path("/root/GitHub/SData/HKEX/SP")
SOLIST_PATH = Path("/root/GitHub/SData/HKEX/SO/solist.csv")

# yfinance 全局鎖 — 確保同時只有一個綫程使用 yfinance
_yf_lock = threading.Lock()
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 3  # 秒


def load_stock_list() -> list:
    """從 solist.csv 載入股票代碼列表"""
    df = cc.pd.read_csv(SOLIST_PATH)
    # SNO 格式如 '2823.HK', '0700.HK' — 直接用作 Yahoo Finance 代碼
    return df['SNO'].dropna().tolist()


def get_last_workday(sdate):
    """計算上一個工作日（排除香港公眾假期）"""
    import holidays
    hk_holidays = holidays.HK(years=sdate.year)
    prev = sdate - timedelta(days=1)
    while prev.weekday() >= 5 or prev in hk_holidays:
        prev -= timedelta(days=1)
    return prev


# ============================================================
# 股票現貨下載
# ============================================================
INDEX_LIST = [
    # (Yahoo Finance 代碼, 目錄名, SNO顯示名)
    ("^HSI",     "HSI",     "HSI"),
    ("HSTECH.HK","HSTECH",  "HSTECH"),
    ("^HSCE",    "HSCE",    "HSCE"),
]

def download_index_range(sno: str, code: str, sdate: str, edate: str) -> tuple[str, int, int, list]:
    """
    下載指數整個日期區間的數據。
    sno: Yahoo Finance 代碼（如 ^HSI）
    code: 保存目錄名（如 HSI）
    """
    dt_start = datetime.strptime(sdate, "%Y%m%d")
    dt_end = datetime.strptime(edate, "%Y%m%d")
    start_str = dt_start.strftime("%Y-%m-%d")
    end_str = (dt_end + timedelta(days=1)).strftime("%Y-%m-%d")

    for attempt in range(1, _MAX_RETRIES + 1):
        with _yf_lock:
            try:
                stock = cc.yf.Ticker(sno)
                data = stock.history(start=start_str, end=end_str, auto_adjust=False)
            except Exception as e:
                print(f"  [SP] Error {sno} (attempt {attempt}): {e}")
                data = None

        if data is None or data.empty:
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"  [SP] {sno} rate limited，{wait}s 後重試 (attempt {attempt}/{_MAX_RETRIES})")
                t.sleep(wait)
                continue
            return sno, 0, 1, []

    if data.empty:
        return sno, 0, 1, []

    # 指數沒有成交量，跳過 Volume > 0 過濾
    data = data.reset_index()
    data['Date'] = cc.pd.to_datetime(data['Date']).dt.date
    data.insert(0, 'sno', code)  # 用乾淨代碼

    date_to_row = {}
    for _, row in data.iterrows():
        d = row['Date']
        date_str = d.strftime("%Y%m%d")
        date_to_row[date_str] = row

    target_dates = cc.pd.date_range(dt_start, dt_end, freq='D').strftime("%Y%m%d").tolist()
    out_dir = SPPATH / code
    out_dir.mkdir(parents=True, exist_ok=True)

    success, missing = 0, []
    for target_d in target_dates:
        if target_d in date_to_row:
            row = date_to_row[target_d]
            row_df = data[data['Date'] == row['Date']].copy()
            row_df.to_csv(out_dir / f"{code}_{target_d}.csv", index=False)
            success += 1
        else:
            missing.append(target_d)

    return sno, success, len(missing), missing


def download_stock_range(sno: str, sdate: str, edate: str) -> tuple[str, int, int, list]:
    """
    下載單隻股票整個日期區間的數據，一次 API call 拿完全部。
    失敗時自動重試，最多重試 _MAX_RETRIES 次。
    回傳 (sno, success_count, fail_count, missing_dates)
    """
    dt_start = datetime.strptime(sdate, "%Y%m%d")
    dt_end = datetime.strptime(edate, "%Y%m%d")
    start_str = dt_start.strftime("%Y-%m-%d")
    end_str = (dt_end + timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance end 是 exclusive

    for attempt in range(1, _MAX_RETRIES + 1):
        with _yf_lock:
            try:
                stock = cc.yf.Ticker(sno)
                # 一次過拿整個區間
                data = stock.history(start=start_str, end=end_str, auto_adjust=False)
            except Exception as e:
                print(f"  [SP] Error {sno} (attempt {attempt}): {e}")
                data = None

        if data is None or data.empty:
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BASE_DELAY * (2 ** (attempt - 1))  # 指數退避
                print(f"  [SP] {sno} rate limited，{wait}s 後重試 (attempt {attempt}/{_MAX_RETRIES})")
                t.sleep(wait)
                continue
            return sno, 0, 1, []

    if data.empty:
        return sno, 0, 1, []

    # 只保留有成交量的日子
    data = data[data['Volume'] > 0]
    if data.empty:
        return sno, 0, 1, []

    data = data.reset_index()
    data['Date'] = cc.pd.to_datetime(data['Date']).dt.date

    # 提取數字代碼如 '0700'（去除 .HK 後綴）
    code = sno.replace('.HK', '')
    data.insert(0, 'sno', code)

    # 建立 {date_str: row} 的查找表
    date_to_row = {}
    for _, row in data.iterrows():
        d = row['Date']
        if hasattr(d, 'strftime'):
            date_str = d.strftime("%Y%m%d")
        else:
            date_str = d.strftime("%Y%m%d")
        date_to_row[date_str] = row

    # 產生目標日期列表
    target_dates = cc.pd.date_range(dt_start, dt_end, freq='D').strftime("%Y%m%d").tolist()
    out_dir = SPPATH / code
    out_dir.mkdir(parents=True, exist_ok=True)

    success, missing = 0, []
    for target_d in target_dates:
        if target_d in date_to_row:
            row = date_to_row[target_d]
            row_df = data[data['Date'] == row['Date']].copy()
            row_df.to_csv(out_dir / f"{code}_{target_d}.csv", index=False)
            success += 1
        else:
            missing.append(target_d)

    return sno, success, len(missing), missing


def download_file(sdate: str, edate: str, stock_list: list) -> tuple[int, int, list]:
    """
    下載所有股票一個區間的數據（每股票一次 API call）
    回傳 (success_count, fail_count, failed_snos)
    """
    success, fail = 0, []
    for sno in stock_list:
        _, ok, fail_count, _ = download_stock_range(sno, sdate, edate)
        if fail_count == 0:
            success += 1
        else:
            fail.append(sno)
    return success, len(fail), fail


def process_download(sdate: str, edate: str, stock_list: list):
    """下載股票 + 指數"""
    # 先下載指數
    print(f"\n  === 指數 ===")
    idx_success, idx_fail = 0, []
    for sno, code, _ in INDEX_LIST:
        _, ok, fail_count, missing = download_index_range(sno, code, sdate, edate)
        if fail_count == 0:
            idx_success += 1
            print(f"  ✅ {code} ({sno})")
        else:
            idx_fail.append(f"{code}({sno})")
            print(f"  ❌ {code} ({sno})")
    if idx_fail:
        print(f"  ⚠️  指數: {idx_success} 成功，{len(idx_fail)} 失敗：{idx_fail}")

    # 再下載股票
    print(f"\n  === 股票 ({len(stock_list)} 隻) ===")
    total_success, total_fail_count, total_fails = download_file(sdate, edate, stock_list)
    if total_fails:
        print(f"  ⚠️  {sdate}~{edate}: {total_success} 成功，{total_fail_count} 失敗")
        print(f"      失敗股票: {total_fails[:5]}{'...' if len(total_fails) > 5 else ''}")
    else:
        print(f"  ✅ {sdate}~{edate}: {total_success} 隻股票")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HKEX 股票現貨 OHLCV 下載（Yahoo Finance）")
    parser.add_argument("--sdate", type=str, help="起始日期 (YYYYMMDD)，預設為前一工作日")
    parser.add_argument("--edate", type=str, help="結束日期 (YYYYMMDD)，預設為前一工作日")
    args = parser.parse_args()

    start = t.perf_counter()

    # 動態載入股票列表
    stock_list = load_stock_list()
    print(f"從 solist.csv 載入 {len(stock_list)} 隻股票")

    if args.sdate and args.edate:
        sdate = args.sdate
        edate = args.edate
    else:
        now = datetime.now(HK_TZ)
        last_wd = cc.getLastWorkday(date.today())
        if now.hour < 20:
            sdate = last_wd.strftime("%Y%m%d")
            edate = last_wd.strftime("%Y%m%d")
        else:
            sdate = last_wd.strftime("%Y%m%d")
            edate = now.strftime("%Y%m%d")

    print("=" * 60)
    print(f"  Stock Price Download（Yahoo Finance）")
    print(f"  日期區間：{sdate} → {edate}")
    print(f"  輸出目錄：{SPPATH}")
    print(f"  股票數量：{len(stock_list)} 隻")
    print("=" * 60)

    process_download(sdate, edate, stock_list)

    finish = t.perf_counter()
    print(f"\n完成，耗時 {round(finish - start, 2)} 秒")
