"""
RunOI_Daily.py — HKEX OI 每日整合下載與解析
整合三個房間：
  1. IndexOption  (IO 指數期權)
  2. StockOption  (SO 股票期權)
  3. IndexFuture  (IF 指數期貨)

用法：
  python RunOI_Daily.py                        # 自動判斷日期（收盤前用昨日，收盤後用今日）
  python RunOI_Daily.py --sdate 20260504       # 指定日期
  python RunOI_Daily.py --sdate 20260501 --edate 20260504  # 區間
"""
import sys
import os
import re

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import UTIL.CommonConfig as cc
import threading

HK_TZ = ZoneInfo('Asia/Hong_Kong')

# ============================================================
# 路徑
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
IOPATH = Path("/root/GitHub/SData/HKEX/IO")
SOPATH = Path("/root/GitHub/SData/HKEX/SO")
IFPATH = Path("/root/GitHub/SData/HKEX/IF")

# ============================================================
# Room 1：IndexOption（IO 指數期權）
# ============================================================
def io_download_file(odate):
    url = f"https://www.hkex.com.hk/eng/stat/dmstat/oi/DTOP_F_{odate}.zip"
    local_filename = IOPATH / "DATA" / f"{odate}.zip"
    try:
        with cc.requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            local_filename.parent.mkdir(parents=True, exist_ok=True)
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"[IO] 下載成功: {local_filename}")
        return True
    except Exception as e:
        print(f"[IO] 下載失敗 {local_filename}: {e}")
        return False

def io_extract_data(odate):
    raw_filename = f"{odate}_1_dtop_f_hkcc_opt_dtl_all.raw"
    zip_filename = IOPATH / "DATA" / f"{odate}.zip"
    extract_dir = IOPATH / "TEMP"
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with cc.zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            if raw_filename not in zip_ref.namelist():
                raise FileNotFoundError(f"ZIP中未找到 {raw_filename}")
            zip_ref.extract(raw_filename, path=extract_dir)
            raw_file_path = extract_dir / raw_filename
    except Exception as e:
        print(f"[IO] 解壓失敗: {e}")
        return

    df = cc.pd.read_csv(raw_file_path, encoding='utf-8', skiprows=1)
    df.columns = [
        'record_type', 'product_code', 'product_desc', 'series',
        'month_num', 'month_abbr', 'year', 'strike',
        'call_gross', 'call_net', 'call_net_change', 'call_turnover',
        'call_deals', 'call_settle_price', 'call_price_change',
        'put_gross', 'put_net', 'put_net_change', 'put_turnover',
        'put_deals', 'put_settle_price', 'put_price_change'
    ]

    df['call_ratio'] = round(df['call_turnover'].astype(float) /
                              df['call_deals'].astype(float).replace(0, float('nan')), 2)
    df['put_ratio']  = round(df['put_turnover'].astype(float)  /
                              df['put_deals'].astype(float).replace(0, float('nan')),  2)

    desired_order = [
        'series', 'month_num', 'month_abbr', 'year', 'strike',
        'call_deals', 'call_turnover', 'call_ratio',
        'call_settle_price', 'call_price_change',
        'call_net_change', 'call_net', 'call_gross',
        'put_deals', 'put_turnover', 'put_ratio',
        'put_settle_price', 'put_price_change',
        'put_net_change', 'put_net', 'put_gross',
    ]
    df_op = df[desired_order]

    # 計算 change（對比上個工作日）
    pdate = cc.getLastWorkday(cc.datetime.strptime(odate, "%Y%m%d").date()).strftime("%Y%m%d")
    prev_cols   = ['call_turnover_prev', 'call_gross_prev', 'put_gross_prev', 'put_turnover_prev']
    change_cols = ['call_turnover_change', 'call_gross_change', 'put_gross_change', 'put_turnover_change']
    merge_key   = ['series', 'month_num', 'month_abbr', 'year', 'strike']

    for op in ['HSI', 'MHI', 'HTI', 'HHI', 'MCH']:
        op_dir = IOPATH / op
        op_dir.mkdir(parents=True, exist_ok=True)
        tempdf = df_op.loc[df_op['series'] == op].copy()

        prev_file = op_dir / f"{op}_{pdate}.csv"
        if prev_file.exists():
            prev_df = cc.pd.read_csv(prev_file, usecols=merge_key + ['call_turnover', 'call_gross', 'put_gross', 'put_turnover'])
            prev_df = prev_df.rename(columns={
                'call_turnover': 'call_turnover_prev',
                'call_gross':    'call_gross_prev',
                'put_gross':     'put_gross_prev',
                'put_turnover':  'put_turnover_prev'
            })
            tempdf = tempdf.merge(prev_df, on=merge_key, how='left')
            for cur, prev, chg in zip(
                    ['call_turnover', 'call_gross', 'put_gross', 'put_turnover'],
                    ['call_turnover_prev', 'call_gross_prev', 'put_gross_prev', 'put_turnover_prev'],
                    ['call_turnover_change', 'call_gross_change', 'put_gross_change', 'put_turnover_change']):
                tempdf[chg] = (tempdf[cur] - tempdf[prev]).round(2)
        else:
            for col in prev_cols + change_cols:
                tempdf[col] = float('nan')

        final_order = [
            'series', 'month_num', 'month_abbr', 'year',
            'call_settle_price', 'call_price_change', 'call_ratio',
            'call_deals',
            'call_turnover_change', 'call_turnover_prev', 'call_turnover',
            'call_net_change', 'call_net',
            'call_gross_change', 'call_gross_prev', 'call_gross',
            'strike',
            'put_gross', 'put_gross_prev', 'put_gross_change',
            'put_net', 'put_net_change',
            'put_turnover', 'put_turnover_prev', 'put_turnover_change',
            'put_deals', 'put_ratio',
            'put_price_change', 'put_settle_price',
        ]
        tempdf = tempdf.reindex(columns=final_order)
        tempdf.to_csv(op_dir / f"{op}_{odate}.csv", index=False)

    os.remove(raw_file_path)

def ProcessIO(sdate, edate):
    """下載 + 解壓 IO"""
    print("\n" + "=" * 50)
    print("  Room 1：IndexOption（IO 指數期權）")
    print("=" * 50)
    dates = cc.pd.date_range(cc.datetime.strptime(sdate, "%Y%m%d"),
                              cc.datetime.strptime(edate, "%Y%m%d"),
                              freq='D').strftime("%Y%m%d").tolist()
    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        list(cc.tqdm(executor.map(io_download_file, dates, chunksize=1), total=len(dates)))
    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        list(cc.tqdm(executor.map(io_extract_data, dates, chunksize=1), total=len(dates)))

# ============================================================
# Room 2：StockOption（SO 股票期權）
# ============================================================
def so_download_file(odate):
    url = f"https://www.hkex.com.hk/eng/stat/dmstat/oi/DTOP_O_{odate}.zip"
    local_filename = SOPATH / "DATA" / f"{odate}.zip"
    try:
        with cc.requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            local_filename.parent.mkdir(parents=True, exist_ok=True)
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"[SO] 下載成功: {local_filename}")
        return True
    except Exception as e:
        print(f"[SO] 下載失敗 {local_filename}: {e}")
def so_extract_data(odate):
    raw_filename = f"{odate}_1_dtop_o_seoch_opt_dtl_all.raw"
    zip_filename = SOPATH / "DATA" / f"{odate}.zip"
    extract_dir = SOPATH / "TEMP"
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with cc.zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            if raw_filename not in zip_ref.namelist():
                raise FileNotFoundError(f"ZIP中未找到 {raw_filename}")
            zip_ref.extract(raw_filename, path=extract_dir)
            raw_file_path = extract_dir / raw_filename
    except Exception as e:
        print(f"[SO] 解壓失敗: {e}")
        return

    df = cc.pd.read_csv(raw_file_path, encoding='utf-8', skiprows=1)
    df.columns = [
        'record_type', 'product_code', 'product_desc', 'series',
        'month_num', 'month_abbr', 'year', 'strike',
        'call_gross', 'call_net', 'call_net_change', 'call_turnover',
        'call_deals', 'call_settle_price', 'call_price_change',
        'put_gross', 'put_net', 'put_net_change', 'put_turnover',
        'put_deals', 'put_settle_price', 'put_price_change'
    ]

    df['call_ratio'] = round(df['call_turnover'].astype(float) /
                              df['call_deals'].astype(float).replace(0, float('nan')), 2)
    df['put_ratio']  = round(df['put_turnover'].astype(float)  /
                              df['put_deals'].astype(float).replace(0, float('nan')),  2)

    desired_order = [
        'series', 'month_num', 'month_abbr', 'year', 'strike',
        'call_deals', 'call_turnover', 'call_ratio',
        'call_settle_price', 'call_price_change',
        'call_net_change', 'call_net', 'call_gross',
        'put_deals', 'put_turnover', 'put_ratio',
        'put_settle_price', 'put_price_change',
        'put_net_change', 'put_net', 'put_gross',
    ]
    df_op = df[desired_order]

    pdate = cc.getLastWorkday(cc.datetime.strptime(odate, "%Y%m%d").date()).strftime("%Y%m%d")
    prev_cols   = ['call_turnover_prev', 'call_gross_prev', 'put_gross_prev', 'put_turnover_prev']
    change_cols = ['call_turnover_change', 'call_gross_change', 'put_gross_change', 'put_turnover_change']
    merge_key   = ['series', 'month_num', 'month_abbr', 'year', 'strike']

    # 動態掃描 SO 目錄，建立股票代碼列表
    stock_map = {}  # { 'CKH': '0001_CKH', ... }
    if SOPATH.exists():
        for d in SOPATH.iterdir():
            if d.is_dir():
                m = re.match(r'^(\d+)_(.+)$', d.name)
                if m:
                    stock_map[m.group(2)] = d.name   # hkats → dir_name

    for hkats, dir_name in stock_map.items():
        op_dir = SOPATH / dir_name
        op_dir.mkdir(parents=True, exist_ok=True)

        tempdf = df_op.loc[df_op['series'] == hkats].copy()
        for col in merge_key:
            tempdf[col] = tempdf[col].astype(str)

        prev_file = op_dir / f"{hkats}_{pdate}.csv"
        if prev_file.exists():
            prev_df = cc.pd.read_csv(prev_file, usecols=merge_key + ['call_turnover', 'call_gross', 'put_gross', 'put_turnover'])
            for col in merge_key:
                prev_df[col] = prev_df[col].astype(str)
            prev_df = prev_df.rename(columns={
                'call_turnover': 'call_turnover_prev',
                'call_gross':    'call_gross_prev',
                'put_gross':     'put_gross_prev',
                'put_turnover':  'put_turnover_prev'
            })
            tempdf = tempdf.merge(prev_df, on=merge_key, how='left')
            for cur, prev, chg in zip(
                    ['call_turnover', 'call_gross', 'put_gross', 'put_turnover'],
                    ['call_turnover_prev', 'call_gross_prev', 'put_gross_prev', 'put_turnover_prev'],
                    ['call_turnover_change', 'call_gross_change', 'put_gross_change', 'put_turnover_change']):
                tempdf[chg] = (tempdf[cur] - tempdf[prev]).round(2)
        else:
            for col in prev_cols + change_cols:
                tempdf[col] = float('nan')

        final_order = [
            'series', 'month_num', 'month_abbr', 'year',
            'call_settle_price', 'call_price_change', 'call_ratio',
            'call_deals',
            'call_turnover_change', 'call_turnover_prev', 'call_turnover',
            'call_net_change', 'call_net',
            'call_gross_change', 'call_gross_prev', 'call_gross',
            'strike',
            'put_gross', 'put_gross_prev', 'put_gross_change',
            'put_net', 'put_net_change',
            'put_turnover', 'put_turnover_prev', 'put_turnover_change',
            'put_deals', 'put_ratio',
            'put_price_change', 'put_settle_price',
        ]
        tempdf = tempdf.reindex(columns=final_order)
        tempdf.to_csv(op_dir / f"{hkats}_{odate}.csv", index=False)

    os.remove(raw_file_path)

def ProcessSO(sdate, edate):
    """下載 + 解壓 SO"""
    print("\n" + "=" * 50)
    print("  Room 2：StockOption（SO 股票期權）")
    print("=" * 50)
    dates = cc.pd.date_range(cc.datetime.strptime(sdate, "%Y%m%d"),
                              cc.datetime.strptime(edate, "%Y%m%d"),
                              freq='D').strftime("%Y%m%d").tolist()
    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        list(cc.tqdm(executor.map(so_download_file, dates, chunksize=1), total=len(dates)))
    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        list(cc.tqdm(executor.map(so_extract_data, dates, chunksize=1), total=len(dates)))

# ============================================================
# Room 3：IndexFuture（IF 指數期貨）
# 備註：IndexFuture 的 raw 檔在 IO zip 內，所以依賴 Room 1 完成
# ============================================================
def if_extract_data(odate):
    raw_filename = f"{odate}_1_dtop_f_hkcc_fut_dtl_all.raw"
    zip_filename = IOPATH / "DATA" / f"{odate}.zip"
    extract_dir = IOPATH / "TEMP"
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with cc.zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            if raw_filename not in zip_ref.namelist():
                raise FileNotFoundError(f"ZIP中未找到 {raw_filename}")
            zip_ref.extract(raw_filename, path=extract_dir)
            raw_file_path = extract_dir / raw_filename
    except Exception as e:
        print(f"[IF] 解壓失敗: {e}")
        return

    df = cc.pd.read_csv(raw_file_path, encoding='utf-8', skiprows=1)
    df.columns = [
        'record_type', 'product_code', 'product_desc', 'series',
        'month_num', 'month_abbr', 'year',
        'gross', 'net', 'net_change', 'turnover', 'deals',
        'settle_price', 'price_change',
    ]

    df['ratio'] = round(df['turnover'].astype(float) / df['deals'].astype(float).replace(0, float('nan')), 2)

    desired_order = [
        'series', 'month_num', 'month_abbr', 'year',
        'gross', 'net', 'net_change', 'turnover', 'deals', 'ratio',
        'settle_price', 'price_change'
    ]
    df_if = df[desired_order]

    for op in ['HSI', 'MHI', 'HTI', 'HHI', 'MCH']:
        op_dir = IFPATH / op
        op_dir.mkdir(parents=True, exist_ok=True)
        tempdf = df_if.loc[df_if['series'] == op].copy()
        tempdf.to_csv(op_dir / f"{op}_{odate}.csv", index=False)

    os.remove(raw_file_path)

def ProcessIF(sdate, edate):
    """解壓 IF（依賴 IO zip）"""
    print("\n" + "=" * 50)
    print("  Room 3：IndexFuture（IF 指數期貨）")
    print("=" * 50)
    dates = cc.pd.date_range(cc.datetime.strptime(sdate, "%Y%m%d"),
                              cc.datetime.strptime(edate, "%Y%m%d"),
                              freq='D').strftime("%Y%m%d").tolist()
    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        list(cc.tqdm(executor.map(if_extract_data, dates, chunksize=1), total=len(dates)))

# ============================================================
# 主程式
# ============================================================
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="HKEX OI 每日整合下載與解析")
    parser.add_argument("--sdate", type=str, help="起始日期 (YYYYMMDD)，預設為前一工作日")
    parser.add_argument("--edate", type=str, help="結束日期 (YYYYMMDD)，預設為前一工作日")
    args = parser.parse_args()

    start = cc.t.perf_counter()

    if args.sdate and args.edate:
        sdate = args.sdate
        edate = args.edate
    else:
        now = datetime.now(HK_TZ)
        last_wd = cc.getLastWorkday(cc.date.today())
        if now.hour < 20:
            sdate = last_wd.strftime("%Y%m%d")
            edate = last_wd.strftime("%Y%m%d")
        else:
            sdate = now.strftime("%Y%m%d")
            edate = now.strftime("%Y%m%d")

    print("=" * 50)
    print("  HKEX OI 每日整合任務")
    print(f"  日期區間：{sdate} ~ {edate}")
    print(f"  使用時區：Asia/Hong_Kong（GMT+8）")
    print("=" * 50)

    # Room 1 + 2 同時下載（兩者獨立）
    ProcessIO(sdate, edate)
    ProcessSO(sdate, edate)

    # Room 3 依賴 Room 1 的 zip
    ProcessIF(sdate, edate)

    finish = cc.t.perf_counter()
    print("\n" + "=" * 50)
    print(f"  全部完成！耗時 {round(finish-start, 2)} 秒")
    print("=" * 50)
