"""
HKEX 月度合併檔生成器
將每日 CSV 合併為月度合併檔（{PREFIX}_{YYYYMM}.csv）
"""
import os
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

IO_ROOT  = Path("/root/GitHub/SData/HKEX/IO")
SO_ROOT  = Path("/root/GitHub/SData/HKEX/SO")
IO_M_ROOT = Path("/root/GitHub/SData/HKEX/IO_M")
SO_M_ROOT = Path("/root/GitHub/SData/HKEX/SO_M")

USECOLS = [
    'date', 'series', 'month_num', 'month_abbr', 'year',
    'call_turnover_prev', 'call_turnover', 'call_ratio', 'call_settle_price',
    'call_net_change', 'call_net', 'call_gross_change', 'call_gross_prev', 'call_gross',
    'strike',
    'put_gross', 'put_gross_prev', 'put_gross_change', 'put_net', 'put_net_change',
    'put_turnover', 'put_turnover_prev', 'put_turnover_change', 'put_deals', 'put_ratio',
    'put_price_change', 'put_settle_price',
]

def merge_index_options(index_code: str):
    """合併單一指數期權（HSI/MHI/HTI/HHI/MCH）的每日檔為月度檔"""
    src_dir = IO_ROOT / index_code
    dst_dir = IO_M_ROOT / index_code
    if not src_dir.exists():
        print(f"  [SKIP] {index_code}: 源目錄不存在")
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有每日 CSV，按年月分組
    monthly_files: dict[str, list[Path]] = {}
    for f in src_dir.glob("*.csv"):
        m = re.search(r'_(\d{8})\.csv$', f.name)
        if m:
            ym = m.group(1)[:6]  # YYYYMM
            monthly_files.setdefault(ym, []).append(f)

    written = 0
    for ym, files in sorted(monthly_files.items()):
        # 讀取並合併
        dfs = []
        for f in sorted(files):
            try:
                df = pd.read_csv(f, low_memory=False)
                # 日期從檔名提取（如 HSI_20260102.csv → 20260102）
                date_from_name = re.search(r'_(\d{8})\.csv$', f.name)
                if date_from_name:
                    df['date'] = date_from_name.group(1)
                dfs.append(df)
            except Exception as e:
                print(f"  [WARN] {f.name}: {e}")
                continue

        if not dfs:
            continue

        merged = pd.concat(dfs, ignore_index=True)
        # 按日期排序
        merged = merged.sort_values('date').reset_index(drop=True)
        # 處理 strike / settle_price NaN — 填充為 -1 避免去重時撞 NaN
        merged['strike'] = pd.to_numeric(merged['strike'], errors='coerce').fillna(-1.0)
        merged['call_settle_price'] = pd.to_numeric(merged['call_settle_price'], errors='coerce').fillna(-1.0)
        merged['put_settle_price'] = pd.to_numeric(merged['put_settle_price'], errors='coerce').fillna(-1.0)
        # 去重：同 (date, month, year, strike) 配唔同 settle_price 係唔同合約，必須保留
        # 實際上一個 CSV 內唔會有完全重複嘅 row，所以理論上無需去重
        # 為保險只刪除完全重複（所有欄位都一樣）嘅 row
        merged = merged.drop_duplicates(keep='last')

        out_path = dst_dir / f"{index_code}_{ym}.csv"
        merged.to_csv(out_path, index=False)
        print(f"  ✅ {index_code}/{out_path.name}: {len(merged)} rows, {len(files)} daily files")
        written += 1

    return written


def merge_stock_options(stock_dir_name: str, csv_prefix: str):
    """合併單一股票期權的每日檔為月度檔"""
    src_dir = SO_ROOT / stock_dir_name
    dst_dir = SO_M_ROOT / stock_dir_name
    if not src_dir.exists():
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)

    monthly_files: dict[str, list[Path]] = {}
    for f in src_dir.glob("*.csv"):
        m = re.search(r'_(\d{8})\.csv$', f.name)
        if m:
            ym = m.group(1)[:6]
            monthly_files.setdefault(ym, []).append(f)

    written = 0
    for ym, files in sorted(monthly_files.items()):
        dfs = []
        for f in sorted(files):
            try:
                df = pd.read_csv(f, low_memory=False)
                # 日期從檔名提取（如 CKH_20260527.csv → 20260527）
                date_from_name = re.search(r'_(\d{8})\.csv$', f.name)
                if date_from_name:
                    df['date'] = date_from_name.group(1)
                dfs.append(df)
            except Exception:
                continue

        if not dfs:
            continue

        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.sort_values('date').reset_index(drop=True)
        # 處理 NaN — 填充為 -1 避免去重時撞 NaN
        for col in ['strike', 'call_settle_price', 'put_settle_price']:
            merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(-1.0)
        # 去重：完整 row 完全重複才刪除（同 strike 配不同 settle_price 係唔同合約必須保留）
        merged = merged.drop_duplicates(keep='last')

        out_path = dst_dir / f"{csv_prefix}_{ym}.csv"
        merged.to_csv(out_path, index=False)
        written += 1

    return written


def main():
    print("=" * 60)
    print("HKEX 月度合併檔生成器")
    print("=" * 60)
    start = datetime.now()

    total_written = 0

    # IO 指數期權
    io_indices = ["HSI", "MHI", "HTI", "HHI", "MCH"]
    print(f"\n📊 IO 期權（月度合併）...")
    for idx in io_indices:
        n = merge_index_options(idx)
        total_written += n

    # SO 股票期權（只處理已有月度檔目錄的股票）
    print(f"\n📊 SO 股票期權（月度合併）...")
    so_dirs = [d.name for d in SO_ROOT.iterdir() if d.is_dir()]
    for stock_dir in sorted(so_dirs):
        # CSV prefix = 目錄名去掉股票代碼前綴
        # 例如 "0700_TCH" → prefix "TCH"
        if '_' in stock_dir:
            prefix = stock_dir.split('_', 1)[1]
        else:
            prefix = stock_dir
        n = merge_stock_options(stock_dir, prefix)
        if n > 0:
            print(f"  ✅ {stock_dir}: {n} 個月")

    print(f"\n完成！耗時 {(datetime.now() - start).total_seconds():.1f}s")
    print(f"總共生成 {total_written} 個 IO 月度檔")


if __name__ == "__main__":
    main()