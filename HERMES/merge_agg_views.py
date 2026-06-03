"""
HKEX 指數期權合併視圖生成器
為 HSI / HTI / HHI 生成「按 strike 聚合所有合約」嘅月度視圖
由本月開始的所有合約 (含本月同未來)，相同 strike 嘅 call_net / put_net / 各成交量相加
"""
import os
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

IO_M_ROOT  = Path("/root/GitHub/SData/HKEX/IO_M")
IO_AGG_ROOT = Path("/root/GitHub/SData/HKEX/IO_AGG")

# 要生成嘅指數
INDICES = ["HSI", "HTI", "HHI"]

# 要聚合嘅數值欄位
AGG_COLS = {
    "call_net_change": "sum",
    "call_net": "sum",
    "call_turnover": "sum",
    "call_turnover_prev": "sum",
    "call_gross_change": "sum",
    "call_gross": "sum",
    "call_deals": "sum",
    "put_net_change": "sum",
    "put_net": "sum",
    "put_turnover": "sum",
    "put_turnover_prev": "sum",
    "put_gross_change": "sum",
    "put_gross": "sum",
    "put_deals": "sum",
}


def get_current_month_num() -> int:
    """取得當前月份（1-12）"""
    return datetime.now().month


def get_current_year_short() -> int:
    """取得當前年份嘅最後兩位（如 26）"""
    return int(str(datetime.now().year)[-2:])


def build_agg_view(index_code: str):
    """
    為單一指數生成「按 strike 聚合」嘅月度視圖
    只取由當月開始嘅合約
    """
    src_dir = IO_M_ROOT / index_code
    dst_dir = IO_AGG_ROOT / index_code
    if not src_dir.exists():
        print(f"  [SKIP] {index_code}: 月度檔目錄不存在")
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)

    # 取得當前月份
    cur_month_num = get_current_month_num()
    cur_year_short = get_current_year_short()
    # 用 2 位年份，例如 26 而非 2026，跟 year_int 對齊
    cur_contract_num = cur_year_short * 100 + cur_month_num

    # 找出所有月度檔
    monthly_files = {}
    for f in src_dir.glob("*.csv"):
        m = re.search(r'_(\d{6})\.csv$', f.name)
        if m:
            ym = m.group(1)
            monthly_files[ym] = f

    if not monthly_files:
        print(f"  [SKIP] {index_code}: 冇月度檔")
        return 0

    written = 0
    for ym, csv_path in sorted(monthly_files.items()):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            print(f"  [WARN] {index_code}/{csv_path.name}: {e}")
            continue

        # 過濾：由當月開始嘅合約
        month_to_num = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        df['month_num'] = df['month_abbr'].astype(str).map(month_to_num)
        df['year_int'] = df['year'].fillna(0).astype(int)
        df['contract_num'] = df['year_int'] * 100 + df['month_num']
        df = df[df['contract_num'] >= cur_contract_num]

        if df.empty:
            continue

        # 處理 strike NaN
        df['strike'] = pd.to_numeric(df['strike'], errors='coerce')

        # 為每個 (date, strike) 聚合
        agg_dict = {col: 'sum' for col in AGG_COLS if col in df.columns}
        # 用 first 取 call_settle_price 同 put_settle_price 嘅代表值
        first_cols = ['call_settle_price', 'put_settle_price', 'series']
        for col in first_cols:
            if col in df.columns:
                agg_dict[col] = 'first'

        # 記錄涉及嘅合約月份（每個 (date, strike) 內先 group 去重再加逗號）
        df['contract_label'] = df['month_abbr'] + df['year_int'].astype(str).str.zfill(2)

        grouped = df.groupby(['date', 'strike'], dropna=True).agg(agg_dict).reset_index()
        # contract_label 用 group 內去重版
        grouped['contract_label'] = df.groupby(['date', 'strike'], dropna=True)['contract_label'].apply(
            lambda x: ','.join(sorted(set(x)))
        ).reset_index(drop=True)

        # 重新排序欄位
        out_cols = [
            'date', 'series', 'strike',
            'call_settle_price', 'call_net_change', 'call_net',
            'call_turnover', 'call_turnover_prev', 'call_deals',
            'call_gross', 'call_gross_change',
            'put_settle_price', 'put_net_change', 'put_net',
            'put_turnover', 'put_turnover_prev', 'put_deals',
            'put_gross', 'put_gross_change',
            'contract_label',
        ]
        # 只保留存在嘅欄位
        out_cols = [c for c in out_cols if c in grouped.columns]
        grouped = grouped[out_cols]

        # 排序：先日期，後 strike
        grouped = grouped.sort_values(['date', 'strike']).reset_index(drop=True)

        out_path = dst_dir / f"{index_code}_{ym}_AGG.csv"
        grouped.to_csv(out_path, index=False)
        print(f"  ✅ {index_code}/{out_path.name}: {len(grouped)} rows ({len(grouped['date'].unique())} dates × {len(grouped['strike'].unique())} strikes)")
        written += 1

    return written


def main():
    print("=" * 60)
    print("HKEX 指數期權 — 聚合視圖生成器")
    print(f"當前月份: {datetime.now().strftime('%Y-%m')}")
    print("=" * 60)
    start = datetime.now()

    total = 0
    for idx in INDICES:
        n = build_agg_view(idx)
        total += n

    print(f"\n完成！耗時 {(datetime.now() - start).total_seconds():.1f}s")
    print(f"總共生成 {total} 個聚合視圖")


if __name__ == "__main__":
    main()