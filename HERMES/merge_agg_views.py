"""
HKEX 期權合併視圖生成器（指數 + 股票期權全部支援）
為 HSI/HTI/HHI/MCH/MHI + 全部 14x 個股票期權生成「按 strike 聚合所有合約」嘅月度視圖
由本月開始的所有合約 (含本月同未來)，相同 strike 嘅所有數值欄位相加
"""
import os
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================================
# 路徑
# ============================================================
IO_M_ROOT  = Path("/root/GitHub/SData/HKEX/IO_M")
SO_M_ROOT  = Path("/root/GitHub/SData/HKEX/SO_M")
IO_AGG_ROOT = Path("/root/GitHub/SData/HKEX/IO_AGG")
SO_AGG_ROOT = Path("/root/GitHub/SData/HKEX/SO_AGG")

# ============================================================
# 排除子目錄（DATA / TEMP 等）
# ============================================================
EXCLUDE_SUBDIRS = {"DATA", "TEMP", "_TEMP", "BAK", "OLD"}

# ============================================================
# 唔聚合（每個 strike 只取第一個值代表）嘅欄位
# ============================================================
PASSTHROUGH_COLS = ['series']

# ============================================================
# 唔聚合（識別用）嘅欄位
# ============================================================
KEY_COLS = ['date', 'strike']


def get_current_month_num() -> int:
    """取得當前月份（1-12）"""
    return datetime.now().month


def get_current_year_short() -> int:
    """取得當前年份嘅最後兩位（如 26）"""
    return int(str(datetime.now().year)[-2:])


def build_agg_view(src_dir: Path, dst_dir: Path, subdir_name: str, kind: str):
    """
    為單一 IO 或 SO 子目錄生成「按 strike 聚合」嘅月度視圖
    只取由當月開始嘅合約，所有數值欄位都 sum
    kind: "IO" or "SO"  (用嚟 print label)
    """
    if not src_dir.exists():
        print(f"  [SKIP] {kind}/{subdir_name}: 月度檔目錄不存在")
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)

    # 取得當前月份
    cur_month_num = get_current_month_num()
    cur_year_short = get_current_year_short()
    cur_contract_num = cur_year_short * 100 + cur_month_num

    # 找出所有月度檔
    monthly_files = {}
    for f in src_dir.glob("*.csv"):
        m = re.search(r'_(\d{6})\.csv$', f.name)
        if m:
            ym = m.group(1)
            monthly_files[ym] = f

    if not monthly_files:
        print(f"  [SKIP] {kind}/{subdir_name}: 冇月度檔")
        return 0

    written = 0
    for ym, csv_path in sorted(monthly_files.items()):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            print(f"  [WARN] {kind}/{subdir_name}/{csv_path.name}: {e}")
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

        # 構建聚合字典
        # *_settle_price、*_price_change：唔同合約有唔同值，sum 冇意義 → 完全排除
        # *_change（淨變化類）：拆成 _add (正值加) + _reduce (負值加)，避免 6月+1000 + 7月-1000 = 0 嘅假象
        EXCLUDED_COLS = ['call_settle_price', 'put_settle_price', 'call_price_change', 'put_price_change']
        CHANGE_COLS = ['call_net_change', 'put_net_change',
                        'call_turnover_change', 'put_turnover_change',
                        'call_gross_change', 'put_gross_change']

        # 預處理：將 *_change 拆成 _add / _reduce 兩欄（加總後有意義）
        for col in CHANGE_COLS:
            if col in df.columns:
                df[f'{col}_add']    = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)  # 只保留正值
                df[f'{col}_reduce'] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(upper=0)  # 只保留負值（保持負號）
                df.drop(columns=[col], inplace=True)  # 移除原欄，避免 sum 出 0

        agg_dict = {}
        for col in df.columns:
            if col in KEY_COLS or col in ['month_num', 'year_int', 'contract_num', 'month_abbr', 'year']:
                continue
            if col in EXCLUDED_COLS:
                continue
            if col in PASSTHROUGH_COLS:
                agg_dict[col] = 'first'
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = 'sum'

        # 記錄涉及嘅合約月份
        df['contract_label'] = df['month_abbr'] + df['year_int'].astype(str).str.zfill(2)

        # groupby + agg
        grouped = df.groupby(KEY_COLS, dropna=True).agg(agg_dict).reset_index()

        # 重算 ratio（總成交金額 / 總成交筆數 = 每筆平均金額）
        if 'call_turnover' in grouped.columns and 'call_deals' in grouped.columns:
            grouped['call_ratio'] = grouped['call_turnover'] / grouped['call_deals'].replace(0, float('nan'))
        if 'put_turnover' in grouped.columns and 'put_deals' in grouped.columns:
            grouped['put_ratio'] = grouped['put_turnover'] / grouped['put_deals'].replace(0, float('nan'))

        for col in ['call_ratio', 'put_ratio']:
            if col in grouped.columns:
                grouped[col] = grouped[col].round(2)

        # contract_label 用 group 內去重版
        grouped['contract_label'] = df.groupby(KEY_COLS, dropna=True)['contract_label'].apply(
            lambda x: ','.join(sorted(set(x)))
        ).reset_index(drop=True)

        # 重新排序欄位
        all_cols = list(grouped.columns)
        out_cols = ['date', 'series', 'strike']
        for col in all_cols:
            if col in out_cols or col == 'contract_label':
                continue
            out_cols.append(col)
        out_cols.append('contract_label')
        seen = set()
        out_cols = [c for c in out_cols if c in grouped.columns and not (c in seen or seen.add(c))]

        grouped = grouped[out_cols]

        # 排序：先日期，後 strike
        grouped = grouped.sort_values(['date', 'strike']).reset_index(drop=True)

        out_path = dst_dir / f"{subdir_name}_{ym}_AGG.csv"
        grouped.to_csv(out_path, index=False)
        print(f"  ✅ {kind}/{out_path.name}: {len(grouped)} rows ({len(grouped['date'].unique())} dates × {len(grouped['strike'].unique())} strikes, {len(grouped.columns)} cols)")
        written += 1

    return written


def scan_subdirs(root: Path) -> list:
    """
    掃描 root 下嘅子目錄，排除 DATA/TEMP/BAK/OLD
    返回 [(subdir_name, subdir_path), ...]
    """
    if not root.exists():
        return []
    out = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if p.name in EXCLUDE_SUBDIRS or p.name.startswith('.'):
            continue
        # 必須有至少一個月度檔 (*_YYYYMM.csv)
        has_monthly = any(re.search(r'_\d{6}\.csv$', f.name) for f in p.glob("*.csv"))
        if not has_monthly:
            continue
        out.append((p.name, p))
    return out


def main():
    print("=" * 60)
    print("HKEX 期權 — 聚合視圖生成器（IO 指數 + SO 股票期權）")
    print(f"當前月份: {datetime.now().strftime('%Y-%m')}")
    print("=" * 60)
    start = datetime.now()

    total = 0
    total_series = 0

    # 指數期權
    io_subdirs = scan_subdirs(IO_M_ROOT)
    print(f"\n[IO 指數期權] 發現 {len(io_subdirs)} 個系列: {[s[0] for s in io_subdirs]}")
    for subdir_name, subdir_path in io_subdirs:
        n = build_agg_view(subdir_path, IO_AGG_ROOT / subdir_name, subdir_name, "IO")
        total += n
        if n > 0:
            total_series += 1

    # 股票期權
    so_subdirs = scan_subdirs(SO_M_ROOT)
    print(f"\n[SO 股票期權] 發現 {len(so_subdirs)} 個系列")
    for subdir_name, subdir_path in so_subdirs:
        # SO_M 子目錄係 0001_CKH 格式 → 抽出 CKH 用嚟做檔名
        m = re.match(r'^\d+_(.+)$', subdir_name)
        short_name = m.group(1) if m else subdir_name
        n = build_agg_view(subdir_path, SO_AGG_ROOT / subdir_name, subdir_name, "SO")
        total += n
        if n > 0:
            total_series += 1

    print(f"\n{'=' * 60}")
    print(f"完成！耗時 {(datetime.now() - start).total_seconds():.1f}s")
    print(f"總共生成 {total} 個聚合視圖（{total_series} 個系列）")
    print(f"輸出目錄：{IO_AGG_ROOT} + {SO_AGG_ROOT}")


if __name__ == "__main__":
    main()
