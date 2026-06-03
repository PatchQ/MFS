"""
HKEX Option Viewer — Stock & Index Options
按產品代碼 + 日期 + 月份/年份 篩選 SO/IO CSV，顯示 29 欄完整表格
純數字（00700）→ 股票期權（SO）
純字母（HSI）→ 指數期權（IO）
"""
import os
import glob
import re
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
import calendar
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ============================================================
# 路徑設定
# ============================================================
SO_ROOT = Path("/root/GitHub/SData/HKEX/SO")
IO_ROOT = Path("/root/GitHub/SData/HKEX/IO")
SP_ROOT = Path("/root/GitHub/SData/HKEX/SP")
IF_ROOT = Path("/root/GitHub/SData/HKEX/IF")
# 月度合併檔（加速 scan）
SO_M_ROOT = Path("/root/GitHub/SData/HKEX/SO_M")
IO_M_ROOT = Path("/root/GitHub/SData/HKEX/IO_M")

# IF 指數列表
IF_INDICES = [
    {"code": "HSI", "label": "HSI (恒生指數期貨)", "yf_code": "^HSI"},
    {"code": "HTI", "label": "HTI (恒生科指期貨)", "yf_code": "HSTECH.HK"},
    {"code": "HHI", "label": "HHI (恒生中國企業指數)", "yf_code": "^HSCE"},
]

# ============================================================
# 固定月份列表（與 year 組合形成完整到期日）
# ============================================================
FIXED_MONTH_ABBR = [
    "JAN", "FEB", "MAR", "APR",
    "MAY", "JUN", "JUL", "AUG",
    "SEP", "OCT", "NOV", "DEC",
]
FIXED_YEARS = [str(y) for y in range(26, 37)]  # 26 ~ 36

# ============================================================
# 中英欄位對照表
# ============================================================
COLUMN_NAMES_CN = {
    "series":              None,
    "month_num":           "日",
    "month_abbr":          "月",
    "year":                "年",
    "call_settle_price":   "C價",
    "call_price_change":   "C價c",
    "call_ratio":          "C比率",
    "call_deals":          "C數",
    "call_turnover_change":"CVolc",
    "call_turnover_prev":  "C上日Vol",
    "call_turnover":       "CVol",
    "call_net_change":     "C淨數c",
    "call_net":            "C淨數",
    "call_gross_change":   "COIc",
    "call_gross_prev":     "C上日OI",
    "call_gross":          "COI",
    "strike":              "行使價",
    "put_gross":           "POI",
    "put_gross_prev":      "P上日OI",
    "put_gross_change":    "POIc",
    "put_net":             "P淨數",
    "put_net_change":      "P淨數c",
    "put_turnover":        "PVol",
    "put_turnover_prev":   "P上日Vol",
    "put_turnover_change": "PVolc",
    "put_deals":           "P數",
    "put_ratio":           "P比率",
    "put_price_change":    "P價c",
    "put_settle_price":    "P價",
}

# ============================================================
# 工具函式
# ============================================================
def is_io_code(code):
    """純字母 → IO 指數期權"""
    return code.isalpha()

def is_so_code(code):
    """純數字 → SO 股票期權"""
    return code.isdigit()

def get_product_list():
    """
    回傳所有可用產品（SO + IO + IS），每項：
    { code, type: 'SO'|'IO'|'IS', label }
    """
    products = []

    # SO（股票期權）
    if SO_ROOT.exists():
        for d in SO_ROOT.iterdir():
            if d.is_dir():
                m = re.match(r'^(\d+)_(.+)$', d.name)
                if m:
                    code = m.group(1)
                    name = m.group(2)
                    products.append({
                        "code": code,
                        "short_name": name,  # CSV prefix, e.g. CTS
                        "dir_name": d.name,   # full dir e.g. 6030_CTS
                        "type": "SO",
                        "label": f"{code} {name}"
                    })

    # IO（指數期權）— HSI, HTI, HHI
    IO_INDICES = ["HSI", "HTI", "HHI"]
    for idx in IO_INDICES:
        products.append({
            "code": idx,
            "type": "IO",
            "label": idx
        })

    return products

def get_available_dates(product_dir_or_index, product_type, date_str=None):
    """
    SO: product_dir_or_index = '0700_TCH'，從 SO_ROOT/下找 csv
    IO: product_dir_or_index = 'HSI'，從 IO_ROOT/HSI/ 下找 csv
    回傳 ['20260430', '20260504', ...]，倒序
    """
    dates = []

    if product_type == "SO":
        stock_path = SO_ROOT / product_dir_or_index
        if not stock_path.exists():
            return dates
        for f in stock_path.glob("*.csv"):
            m = re.search(r'(\d{8})\.csv$', f.name)
            if m:
                dates.append(m.group(1))

    elif product_type == "IO":
        io_path = IO_ROOT / product_dir_or_index
        if not io_path.exists():
            return dates
        for f in io_path.glob("*.csv"):
            m = re.search(r'(\d{8})\.csv$', f.name)
            if m:
                dates.append(m.group(1))

    dates.sort(reverse=True)
    return dates

def normalize_so_code(code):
    """SO 代碼：5位→4位，4位直接回傳"""
    code = code.strip()
    if code.isdigit() and len(code) == 5:
        return f"{int(code):04d}"
    return code  # 已是4位或非純數字

def resolve_product(code):
    """
    根據產品代碼解析出 (type, dir_or_index, name_for_csv)
    - SO: ('SO', '0700_TCH', 'TCH')  → csv: TCH_20260430.csv
    - IO: ('IO', 'HSI', 'HSI')        → csv: HSI_20260430.csv
    """
    code = code.strip()

    if is_io_code(code):
        return ("IO", code, code)

    elif is_so_code(code):
        # 5位→4位標準化（00700 → 0700）
        std_code = normalize_so_code(code)
        # 在 SO_ROOT 找到對應目錄（目錄格式：0700_TCH）
        if SO_ROOT.exists():
            for d in SO_ROOT.iterdir():
                if d.is_dir() and d.name.startswith(std_code + "_"):
                    # d.name = '0700_TCH' → short_name = 'TCH'
                    short_name = d.name.split('_', 1)[1]
                    return ("SO", d.name, short_name)
        # 找不到 → 回傳 None
        return None

    return None

def parse_date_display(date_str):
    """ '20260430' → '2026-04-30' """
    if len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return date_str

def format_number(val):
    """格式化數值：加千分位，NaN/None → 空字串"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    try:
        f = float(val)
        if f == int(f):
            return f"{int(f):,}"
        else:
            return f"{f:,.2f}"
    except (TypeError, ValueError):
        return str(val)

# ============================================================
# 路由
# ============================================================
from zoneinfo import ZoneInfo
HK_TZ = ZoneInfo('Asia/Hong_Kong')
NOW = datetime.now(HK_TZ)
CURRENT_MONTH = NOW.strftime("%b").upper()
CURRENT_YEAR  = str(NOW.year)[-2:]

@app.route("/")
def index():
    products = get_product_list()
    return render_template(
        "index.html",
        products=products,
        month_abbr_list=FIXED_MONTH_ABBR,
        year_list=FIXED_YEARS,
        col_names=COLUMN_NAMES_CN,
        current_month=CURRENT_MONTH,
        current_year=CURRENT_YEAR,
    )

@app.route("/api/products")
def api_products():
    """回傳所有可用產品（SO + IO）"""
    return jsonify(get_product_list())

@app.route("/api/dates", methods=["GET"])
def api_dates():
    """
    GET ?code=00700  或  ?code=HSI
    回應該產品的可用日期列表
    """
    code = request.args.get("code", "").strip()
    resolved = resolve_product(code)
    if not resolved:
        return jsonify({"error": f"找不到產品：{code}"}), 404

    product_type, dir_or_index, _ = resolved
    dates = get_available_dates(dir_or_index, product_type)
    return jsonify({
        "dates": dates,
        "display": [parse_date_display(d) for d in dates],
        "type": product_type,
    })

@app.route("/api/data", methods=["GET"])
def api_data():
    code     = request.args.get("code", "").strip()
    date_str = request.args.get("date", "")
    m_abbr   = request.args.get("month_abbr", "").upper()
    year_val = request.args.get("year", "")

    resolved = resolve_product(code)
    if not resolved:
        return jsonify({"error": f"找不到產品：{code}"}), 404

    product_type, dir_or_index, name_for_csv = resolved

    if product_type == "SO":
        csv_path = SO_ROOT / dir_or_index / f"{name_for_csv}_{date_str}.csv"
    else:  # IO
        csv_path = IO_ROOT / dir_or_index / f"{name_for_csv}_{date_str}.csv"

    if not csv_path.exists():
        return jsonify({"error": f"找不到檔案：{csv_path}"}), 404

    try:
        import numpy as np

        df = pd.read_csv(csv_path)

        # 月份/年份 篩選
        if m_abbr:
            df = df[df['month_abbr'].astype(str).str.upper() == m_abbr]
        if year_val:
            df = df[df['year'].astype(str).str.replace('.0','',regex=False) == year_val]

        # strike 空值過濾
        df = df.dropna(subset=['strike'])
        df = df[df['strike'] != '']

        df = df.fillna("")

        return jsonify({
            "columns": list(df.columns),
            "rows": df.values.tolist(),
            "code": code,
            "type": product_type,
            "date": date_str,
            "date_display": parse_date_display(date_str),
            "total_rows": len(df),
            "col_names_cn": COLUMN_NAMES_CN,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# OI 期權篩選掃描 API（跨所有產品）
# ============================================================
@app.route("/api/scan", methods=["GET"])
def api_scan():
    """
    掃描所有 SO + IO 產品，根據條件篩選

    GET params:
      date_from: YYYYMMDD
      date_to:   YYYYMMDD
      month_start / year_start: 合約月份範圍起點（可選）
      month_end   / year_end:   合約月份範圍終點（可選）
      threshold:  淨數變化閾值（默認 1000）
      side:       call | put | both（默認 both）
    """
    date_from   = request.args.get("date_from", "").strip()
    date_to     = request.args.get("date_to",   "").strip()
    month_start = request.args.get("month_start", "").upper()
    year_start  = request.args.get("year_start",  "").strip()
    month_end   = request.args.get("month_end",   "").upper()
    year_end    = request.args.get("year_end",    "").strip()
    threshold   = int(request.args.get("threshold", 1000))
    side        = request.args.get("side", "both").lower()
    product_type = request.args.get("product_type", "all").lower()  # all | io | so

    if not date_from or not date_to:
        return jsonify({"error": "需要 date_from 和 date_to"}), 400

    # ── 月份範圍起點/終點 → 數值用於比較 ───────────────
    def month_year_to_num(m_abbr, yr):
        if not m_abbr or not yr:
            return None
        try:
            mi = FIXED_MONTH_ABBR.index(m_abbr) + 1
            yi = int(yr)
            return yi * 100 + mi  # e.g. 2626 = 2026 JUN
        except (ValueError, IndexError):
            return None

    start_num = month_year_to_num(month_start, year_start)
    end_num   = month_year_to_num(month_end,   year_end)

    # ── 收集所有要掃描的年月 ─────────────────────────
    # date_from/date_to (YYYYMMDD) → 年月 (YYYYMM)
    ym_from = date_from[:6] if len(date_from) >= 6 else None
    ym_to   = date_to[:6]   if len(date_to)   >= 6 else None

    # ── 掃描所有產品 + 月份 ─────────────────────────
    # 欄位順序：產品、日期、合約、C→P（行使價在中間）
    OUTPUT_COLS = [
        "code", "date", "month_label",
        "call_turnover_prev", "call_turnover", "call_ratio", "call_settle_price",
        "call_net_change", "call_net",
        "strike",
        "put_net", "put_net_change", "put_settle_price", "put_ratio",
        "put_turnover", "put_gross_prev",
    ]
    USECOLS = [
        'date', 'month_abbr', 'year',
        'call_turnover_prev', 'call_turnover', 'call_ratio', 'call_settle_price',
        'call_net_change', 'call_net',
        'strike',
        'put_net', 'put_net_change', 'put_settle_price', 'put_ratio',
        'put_turnover', 'put_gross_prev',
    ]
    MAX_ROWS = 5000
    result_rows: list[list] = []

    products = get_product_list()
    product_type = request.args.get("product_type", "all").lower()  # all | so | hsi | hti | hhi
    stock_codes_raw = request.args.get("stock_codes", "").strip()  # e.g. "9992, 0175, 2318"

    # 優先：自定義股票編號清單（覆蓋 product_type）
    if stock_codes_raw:
        code_list = [c.strip().zfill(4) for c in stock_codes_raw.split(",") if c.strip().isdigit() or c.strip().lstrip('0').isdigit()]
        products = [p for p in products if p["code"] in code_list and p["type"] == "SO"]
    elif product_type == 'all':
        pass  # scan all
    elif product_type == 'so':
        products = [p for p in products if p["type"] == "SO"]
    else:  # HSI | HTI | HHI — specific index
        products = [p for p in products if p["code"] == product_type.upper()]

    for p in products:
        if len(result_rows) >= MAX_ROWS:
            break
        pcode  = p["code"]
        ptype  = p["type"]

        # 月度檔根目錄
        if ptype == "SO":
            so_dir = p.get("dir_name") or p["code"]
            scan_root = SO_M_ROOT / so_dir if SO_M_ROOT.exists() else None
            # fallback to daily
            if not scan_root or not scan_root.exists():
                scan_root = SO_ROOT / so_dir
            csv_prefix = p.get("short_name") or pcode
        else:
            scan_root = IO_M_ROOT / pcode if IO_M_ROOT.exists() else None
            if not scan_root or not scan_root.exists():
                scan_root = IO_ROOT / pcode
            csv_prefix = pcode

        if not scan_root.exists():
            continue

        # 找出所有月份檔
        monthly_files = {}
        for f in scan_root.glob("*.csv"):
            # 檔名格式: CKH_202605.csv → ym=202605
            m = re.search(r'_(\d{6})\.csv$', f.name)
            if m:
                ym = m.group(1)
                if ym_from and ym_to and not (ym_from <= ym <= ym_to):
                    continue
                monthly_files[ym] = f

        for ym, csv_path in sorted(monthly_files.items()):
            if len(result_rows) >= MAX_ROWS:
                break

            try:
                df = pd.read_csv(csv_path, usecols=USECOLS)
            except Exception:
                continue

            df = df.fillna("")

            # 日期範圍過濾（精確到日）
            if 'date' in df.columns:
                df['date'] = df['date'].astype(str).str[:8]
                df = df[(df['date'] >= str(date_from)) & (df['date'] <= str(date_to))]
                if df.empty:
                    continue

            # 月份範圍 filter
            if start_num is not None:
                mynums = df.apply(
                    lambda r: month_year_to_num(str(r.get('month_abbr', '')),
                                                str(r.get('year', '')).replace('.0', '')) or 0,
                    axis=1)
                df = df[mynums >= start_num]
            if end_num is not None:
                mynums = df.apply(
                    lambda r: month_year_to_num(str(r.get('month_abbr', '')),
                                                str(r.get('year', '')).replace('.0', '')) or 0,
                    axis=1)
                df = df[mynums <= end_num]

            # 淨數變化 filter（取絕對值，正負都算）
            cn_col = df['call_net_change'] if 'call_net_change' in df.columns else pd.Series([0]*len(df))
            pn_col = df['put_net_change']  if 'put_net_change'  in df.columns else pd.Series([0]*len(df))
            mask_call = pd.to_numeric(cn_col, errors='coerce').fillna(0).astype(float).abs() >= threshold
            mask_put  = pd.to_numeric(pn_col, errors='coerce').fillna(0).astype(float).abs() >= threshold

            if side == 'call':
                df = df[mask_call]
            elif side == 'put':
                df = df[mask_put]
            else:
                df = df[mask_call | mask_put]

            if df.empty:
                continue

            for _, row in df.iterrows():
                # 日期：直接從 CSV 的 date 欄位取
                r_date  = str(row.get('date', ym))[:8]
                m_abbr  = str(row.get('month_abbr', '')).upper()
                m_year  = str(row.get('year', '')).replace('.0','')
                m_label = f"{m_abbr}{m_year}"
                strike  = row.get('strike', '')

                result_rows.append([
                    pcode,
                    r_date,
                    m_label,
                    row.get('call_turnover_prev', ''),
                    row.get('call_turnover', ''),
                    row.get('call_ratio', ''),
                    row.get('call_settle_price', ''),
                    row.get('call_net_change', ''),
                    row.get('call_net', ''),
                    strike,
                    row.get('put_net', ''),
                    row.get('put_net_change', ''),
                    row.get('put_settle_price', ''),
                    row.get('put_ratio', ''),
                    row.get('put_turnover', ''),
                    row.get('put_gross_prev', ''),
                ])

    # 中文欄位名（Flask jsonify 會按字母排序 key，故 dict key 順序無關緊要；
    # 前端 renderScanTable 已改用 data.columns 順序渲染，故此 dict 僅用於 label lookup）
    COL_NAMES_SCAN = {
        "code": "產品", "date": "日期", "month_label": "合約",
        "call_turnover_prev": "C上日VOL", "call_turnover": "CVOL",
        "call_ratio": "C比率", "call_settle_price": "C價",
        "call_net_change": "C淨數c", "call_net": "C淨數",
        "strike": "行使價",
        "put_net": "P淨數", "put_net_change": "P淨數c",
        "put_settle_price": "P價", "put_ratio": "P比率",
        "put_turnover": "PVOL", "put_gross_prev": "P上日VOL",
    }

    return jsonify({
        "columns": OUTPUT_COLS,
        "col_names_cn": COL_NAMES_SCAN,
        "rows": result_rows,
        "total_rows": len(result_rows),
        "filters": {
            "date_from": date_from,
            "date_to": date_to,
            "month_range": f"{month_start}{year_start}～{month_end}{year_end}" if month_start else "全部",
            "threshold": threshold,
            "side": side,
            "product_type": product_type,
            "stock_codes": stock_codes_raw,
        },
        "truncated": len(result_rows) >= MAX_ROWS,
    })


# ============================================================
# SP 現貨股價 API（支援 CSV fallback + yfinance 即時拉取）
# ============================================================
import threading
import time as _t

# yfinance 全局鎖
_yf_lock = threading.Lock()
_MAX_RETRIES = 3
_RETRY_DELAY = 2  # 秒

# In-memory cache: (code, date_str) → (result_dict, timestamp)
_sp_cache: dict[tuple[str, str], tuple[dict, float]] = {}
_CACHE_TTL = 300  # 5分鐘


# IO 指數的 Yahoo Finance 代碼對照
# 格式：'CODE' → yf_code；'CODE+MONTH+YR' → yf_code（如 HSIU26）
_IO_YF_CODES: dict[str, str] = {
    # 現貨（HTI keep 現貨）
    "HTI": "HSTECH.HK",
    # 期貨月合約
    "HSIU26": "^HSI",  "HSIZ26": "^HSI",
    "HSIU25": "^HSI",  "HSIZ25": "^HSI",
    "HHIU26": "^HSCE", "HHIZ26": "^HSCE",
    "HHIU25": "^HSCE", "HHIZ25": "^HSCE",
    # 默認：HSI/HHI → 現貨（舊代碼 fallback）
    "HSI":  "^HSI",
    "HHI":  "^HSCE",
}

# Yahoo Finance API HTTP headers（用正常瀏覽器 User-Agent 避免被拒）
_YF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def _fetch_sp_from_yf(code: str, date_str: str) -> dict | None:
    """
    用 requests 直接調用 Yahoo Finance API，取代 yfinance（yfinance 的預設
    User-Agent 容易被識別為爬蟲而被 429）。
    SO 股票：'0700' → '0700.HK'
    IO 期貨/指數：'HSIU26' → '^HSI'、'HHIU26' → '^HSCE'、'HTI' → 'HSTECH.HK'
    """
    import requests as _req

    upper_code = code.upper()
    if upper_code in _IO_YF_CODES:
        yf_code = _IO_YF_CODES[upper_code]
    else:
        yf_code = f"{int(code):04d}.HK"

    # Yahoo Finance 查詢日期區間（用 period1/period2 取代 range=10d）
    # HKT 00:00 → UTC 前一日 16:00（因為 HKT = UTC+8）
    dt = datetime.strptime(date_str, "%Y%m%d")
    utc_dt = dt - timedelta(hours=8)  # HKT → UTC
    period1 = int(utc_dt.replace(tzinfo=timezone.utc).timestamp())
    period2 = period1 + 86400  # 下一日 00:00 UTC
    interval = "1d"

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{_req.utils.quote(yf_code)}?interval={interval}&period1={period1}&period2={period2}"

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = _req.get(url, headers=_YF_HEADERS, timeout=10)
            if resp.status_code == 429:
                if attempt < _MAX_RETRIES:
                    wait = _RETRY_DELAY * (2 ** (attempt - 1))
                    print(f"[SP API] {yf_code} rate limited (429)，{wait}s 後重試")
                    _t.sleep(wait)
                    continue
                return None
            if resp.status_code != 200:
                print(f"[SP API] {yf_code} HTTP {resp.status_code}")
                return None

            data = resp.json()
            result = data.get("chart", {}).get("result")
            if not result:
                return None

            meta = result[0]["meta"]
            timestamps = result[0].get("timestamp", [])
            quote = result[0]["indicators"]["quote"][0]

            # 在 timestamp 列表中找到最接近指定日期的記錄
            target_ts = dt.timestamp()
            closest_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - target_ts))

            row = {k: quote[k][closest_idx] for k in ["open", "high", "low", "close", "volume"]}
            adj_close = None
            if "adjclose" in result[0]["indicators"]:
                adj_close = result[0]["indicators"]["adjclose"][0]["adjclose"][closest_idx]

            return {
                "code": code,
                "date": date_str,
                "date_display": parse_date_display(date_str),
                "open": float(row["open"]) if row["open"] is not None else None,
                "high": float(row["high"]) if row["high"] is not None else None,
                "low": float(row["low"]) if row["low"] is not None else None,
                "close": float(row["close"]) if row["close"] is not None else None,
                "volume": int(row["volume"]) if row["volume"] is not None else None,
                "adj_close": float(adj_close) if adj_close is not None else None,
            }

        except Exception as e:
            print(f"[SP API] Error {yf_code} (attempt {attempt}): {e}")
            if attempt < _MAX_RETRIES:
                _t.sleep(_RETRY_DELAY * (2 ** (attempt - 1)))
                continue
            return None

    return None


# ============================================================
# IF 指數期貨 API
# ============================================================
@app.route("/api/if/products")
def api_if_products():
    """回傳 IF 指數列表"""
    return jsonify(IF_INDICES)

@app.route("/api/if/dates", methods=["GET"])
def api_if_dates():
    """
    GET /api/if/dates?code=HSI
    回應該指數的所有可用日期
    """
    code = request.args.get("code", "").strip().upper()
    if not code:
        return jsonify({"error": "缺少 code 參數"}), 400

    # 驗證 code
    valid_codes = [idx["code"] for idx in IF_INDICES]
    if code not in valid_codes:
        return jsonify({"error": f"無效的指數代碼：{code}"}), 400

    index_path = IF_ROOT / code
    if not index_path.exists():
        return jsonify({"error": f"找不到目錄：{index_path}"}), 404

    dates = []
    for f in index_path.glob("*.csv"):
        m = re.search(r'(\d{8})\.csv$', f.name)
        if m:
            dates.append(m.group(1))

    dates.sort(reverse=True)
    return jsonify({
        "code": code,
        "dates": dates,
        "display": [parse_date_display(d) for d in dates],
    })

@app.route("/api/if/data", methods=["GET"])
def api_if_data():
    """
    GET /api/if/data?code=HSI&date_from=20260101&date_to=20260430&months=JAN&year=26
    回傳 IF 期貨數據（IF CSV，預設當月合約）
    months 省略時預設為當月月份，year 省略時不限制年份
    """
    code = request.args.get("code", "").strip().upper()
    date_from = request.args.get("date_from", "").strip()
    date_to = request.args.get("date_to", "").strip()
    months_param = request.args.get("months", "").strip()  # e.g. "JAN" or "MAY"
    year_param = request.args.get("year", "").strip()       # e.g. "26"

    if not code:
        return jsonify({"error": "缺少 code 參數"}), 400

    valid_codes = [idx["code"] for idx in IF_INDICES]
    if code not in valid_codes:
        return jsonify({"error": f"無效的指數代碼：{code}"}), 400

    # 取得 YF code（如有需要）
    yf_code = next((idx["yf_code"] for idx in IF_INDICES if idx["code"] == code), None)

    # 月份 abbreviation → number
    month_abbr_to_num = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

    today = datetime.today()
    current_month_num = today.month

    selected_months = set()
    if months_param:
        # 支援數字（1-12）或英文縮寫（JAN-DEC）
        m = months_param.upper()
        if m in month_abbr_to_num:
            selected_months.add(month_abbr_to_num[m])
        elif m.isdigit() and 1 <= int(m) <= 12:
            selected_months.add(int(m))
    else:
        # 預設：當月
        selected_months.add(current_month_num)

    # 年份篩選
    selected_year = None
    if year_param:
        try:
            selected_year = int(year_param)
        except ValueError:
            pass

    # 讀取所有 IF CSV 數據
    index_path = IF_ROOT / code
    if not index_path.exists():
        return jsonify({"error": f"找不到目錄：{index_path}"}), 404

    all_rows = []
    csv_files = sorted(index_path.glob("*.csv"), key=lambda f: f.name)

    for csv_file in csv_files:
        m = re.search(r'(\d{8})\.csv$', csv_file.name)
        if not m:
            continue
        file_date = m.group(1)

        # 日期篩選
        if date_from and file_date < date_from:
            continue
        if date_to and file_date > date_to:
            continue

        try:
            df = pd.read_csv(csv_file)
            df["_file_date"] = file_date
            all_rows.append(df)
        except Exception as e:
            print(f"[IF] 讀取失敗 {csv_file}: {e}")
            continue

    if not all_rows:
        return jsonify({
            "code": code,
            "columns": [],
            "rows": [],
            "total_rows": 0,
            "date_range": f"{date_from}~{date_to}" if date_from or date_to else "全部",
        })

    # 合併所有 CSV
    combined_df = pd.concat(all_rows, ignore_index=True)

    # 月份篩選
    if selected_months:
        mask = combined_df["month_abbr"].astype(str).str.upper().map(month_abbr_to_num).isin(selected_months)
        combined_df = combined_df[mask]

    # 年份篩選
    if selected_year is not None:
        combined_df = combined_df[combined_df["year"] == float(selected_year)]

    # 建立 index：用 (month_abbr, year, series) + date 排序
    combined_df["month_num_int"] = combined_df["month_abbr"].astype(str).str.upper().map(month_abbr_to_num)
    combined_df = combined_df.sort_values(["month_abbr", "year", "series", "_file_date"])

    # 計算同合約日對日差值
    combined_df["_prev_gross"] = combined_df.groupby(["month_abbr", "year", "series"])["gross"].shift(1)
    combined_df["gross_change"] = combined_df.apply(
        lambda r: round(r["gross"] - r["_prev_gross"], 0)
        if pd.notna(r["_prev_gross"]) and pd.notna(r["gross"]) else None,
        axis=1
    )

    combined_df["_prev_net"] = combined_df.groupby(["month_abbr", "year", "series"])["net"].shift(1)
    combined_df["net_change"] = combined_df.apply(
        lambda r: round(r["net"] - r["_prev_net"], 0)
        if pd.notna(r["_prev_net"]) and pd.notna(r["net"]) else None,
        axis=1
    )

    combined_df["_prev_settle"] = combined_df.groupby(["month_abbr", "year", "series"])["settle_price"].shift(1)
    combined_df["rise_fall"] = combined_df.apply(
        lambda r: round(r["settle_price"] - r["_prev_settle"], 2)
        if pd.notna(r["_prev_settle"]) and pd.notna(r["settle_price"]) else None,
        axis=1
    )

    # 構建輸出
    result_rows = []
    for _, row in combined_df.iterrows():
        file_date = row["_file_date"]
        month_abbr = str(row.get("month_abbr", "")).strip().upper()
        year_val = int(float(row["year"])) if pd.notna(row.get("year")) else 0
        month_label = f"{month_abbr}{year_val}"

        # 成交比例 (turnover / deals)
        deals_ratio = None
        d_val = row.get("deals")
        t_val = row.get("turnover")
        try:
            d_float = float(d_val) if d_val is not None else 0
            t_float = float(t_val) if t_val is not None else 0
            if d_float != 0:
                deals_ratio = round(t_float / d_float, 2)
        except (ValueError, TypeError, ZeroDivisionError):
            deals_ratio = None

        result_rows.append({
            "date": file_date,
            "date_display": parse_date_display(file_date),
            "month_label": month_label,
            "gross": row.get("gross"),
            "gross_change": row.get("gross_change"),
            "net": row.get("net"),
            "net_change": row.get("net_change"),
            "rise_fall": row.get("rise_fall"),
            "settle_price": row.get("settle_price"),
            "turnover": row.get("turnover"),
            "deals_ratio": deals_ratio,
            "deals": row.get("deals"),
        })

    # 按日期倒序
    result_rows.sort(key=lambda x: x["date"], reverse=True)

    columns = ["date", "date_display", "month_label", "gross", "gross_change", "net", "net_change", "rise_fall", "settle_price", "turnover", "deals_ratio", "deals"]

    # 過濾 NaN → None（JSON 必須用 null）
    def _clean(v):
        if isinstance(v, float) and v != v:  # NaN check: NaN != NaN
            return None
        return v

    return jsonify({
        "code": code,
        "yf_code": yf_code,
        "date_range": f"{parse_date_display(date_from)} ~ {parse_date_display(date_to)}" if date_from or date_to else "全部",
        "columns": columns,
        "rows": [[_clean(r.get(c)) for c in columns] for r in result_rows],
        "total_rows": len(result_rows),
    })


@app.route("/api/sp", methods=["GET"])
def api_sp():
    """
    GET /api/sp?code=00700&date=20260504  (SO 股票現貨)
    GET /api/sp?code=HSI&date=20260505    (IO 指數現貨)
    全部直接用 yfinance 即時拉取（附 5min cache）
    """
    code = request.args.get("code", "").strip()
    date_str = request.args.get("date", "").strip()

    if not code:
        return jsonify({"error": "缺少 code 參數"}), 400
    if not date_str:
        return jsonify({"error": "缺少 date 參數"}), 400

    # 1. 檢查 in-memory cache（5分鐘 TTL）
    cache_key = (code, date_str)
    now = _t.time()
    if cache_key in _sp_cache:
        result, ts = _sp_cache[cache_key]
        if now - ts < _CACHE_TTL:
            return jsonify(result)

    # 2. 用 yfinance 即時拉取
    result = _fetch_sp_from_yf(code, date_str)
    if result:
        _sp_cache[cache_key] = (result, now)
        return jsonify(result)

    return jsonify({"error": f"找不到股價數據（{date_str}）"}), 404


# ============================================================
# 聚合視圖 API（Tab 4 — IO 指數按 strike 聚合）
# ============================================================
IO_AGG_ROOT = Path("/root/GitHub/SData/HKEX/IO_AGG")


@app.route("/api/agg/dates", methods=["GET"])
def api_agg_dates():
    """
    GET ?index=HSI
    回傳該指數聚合視圖嘅可用日期（從最新月度檔抽取）
    """
    idx = request.args.get("index", "").strip().upper()
    if idx not in ["HSI", "HTI", "HHI"]:
        return jsonify({"error": f"不支援指數：{idx}"}), 400

    agg_dir = IO_AGG_ROOT / idx
    if not agg_dir.exists():
        return jsonify({"error": f"找不到聚合視圖目錄：{agg_dir}"}), 404

    # 搵最新嘅月度檔
    files = sorted(agg_dir.glob("*_AGG.csv"), reverse=True)
    if not files:
        return jsonify({"error": "冇聚合視圖檔案"}), 404

    # 用最新月度檔抽日期
    try:
        df = pd.read_csv(files[0], usecols=["date"])
        dates = sorted(df["date"].astype(str).str.zfill(8).unique(), reverse=True)
    except Exception as e:
        return jsonify({"error": f"讀取失敗：{e}"}), 500

    return jsonify({
        "index": idx,
        "dates": dates,
        "file": files[0].name,
    })


@app.route("/api/agg/data", methods=["GET"])
def api_agg_data():
    """
    GET ?index=HSI&date=20260601
    回傳該指數某日嘅聚合視圖（按 strike 聚合本月起所有合約）
    """
    idx     = request.args.get("index", "").strip().upper()
    date_str = request.args.get("date", "").strip()

    if idx not in ["HSI", "HTI", "HHI"]:
        return jsonify({"error": f"不支援指數：{idx}"}), 400
    if not date_str:
        return jsonify({"error": "需要 date 參數"}), 400

    # 標準化 date (e.g. 20260601)
    date_str = str(date_str).split(".")[0].zfill(8)

    # 判斷月份 (20260601 → 202606)
    ym = date_str[:6]

    csv_path = IO_AGG_ROOT / idx / f"{idx}_{ym}_AGG.csv"
    if not csv_path.exists():
        return jsonify({"error": f"找不到聚合檔：{csv_path.name}"}), 404

    try:
        df = pd.read_csv(csv_path, low_memory=False)
        # 過濾為該日期
        df["date"] = df["date"].astype(str).str.zfill(8)
        df = df[df["date"] == date_str]

        if df.empty:
            return jsonify({"error": f"{date_str} 冇資料"}), 404

        # strike NaN 過濾
        df = df.dropna(subset=["strike"])
        df = df[df["strike"] != ""]

        df = df.fillna("")

        # 欄位順序：與 OI 期權一致（Strike 放中間）
        # OI 順序：series, month_num, month_abbr, year, call_*, strike, put_*
        # 聚合視圖冇 month_abbr/year（已聚合），但保持 call_* 全部 → strike → put_* → contract_label
        # 排除 call/put_settle_price 同 call/put_price_change（唔同合約唔可比）
        OUTPUT_COLS = [
            "series",
            "call_ratio", "call_deals",
            "call_turnover_change", "call_turnover_prev", "call_turnover",
            "call_net_change", "call_net",
            "call_gross_change", "call_gross_prev", "call_gross",
            "strike",
            "put_gross", "put_gross_prev", "put_gross_change",
            "put_net", "put_net_change",
            "put_turnover", "put_turnover_prev", "put_turnover_change",
            "put_deals", "put_ratio",
            "contract_label",
        ]
        # 只保留實際存在嘅欄位
        out_cols = [c for c in OUTPUT_COLS if c in df.columns]

        return jsonify({
            "columns": out_cols,
            "rows": df[out_cols].values.tolist(),
            "code": idx,
            "date": date_str,
            "date_display": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
            "total_rows": len(df),
            "col_names_cn": COLUMN_NAMES_CN,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 啟動
# ============================================================
if __name__ == "__main__":
    import numpy as np
    print("=" * 60)
    print("  HKEX Option Viewer（SO + IO）")
    print(f"  SO Root: {SO_ROOT}")
    print(f"  IO Root: {IO_ROOT}")
    print(f"  URL:     http://0.0.0.0:80")
    print("=" * 60)
    app.run(host="0.0.0.0", port=80, debug=False)
