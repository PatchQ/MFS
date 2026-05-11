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
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ============================================================
# 路徑設定
# ============================================================
SO_ROOT = Path("/root/GitHub/SData/HKEX/SO")
IO_ROOT = Path("/root/GitHub/SData/HKEX/IO")
SP_ROOT = Path("/root/GitHub/SData/HKEX/SP")

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
                        "type": "SO",
                        "label": f"{code} {name}"
                    })

    # IO（指數期權）— HSI, MHI, HTI, HHI, MCH
    IO_INDICES = ["HSI", "MHI", "HTI", "HHI", "MCH"]
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
    "MHI":  "^HSI",
    "MCH":  "^HSCE",
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

    # Yahoo Finance 查詢日期區間（取指定日期的數據）
    dt = datetime.strptime(date_str, "%Y%m%d")
    # range=10d 確保能拿到指定的日期數據
    range_str = "10d"
    interval = "1d"

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{_req.utils.quote(yf_code)}?interval={interval}&range={range_str}"

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
