import sys
import os
import re

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
HK_TZ = ZoneInfo('Asia/Hong_Kong')

# 專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# SData 在 ~/GitHub/SData/（不是 PROJECT_ROOT/SData）
SDATA_ROOT = Path("/root/GitHub/SData")
SPPATH = SDATA_ROOT / "HKEX" / "SP"

# 月份代碼映射 (HKEX Stock Options uses CBOE monthly codes: M=JAN, N=FEB, O=MAR, Q=APR, E=MAY, F=JUN, G=JUL, H=AUG, I=SEP, J=OCT, K=NOV, L=DEC)
# 額外週期權代碼: C=Mar(?), R, S, T, U, X 等
MONTH_MAP = {
    'M': '01', 'N': '02', 'O': '03', 'Q': '04',
    'E': '05', 'F': '06', 'G': '07', 'H': '08',
    'I': '09', 'J': '10', 'K': '11', 'L': '12',
    # 週期權額外代碼
    'C': '03', 'R': '??', 'S': '??', 'T': '??', 'U': '09', 'X': '11'
}
MONTH_ABBR = {
    '01': 'JAN', '02': 'FEB', '03': 'MAR', '04': 'APR', '05': 'MAY',
    '06': 'JUN', '07': 'JUL', '08': 'AUG', '09': 'SEP', '10': 'OCT',
    '11': 'NOV', '12': 'DEC'
}

def parse_option_code(code):
    """
    解析 HKEX Stock Option 代碼，如 A5010.00E6, A509.75E6, RKB75.00L6
    返回 (stock_code, strike, month_num, month_abbr, year)
    
    HKEX 月份代碼: M=JAN, N=FEB, O=MAR, Q=APR, E=MAY, F=JUN, G=JUL, H=AUG, I=SEP, J=OCT, K=NOV, L=DEC
    """
    # 嘗試不同 stock 長度（由長到短，避免貪心匹配）
    # 格式: stock_code + strike + month_letter + year_digit
    # strike 可以是: 25000 (整數，5位) 或 10.00 (小數，2位小數) 或 9.75
    best_match = None
    for stock_len in range(6, 1, -1):
        if len(code) < stock_len + 5:  # stock(2) + strike(1.00) + month + year
            continue
        stock = code[:stock_len]
        rest = code[stock_len:]
        # 嘗試正則表達式匹配 strike(小數格式) + month + year
        m = re.match(r'^(\d+\.\d{2})([A-Z])(\d)$', rest)
        if m:
            strike_str, month_letter, year_digit = m.groups()
            month_num = MONTH_MAP.get(month_letter)
            if month_num and month_num != '??':
                strike = float(strike_str)
                # 跳過不合理的 strike（如 0.00 或極小的值，可能是 stock 部分被錯誤匹配）
                if strike > 0:
                    month_abbr = MONTH_ABBR.get(month_num, '???')
                    year = f'20{year_digit}'
                    # 優先選擇 strike 較大的匹配（stock 代碼較短更合理）
                    if best_match is None or strike > best_match[1]:
                        best_match = (stock, strike, month_num, month_abbr, year)
        # 嘗試整數 strike + month + year
        m2 = re.match(r'^(\d{4,6})([A-Z])(\d)$', rest)
        if m2:
            strike_str, month_letter, year_digit = m2.groups()
            month_num = MONTH_MAP.get(month_letter)
            if month_num and month_num != '??':
                strike = int(strike_str)
                if strike > 0:
                    month_abbr = MONTH_ABBR.get(month_num, '???')
                    year = f'20{year_digit}'
                    if best_match is None or strike > best_match[1]:
                        best_match = (stock, strike, month_num, month_abbr, year)
    return best_match if best_match else (None, None, None, None, None)

def download_file(odate):
    """
    下載 RP006 zip 文件
    
    odate 可以是 YYYYMMDD 或 YYMMDD 格式
    URL 使用 YYMMDD 格式 (如 260508)
    """
    # 標準化日期格式
    # URL 和本地檔案都使用 YYMMDD 格式 (如 260508)
    if len(odate) == 6:
        odate_zip = odate  # already YYMMDD
    else:
        odate_zip = odate[2:]  # YYYYMMDD -> YYMMDD for URL and local file
    
    url = f"https://www.hkex.com.hk/eng/market/rm/rm_dcrm/riskdata/srprices/RP006_{odate_zip}.zip"
    local_filename = SPPATH / "DATA" / f"RP006_{odate_zip}.zip"
    
    try:
        with cc.requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            local_filename.parent.mkdir(parents=True, exist_ok=True)
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"下載成功: {local_filename}")
        return True
    except Exception as e:
        print(f"下載失敗 {local_filename}: {e}")
        return False

def extract_data(odate):
    """
    解析 RP006 zip 中的 _final_o.raw (options) 檔案
    
    odate 可以是 YYYYMMDD 或 YYMMDD 格式
    URL 和本地檔案使用 YYMMDD 格式 (如 260508)
    ZIP 和 RAW 文件使用 YYYYMMDD 格式 (如 20260508)
    """
    # 標準化日期格式
    if len(odate) == 6:
        odate_zip = f"20{odate}"  # YYMMDD -> YYYYMMDD for zip file inside
        odate_file = f"20{odate}"  # YYMMDD -> YYYYMMDD for raw file inside
        odate_dt = f"20{odate}"  # datetime parsing uses YYYYMMDD
        odate_local = odate  # local file uses YYMMDD
    else:
        odate_zip = odate  # already YYYYMMDD
        odate_file = odate  # already YYYYMMDD
        odate_dt = odate  # datetime parsing uses YYYYMMDD
        odate_local = odate[2:]  # YYYYMMDD -> YYMMDD for local file
    
    zip_filename = SPPATH / "DATA" / f"RP006_{odate_local}.zip"
    extract_dir = SPPATH / "TEMP"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    raw_filename = f"{odate_file}_1_rp006-final_o.raw"
    
    try:
        with cc.zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            if raw_filename not in zip_ref.namelist():
                raise FileNotFoundError(f"ZIP中未找到 {raw_filename}")
            zip_ref.extract(raw_filename, path=extract_dir)
            raw_file_path = extract_dir / raw_filename
    except Exception as e:
        print(f"解壓失敗: {e}")
        return
    
    # 讀取 raw 檔案並解析
    records = []
    
    with open(raw_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 去除 \r\n
            line = line.replace('\r', '').replace('\n', '')
            if not line:
                continue
            
            # 跳過header行
            if line.startswith('"H"'):
                continue
            
            # 使用 csv module 正確解析
            import csv as csv_module
            import io
            reader = csv_module.reader(io.StringIO(line))
            parts = next(reader)
            
            if len(parts) < 10:
                continue
            
            record_type = parts[0].strip('"')
            if record_type != '01':
                continue
            
            option_code = parts[1].strip('"')
            series_type = parts[2].strip('"')
            description = parts[3].strip('"')
            stock_code = parts[4].strip('"')
            name = parts[5].strip('"')
            currency = parts[6].strip('"')
            prev_settle = parts[7].strip('"')
            settle_price = parts[8].strip('"')
            change = parts[9].strip('"')
            
            # IV 是可選的最後一個欄位
            iv = parts[10].strip('"') if len(parts) > 10 else ''
            
            # 解析 option_code
            parsed = parse_option_code(option_code)
            if parsed[0] is None:
                # 無法解析的行（如 header 類的 RKBSP）
                continue
            
            stock, strike, month_num, month_abbr, year = parsed
            
            records.append({
                'stock_code': stock,
                'name': name,
                'series_type': series_type,
                'currency': currency,
                'month_num': month_num,
                'month_abbr': month_abbr,
                'year': year,
                'strike': strike,
                'prev_settle': prev_settle,
                'settle_price': settle_price,
                'change': change,
                'iv': iv,
                'option_code': option_code
            })
    
    if not records:
        print(f"警告: {odate} 沒有找到有效記錄")
        os.remove(raw_file_path)
        return
    
    df = cc.pd.DataFrame(records)
    
    # 讀取上個工作日數據計算 change (使用 YYYYMMDD 格式)
    pdate = cc.getLastWorkday(cc.datetime.strptime(odate_dt, "%Y%m%d").date()).strftime("%Y%m%d")
    
    # 嘗試讀取上個工作日的 CSV
    prev_file = SPPATH / f"SP_{pdate}.csv"
    if prev_file.exists():
        prev_df = cc.pd.read_csv(prev_file, usecols=['option_code', 'settle_price', 'iv'])
        prev_df = prev_df.rename(columns={
            'settle_price': 'settle_price_prev',
            'iv': 'iv_prev'
        })
        df = df.merge(prev_df, on='option_code', how='left')
        df['settle_change'] = (df['settle_price'].astype(float) - df['settle_price_prev'].astype(float)).round(4)
        df['iv_change'] = (df['iv'].astype(float) - df['iv_prev'].astype(float)).round(4)
    else:
        df['settle_price_prev'] = float('nan')
        df['iv_prev'] = float('nan')
        df['settle_change'] = float('nan')
        df['iv_change'] = float('nan')
    
    # 欄位順序
    final_order = [
        'stock_code', 'name', 'series_type', 'currency',
        'month_num', 'month_abbr', 'year', 'strike',
        'prev_settle', 'settle_price', 'settle_change',
        'iv', 'iv_change',
        'change',
        'option_code'
    ]
    
    df = df.reindex(columns=final_order)
    # 輸出檔名使用 YYYYMMDD 格式
    df.to_csv(SPPATH / f"SP_{odate_zip}.csv", index=False)
    print(f"已保存 {len(df)} 條記錄到 SP_{odate_zip}.csv")
    
    # 清理暫存
    os.remove(raw_file_path)

def ProcessDownlaod(sdate, edate):
    start_date = cc.datetime.strptime(sdate, "%Y%m%d")
    end_date = cc.datetime.strptime(edate, "%Y%m%d")
    alldates = cc.pd.date_range(start_date, end_date, freq='D').strftime("%Y%m%d").tolist()
    
    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        list(cc.tqdm(executor.map(download_file, alldates, chunksize=1), total=len(alldates)))

def ProcessExtract(sdate, edate):
    start_date = cc.datetime.strptime(sdate, "%Y%m%d")
    end_date = cc.datetime.strptime(edate, "%Y%m%d")
    alldates = cc.pd.date_range(start_date, end_date, freq='D').strftime("%Y%m%d").tolist()
    
    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        list(cc.tqdm(executor.map(extract_data, alldates, chunksize=1), total=len(alldates)))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HKEX Stock Options (RP006) 下載與解析")
    parser.add_argument("--sdate", type=str, help="起始日期 (YYYYMMDD)，預設為前一工作日")
    parser.add_argument("--edate", type=str, help="結束日期 (YYYYMMDD)，預設為前一工作日")
    parser.add_argument("--download-only", action="store_true", help="僅下載不解析")
    parser.add_argument("--extract-only", action="store_true", help="僅解析不下載")
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
    
    if args.extract_only:
        ProcessExtract(sdate, edate)
    elif args.download_only:
        ProcessDownlaod(sdate, edate)
    else:
        ProcessDownlaod(sdate, edate)
        ProcessExtract(sdate, edate)
    
    finish = cc.t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
