import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc
from datetime import datetime
from zoneinfo import ZoneInfo
HK_TZ = ZoneInfo('Asia/Hong_Kong')


SOPATH = cc.Path(__file__).resolve().parent.parent.parent / "SData" / "HKEX" / "SO"

def download_file(odate):

    url = f"https://www.hkex.com.hk/eng/stat/dmstat/oi/DTOP_O_{odate}.zip"
    local_filename = SOPATH / "DATA" / f"{odate}.zip"

    try:
        with cc.requests.get(url, stream=True, timeout=10) as r:
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
    
    raw_filename = f"{odate}_1_dtop_o_seoch_opt_dtl_all.raw"

    zip_filename = SOPATH / "DATA" / f"{odate}.zip"

    extract_dir = SOPATH / "TEMP"
    extract_dir.mkdir(exist_ok=True)

    try:
        with cc.zipfile.ZipFile(zip_filename, 'r') as zip_ref:            
            if raw_filename not in zip_ref.namelist():
                raise FileNotFoundError(f"ZIP中未找到 {raw_filename}")
            
            zip_ref.extract(raw_filename, path=extract_dir)
            raw_file_path = extract_dir / raw_filename
            
    except Exception as e:
        print(f"解壓失敗: {e}")
        return

    df = cc.pd.read_csv(raw_file_path, encoding='utf-8', skiprows=1)

    df.columns = [
    # 合約基本資訊 (索引0-6)
    'record_type',          # 記錄類型（通常為"01"）
    'product_code',         # 產品代碼（如"HSI"）
    'product_desc',         # 產品描述（如"HANG SENG FUTURES & OPTIONS"）
    'series',               # 合約系列（如"HSI"）
    'month_num',            # 合約月份數值（如"30"）
    'month_abbr',           # 月份縮寫（如"MAR"）
    'year',                 # 年份（如"26"表示2026）
    'strike',               # 行使價

    # 認購期權 (CALL) 資料 (索引8-15)
    'call_gross',           # 未平倉合約總數（CALL）
    'call_net',             # 未平倉合約淨數（CALL）
    'call_net_change',      # 淨數變動（CALL）
    'call_turnover',        # 成交量（CALL）
    'call_deals',           # 成交宗數（CALL）
    'call_settle_price',    # 結算價（CALL）
    'call_price_change',    # 結算價變動（CALL）

    # 認沽期權 (PUT) 資料 (索引16-23)
    'put_gross',            # 未平倉合約總數（PUT）
    'put_net',              # 未平倉合約淨數（PUT）
    'put_net_change',       # 淨數變動（PUT）
    'put_turnover',         # 成交量（PUT）
    'put_deals',            # 成交宗數（PUT）
    'put_settle_price',     # 結算價（PUT）
    'put_price_change'      # 結算價變動（PUT）
    ]

    df['call_ratio'] = round(df['call_turnover'].astype(float) / df['call_deals'].astype(float), 2)
    df['put_ratio'] = round(df['put_turnover'].astype(float) / df['put_deals'].astype(float), 2)

    desired_order = [
        'series', 'month_num', 'month_abbr', 'year',
        'call_settle_price', 'call_price_change', 'call_ratio',
        'call_deals', 'call_turnover',
        'call_net_change', 'call_net', 'call_gross',
        'strike',
        'put_gross', 'put_net', 'put_net_change',
        'put_turnover', 'put_deals', 'put_ratio',
        'put_settle_price', 'put_price_change',
    ]

    df_op_selected = df[desired_order]

    # ── 新增：讀上個工作日數據，計算 change ──
    pdate = cc.getLastWorkday(cc.datetime.strptime(odate, "%Y%m%d").date()).strftime("%Y%m%d")
    prev_cols = ['call_turnover_prev', 'call_gross_prev', 'put_gross_prev', 'put_turnover_prev']
    change_cols = ['call_turnover_change', 'call_gross_change', 'put_gross_change', 'put_turnover_change']

    # merge key：合約唯一識別
    merge_key = ['series', 'month_num', 'month_abbr', 'year', 'strike']

    solistdf = cc.pd.read_csv(SOPATH / "solist.csv", dtype=str)

    for _, row in solistdf.iterrows():

        sno = row["SNO"]
        hkats = row["HKATS"]

        fname = f"{sno.replace('.HK','')}_{hkats}"
        op_dir = SOPATH / fname
        op_dir.mkdir(exist_ok=True)

        tempdf = df_op_selected.loc[df_op_selected['series'] == hkats].copy()

        # ── 讀上個工作日數據前，先將 merge key 轉為統一字串，避免類型衝突 ──
        for col in merge_key:
            tempdf[col] = tempdf[col].astype(str)

        # 嘗試讀上個工作日的 CSV
        prev_file = op_dir / f"{hkats}_{pdate}.csv"
        if prev_file.exists():
            prev_df = cc.pd.read_csv(prev_file, usecols=['series', 'month_num', 'month_abbr', 'year', 'strike',
                                                          'call_turnover', 'call_gross', 'put_gross', 'put_turnover'])
            for col in merge_key:
                prev_df[col] = prev_df[col].astype(str)
            prev_df = prev_df.rename(columns={
                'call_turnover': 'call_turnover_prev',
                'call_gross':    'call_gross_prev',
                'put_gross':     'put_gross_prev',
                'put_turnover':  'put_turnover_prev'
            })
            tempdf = tempdf.merge(prev_df, on=merge_key, how='left')
            # 計算 change
            for cur, prev, chg in zip(
                    ['call_turnover', 'call_gross', 'put_gross', 'put_turnover'],
                    ['call_turnover_prev', 'call_gross_prev', 'put_gross_prev', 'put_turnover_prev'],
                    ['call_turnover_change', 'call_gross_change', 'put_gross_change', 'put_turnover_change']):
                tempdf[chg] = (tempdf[cur] - tempdf[prev]).round(2)
        else:
            # 上個工作日資料不存在，填 NaN
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
    

    # 可選：清理暫存檔案
    os.remove(raw_file_path)
    # os.rmdir(extract_dir)

def ProcessDownlaod(sdate,edate):

    start_date = cc.datetime.strptime(sdate, "%Y%m%d")
    end_date = cc.datetime.strptime(edate, "%Y%m%d")

    alldates = cc.pd.date_range(start_date, end_date, freq='D').strftime("%Y%m%d").tolist()

    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:      
        list(cc.tqdm(executor.map(download_file,alldates,chunksize=1),total=len(alldates)))


def ProcessExtract(sdate,edate):

    start_date = cc.datetime.strptime(sdate, "%Y%m%d")
    end_date = cc.datetime.strptime(edate, "%Y%m%d")

    alldates = cc.pd.date_range(start_date, end_date, freq='D').strftime("%Y%m%d").tolist()

    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:      
        list(cc.tqdm(executor.map(extract_data,alldates,chunksize=1),total=len(alldates)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HKEX Stock Options 下載與解析")
    parser.add_argument("--sdate", type=str, help="起始日期 (YYYYMMDD)，預設為前一工作日")
    parser.add_argument("--edate", type=str, help="結束日期 (YYYYMMDD)，預設為前一工作日")
    args = parser.parse_args()

    if args.sdate and args.edate:
        sdate = args.sdate
        edate = args.edate
    else:
        # 自動判斷：收盤前用昨日，收盤後用今日
        now = datetime.now(HK_TZ)
        if now.hour < 20:
            last_wd = cc.getLastWorkday(cc.date.today())
            sdate = last_wd.strftime("%Y%m%d")
            edate = last_wd.strftime("%Y%m%d")
        else:
            sdate = cc.date.today().strftime("%Y%m%d")
            edate = cc.date.today().strftime("%Y%m%d")

    ProcessDownlaod(sdate, edate)
    ProcessExtract(sdate, edate)    
    

    



