import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  

IOPATH = "../Sdata/HKEX/IO/"

def download_file(odate):

    url = f"https://www.hkex.com.hk/eng/stat/dmstat/oi/DTOP_F_{odate}.zip"
    local_filename = f"{IOPATH}DATA/{odate}.zip"
    
    try:
        with cc.requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            
            os.makedirs(os.path.dirname(local_filename), exist_ok=True)
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"下載成功: {local_filename}")
        return True
    except Exception as e:
        print(f"下載失敗 {local_filename}: {e}")
        return False    

def extract_data(odate):
        
    raw_filename = f"{odate}_1_dtop_f_hkcc_opt_dtl_all.raw"        

    zip_filename = f"{IOPATH}DATA/{odate}.zip"

    extract_dir = cc.Path.cwd().parent / "Sdata/HKEX/IO/TEMP"
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
        'call_price_change', 'call_settle_price', 'call_ratio', 'call_deals', 'call_turnover',
        'call_net_change', 'call_net', 'call_gross',
        'strike',
        'put_gross', 'put_net', 'put_net_change', 'put_turnover', 'put_deals','put_ratio',
        'put_settle_price', 'put_price_change'
    ]

    df_op_selected = df[desired_order]

    oplist = ['HSI','MHI','HTI','HHI','MCH']

    for op in oplist:

        op_dir = cc.Path.cwd().parent / f"Sdata/HKEX/IO/{op}"
        op_dir.mkdir(exist_ok=True)

        tempdf = df_op_selected.loc[df_op_selected['series'] == op]
        tempdf.to_csv(f"{IOPATH}\{op}\{op}_{odate}.csv", index=False)
    

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

    today = cc.datetime.today().strftime("%Y%m%d")

    sdate = today #"20260312"
    edate = today #"20260312"

    start = cc.t.perf_counter()


    ProcessDownlaod(sdate, edate)
    ProcessExtract(sdate, edate)

    finish = cc.t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')    
    

    



