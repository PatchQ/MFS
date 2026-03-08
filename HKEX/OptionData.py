from datetime import datetime, timedelta
import os

import requests
import zipfile
import csv
import pandas as pd
from pathlib import Path

HKEXPATH = "../SData/HKEX/SO/"
   
def download_file(url,odate):
    
    local_filename = f"{HKEXPATH}{odate}.zip"
    
    try:
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            # 确保目录存在
            os.makedirs(os.path.dirname(local_filename), exist_ok=True)
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"下載成功: {local_filename}")
        return True
    except Exception as e:
        print(f"下載失敗 {local_filename}: {e}")
        return False    

def extract_hsi_data(raw_filename, output_csv):

    extract_dir = Path.cwd() / "extracted"
    extract_dir.mkdir(exist_ok=True)

    try:
        with zipfile.ZipFile(f"{HKEXPATH}{odate}.zip", 'r') as zip_ref:            
            if raw_filename not in zip_ref.namelist():
                raise FileNotFoundError(f"ZIP中未找到 {raw_filename}")
            
            zip_ref.extract(raw_filename, path=extract_dir)
            raw_file_path = extract_dir / raw_filename
            print(f"解壓成功: {raw_file_path}")
    except Exception as e:
        print(f"解壓失敗: {e}")
        return

    # 讀取raw檔並篩選HSI記錄
    hsi_rows = []
    try:
        with open(raw_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, quotechar='"')
            for i, row in enumerate(reader):
                # 跳過第一行文件頭
                if i == 0:
                    continue
                # 遇到檔結束標記 'T' 停止
                if row[0] == 'T':
                    break
                # 檢查第二欄是否為 "HSI"
                if len(row) > 1 and row[1] == 'HSI': #and row[5]=="MAR" and row[6]=="26":
                    hsi_rows.append(row)
        print(f"篩選到 {len(hsi_rows)} 條HSI記錄")
    except Exception as e:
        print(f"讀取檔失敗: {e}")
        return

    if not hsi_rows:
        print("沒有找到HSI記錄")
        return

    # 轉換為DataFrame並保存為CSV
    df = pd.DataFrame(hsi_rows)

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
        'month_abbr', 'year',
        'call_price_change', 'call_settle_price', 'call_ratio', 'call_deals', 'call_turnover',
        'call_net_change', 'call_net', 'call_gross',
        'strike',
        'put_gross', 'put_net', 'put_net_change', 'put_turnover', 'put_deals','put_ratio',
        'put_settle_price', 'put_price_change'
    ]

    df_hsi_selected = df[desired_order]

    df_hsi_selected.to_csv(output_csv, index=False)
    print(f"CSV已保存: {output_csv}")

    # 可選：清理暫存檔案
    # os.remove(raw_file_path)
    # os.rmdir(extract_dir)


def getfile(odate):
    #url = f"https://www.hkex.com.hk/eng/stat/dmstat/oi/DTOP_F_{odate}.zip"
    url = f"https://www.hkex.com.hk/eng/stat/dmstat/oi/DTOP_O_{odate}.zip"

    #target_raw = f"{odate}_1_dtop_f_hkcc_opt_dtl_all.raw"
    target_raw = f"{odate}_1_dtop_o_seoch_opt_dtl_all.raw"
    
    output_csv = f"{HKEXPATH}HSIO{odate}.csv"

    success = download_file(url,odate)
    if success is False:
        #extract_hsi_data(target_raw, output_csv)
        #os.remove(f"{HKEXPATH}temp.zip")
    #else:
        print(f"日期 {odate} 下載失敗")

    

if __name__ == "__main__":    

    start_date = datetime(2018, 1, 1)
    end_date = datetime(2025, 12, 31)

    current_date = start_date
    while current_date <= end_date:        
        odate = current_date.strftime("%Y%m%d")    
        getfile(odate)
        
        current_date += timedelta(days=1)


