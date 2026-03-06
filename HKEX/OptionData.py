import os

import requests
import zipfile
import csv
import pandas as pd
from pathlib import Path

HKEXPATH = "../SData/HKEX/"

def download_file(url):
    
    try:
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            with open(f"{HKEXPATH}temp.zip", 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"下載成功: {HKEXPATH}temp.zip")
        return True
    except Exception as e:
        print(f"下載失敗: {e}")
        return False

def extract_hsi_data(raw_filename, output_csv):

    extract_dir = Path.cwd() / "extracted"
    extract_dir.mkdir(exist_ok=True)

    try:
        with zipfile.ZipFile(f"{HKEXPATH}temp.zip", 'r') as zip_ref:            
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
                if len(row) > 1 and row[1] == 'HSI':
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

    df.to_csv(output_csv, index=False)
    print(f"CSV已保存: {output_csv}")

    # 可選：清理暫存檔案
    # os.remove(raw_file_path)
    # os.rmdir(extract_dir)

if __name__ == "__main__":    
    # 配置參數
    odate = "20260305"

    zip_url = f"https://www.hkex.com.hk/eng/stat/dmstat/oi/DTOP_F_{odate}.zip"
    target_raw = f"{odate}_1_dtop_f_hkcc_opt_dtl_all.raw"
    output_csv = f"{HKEXPATH}HSI_options_{odate}.csv"

    # 步驟1：下載ZIP文件
    if not download_file(zip_url):
        exit(1)

    # 步驟2-4：解壓、提取HSI、保存CSV
    extract_hsi_data(target_raw, output_csv)

    # 可選：刪除下載的ZIP檔以節省空間
    # os.remove(local_zip)

