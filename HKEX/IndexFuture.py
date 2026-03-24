import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  

def extract_data(odate):
        
    raw_filename = f"{odate}_1_dtop_f_hkcc_fut_dtl_all.raw"        

    zip_filename = f"{cc.IOPATH}DATA/{odate}.zip"

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

    'record_type',          # 記錄類型（通常為"01"）
    'product_code',         # 產品代碼（如"HSI"）
    'product_desc',         # 產品描述（如"HANG SENG FUTURES & OPTIONS"）
    'series',               # 合約系列（如"HSI"）
    'month_num',            # 合約月份數值（如"30"）
    'month_abbr',           # 月份縮寫（如"MAR"）
    'year',                 # 年份（如"26"表示2026）
    
    'gross',           # 未平倉合約總數
    'net',             # 未平倉合約淨數
    'net_change',      # 淨數變動
    'turnover',        # 成交量
    'deals',           # 成交宗數
    'settle_price',    # 結算價
    'price_change',    # 結算價變動
    ]

    df['ratio'] = round(df['turnover'].astype(float) / df['deals'].astype(float), 2)
    
    desired_order = [
        'series', 'month_num', 'month_abbr', 'year',
        'gross', 'net', 'net_change', 'turnover', 'deals','ratio',
        'settle_price', 'price_change'
    ]

    df_op_selected = df[desired_order]

    oplist = ['HSI','MHI','HTI','HHI','MCH']

    for op in oplist:

        op_dir = cc.Path.cwd().parent / f"Sdata/HKEX/IO/{op}"
        op_dir.mkdir(exist_ok=True)

        tempdf = df_op_selected.loc[df_op_selected['series'] == op]
        tempdf.to_csv(f"{cc.IFPATH}\{op}\{op}_{odate}.csv", index=False)
    

    # 可選：清理暫存檔案
    os.remove(raw_file_path)
    # os.rmdir(extract_dir)

def ProcessExtract(sdate,edate):

    start_date = cc.datetime.strptime(sdate, "%Y%m%d")
    end_date = cc.datetime.strptime(edate, "%Y%m%d")

    alldates = cc.pd.date_range(start_date, end_date, freq='D').strftime("%Y%m%d").tolist()

    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:      
        list(cc.tqdm(executor.map(extract_data,alldates,chunksize=1),total=len(alldates)))


if __name__ == "__main__":    

    start = cc.t.perf_counter()

    sdate = cc.getLastWorkday(cc.date.today()).strftime("%Y%m%d")
    edate = cc.getLastWorkday(cc.date.today()).strftime("%Y%m%d")

    sdate = "20250102"
    edate = "20260323"
    
    ProcessExtract(sdate, edate)

    finish = cc.t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')    
    

    



