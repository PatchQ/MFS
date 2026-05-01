import pandas as pd
import requests
from io import BytesIO

SOPATH = "../SData/HKEX/SO/"

url = "https://www.hkex.com.hk/-/media/HKEX-Market/Products/Listed-Derivatives/Market-Maker-Program/List-of-Market-Makers_Liquidity-Providers/FullListSOMM_MonthlyWebsite.xlsx"


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

try:
    # 步驟1: 下載檔案
    print("正在下載檔案...")
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()  # 檢查是否下載成功
    print(f"檔下載成功，大小: {len(response.content)} 位元組")

    # 步驟2: 將檔內容讀入pandas
    # 使用BytesIO將二進位內容轉換為檔物件
    file_data = BytesIO(response.content)

    # 讀取Excel檔，可以指定sheet_name，這裡先讀取所有sheet
    # sheet_name=None 會返回一個字典，鍵為sheet名，值為DataFrame
    all_sheets = pd.read_excel(file_data, sheet_name=None, header=None)

    # 步驟3: 假設主要資料在第一個sheet中
    # 重新讀取第一個sheet，並嘗試自動識別表頭
    file_data.seek(0)  # 重置文件指針
    main_sheet_name = list(all_sheets.keys())[0]    
    solistdf = pd.read_excel(file_data, sheet_name=main_sheet_name, header=0)  # header=0表示第一行為列名

    solistdf.drop(columns=['Number 編號'], inplace=True)
    
    solistdf.rename(columns={'HKATS Code HKATS 代號': 'HKATS', 'Stock Code 股票編號': 'SNO', 
                             'English Stock Name 股票英文名稱':'ENAME', 'Chinese Stock Name 股票中文名稱': 'CNAME'}, inplace=True)

    solistdf['SNO'] = solistdf['SNO'].astype(str).str.strip()
    solistdf['SNO'] = solistdf['SNO'].str.zfill(4) + '.HK'

    # 可選：保存為CSV檔
    solistdf.to_csv(f"{SOPATH}solist.csv", index=False, encoding='utf-8-sig')
    print("\n資料已保存為 'solist.csv'")

    second_sheet_name = list(all_sheets.keys())[1]
    mmlistdf = pd.read_excel(file_data, sheet_name=second_sheet_name, header=0)  # header=0表示第一行為列名
    mmlistdf.to_csv(f"{SOPATH}mmlist.csv", index=False, encoding='utf-8-sig')
    print("\n資料已保存為 'mmlist.csv'")


except requests.exceptions.RequestException as e:
    print(f"網路或下載錯誤: {e}")
    print("請檢查連結是否有效，或稍後重試。")
except Exception as e:
    print(f"處理文件時出錯: {e}")
    print("可能是檔案格式已變動，請檢查下載的檔。")

