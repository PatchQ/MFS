import requests
from bs4 import BeautifulSoup

def download_excel_with_bs4(symbol="00098"):
    url = f"http://www.aastocks.com/tc/stocks/analysis/peer.aspx?symbol={symbol}&t=4&hk=0"
    
    # --- 步驟 1：優化反爬蟲設定 ---
    # 使用 Session 可以在後續請求中自動保持 Cookies，這對許多防爬蟲網站非常重要
    session = requests.Session()
    
    # 設定逼真的 Headers，讓我們的程式碼看起來像是一個真實的瀏覽器
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "http://www.aastocks.com/", # 告訴伺服器我們是從網站內部連過去的
        "Connection": "keep-alive"
    }
    
    try:
        print(f"正在訪問頁面並獲取驗證金鑰: {url}")
        
        # --- 步驟 2：發送 GET 請求獲取網頁內容 ---
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status() # 如果發生 404 或 500 錯誤，會在此中斷
        
        # --- 步驟 3：使用 BeautifulSoup 解析網頁，提取 ASP.NET 隱藏欄位 ---
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 尋找 ASP.NET 必備的狀態參數 (這是繞過防護的關鍵)
        viewstate = soup.find("input", {"id": "__VIEWSTATE"})
        viewstate_generator = soup.find("input", {"id": "__VIEWSTATEGENERATOR"})
        event_validation = soup.find("input", {"id": "__EVENTVALIDATION"})
        
        # --- 步驟 4：準備觸發下載的 POST 資料 ---
        # ⚠️ 注意：這裡的 payload 模擬了點擊 ExportExcel 的動作
        payload = {
            # __EVENTTARGET 代表觸發事件的元件 ID。
            # 你可能需要按 F12 打開開發者工具的 Network 頁籤，點擊一次匯出按鈕，觀察送出的實際數值
            "__EVENTTARGET": "ctl00$cpDefault$btnExport",  # 這是一個常見的範例名稱，請往下看我的教學來確認
            "__EVENTARGUMENT": "",
            "__VIEWSTATE": viewstate["value"] if viewstate else "",
            "__VIEWSTATEGENERATOR": viewstate_generator["value"] if viewstate_generator else "",
            "__EVENTVALIDATION": event_validation["value"] if event_validation else "",
        }
        
        print("正在模擬執行 ExportExcel，送出下載請求...")
        
        # --- 步驟 5：發送 POST 請求下載 Excel 檔案 ---
        excel_response = session.post(url, headers=headers, data=payload, timeout=10)
        
        # 檢查伺服器回傳的是否為檔案 (避免下載到失敗的錯誤網頁)
        content_type = excel_response.headers.get("Content-Type", "")
        if "html" not in content_type:
            filename = f"Peer_{symbol}.xls"
            with open(filename, "wb") as f:
                f.write(excel_response.content)
            print(f"✅ 成功下載 Excel 檔案，已儲存為：{filename}")
        else:
            print("❌ 下載失敗，回傳的仍是網頁。請檢查 payload 中的 __EVENTTARGET 參數是否正確。")
            
    except Exception as e:
        print(f"發生錯誤: {e}")

if __name__ == "__main__":
    download_excel_with_bs4("00098")