import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

def getStockNo(tab,indno):
    
    url = "http://www.aastocks.com/tc/stocks/market/industry/sector-industry-details.aspx?s=&o=1&p=&t={}".format(tab)+"&industrysymbol={}".format(indno)
    sess = requests.session()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
        "Referer": "http://www.aastocks.com/tc/stocks/market/industry/sector-industry-details.aspx"
    }

    req = sess.get(url, headers=headers)
    soup = BeautifulSoup(req.text, features="html.parser")
    table =  soup.find('table',attrs={'id':'tblTS2'})
    table_rows = table.find_all('tr')

    th = table_rows[0].find_all('th')
    headerlist = [val.text.strip() for val in th if val.text.strip()]

    if (tab==1):
        headerlist = headerlist[:-1]

    if (tab==4):
        th = table_rows[1].find_all('th')
        templist = [val.text.strip() for val in th if val.text.strip()]        
        headerlist = headerlist[:2]+templist+headerlist[-1:]        

    res_td = []

    for tr in table_rows:
        td = tr.find_all('td')
        row_td = [val.text.strip() for val in td if val.text.strip()]

        if row_td:
            res_td.append(row_td)

    #df_td = pd.DataFrame(res_td, columns=headerlist)
    # 確保 res_td 裡面有資料
    if res_td:
        # 取得實際抓取到的資料欄位數
        data_col_count = len(res_td[0])
    
        # 如果抓到的資料欄位數量與 headerlist 不同，自動調整標題長度以符合資料
        if data_col_count != len(headerlist):
            print(f" [提示] 資料欄位數 ({data_col_count}) 與標題數 ({len(headerlist)}) 不符，已自動調整。")
            current_headers = headerlist[:data_col_count] # 截取符合長度的標題
        else:
            current_headers = headerlist
            
        try:
            # 使用動態調整後的標題建立 DataFrame
            df_td = pd.DataFrame(res_td, columns=current_headers)
        except Exception as e:
            print(f" [錯誤] 無法轉換為 DataFrame，略過此筆。錯誤訊息：{e}")
            df_td = pd.DataFrame() # 發生意外錯誤時，回傳空表以防程式崩潰
    else:
        # 如果根本沒有抓到資料，回傳空表
        df_td = pd.DataFrame(columns=headerlist)


    print(df_td)
   
    nolist = []
    for val in df_td["名稱/  \r代號"]:
        if val.strip():
            if "   " in val:
                nolist.append(("|".join(val.split("   "))).split("|")[1].split()[0])                
            else:
                nolist.append(val.split()[0])
  
    df_td.insert(0,"股票編號",nolist)

    if (tab==1):
        namelist = []
        for val in df_td["名稱/  \r代號"]:
            if val.strip():
                if "   " in val:
                    namelist.append(("|".join(val.split("   "))).split("|")[0].split()[0])                
                else:
                    namelist.append(val.split()[0])  

        indnolist = [indno for val in df_td["名稱/  \r代號"]]
        df_td.insert(1,"股票名稱",namelist)
        df_td.insert(2,"行業編號",indnolist)
    else:
        df_td = df_td.drop(columns=["現價#"])

    df_td = df_td.drop(columns=["名稱/  \r代號"])

    return df_td



def changeAmount(val):
    if ("億" in val):
        val = val.replace("億","").replace(",","")
        val = float(val) * 100000000
    elif("千萬" in val):
        val = val.replace("千萬","").replace(",","")
        val = float(val) * 10000000
    elif("百萬" in val):
        val = val.replace("百萬","").replace(",","")
        val = float(val) * 1000000
    else:
        val=0
    return val


def getStockListData():

    indlist = pd.read_csv("data/indlist.csv",dtype=str)
    indnolist = indlist["行業編號"][:]

    stocklist = pd.DataFrame()

    for val in tqdm(indnolist):
        
        df1 = getStockNo(1,val)
        df4 = getStockNo(4,val)    
        df6 = getStockNo(6,val)    

        df = df1.join(df4,rsuffix='_df4').join(df6,rsuffix='_df6')
        stocklist = pd.concat([stocklist, df], ignore_index=True)

    stocklist["sno"] = stocklist["股票編號"].apply(lambda s: s.lstrip("0").zfill(7))



    #filter
    stocklist["數字市值"] = stocklist["市值"].apply(lambda s: changeAmount(s)).astype(float)

    stocklistL = stocklist.query("數字市值 >= 20000000000")
    stocklistL = stocklistL.assign(type="L")
    stocklistL = stocklistL.sort_values(by="股票編號")

    stocklistM = stocklist.query("數字市值 < 20000000000").query("數字市值 >= 5000000000")
    stocklistM = stocklistM.assign(type="M")
    stocklistM = stocklistM.sort_values(by="股票編號")

    stocklistS = stocklist.query("數字市值 < 5000000000")
    stocklistS = stocklistS.assign(type="S")
    stocklistS = stocklistS.sort_values(by="股票編號")


    stocklist.to_csv("data/stocklist.csv",index=False,encoding="utf-8-sig")
    stocklistL.to_csv("data/stocklist_L.csv",index=False,encoding="utf-8-sig")
    stocklistM.to_csv("data/stocklist_M.csv",index=False,encoding="utf-8-sig")
    stocklistS.to_csv("data/stocklist_S.csv",index=False,encoding="utf-8-sig")


   
