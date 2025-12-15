import pandas as pd
import requests
import time as t

def getStockList():    
    try:
        url = "https://www.nyse.com/api/quotes/filter"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Content-Type': 'application/json'
        }
        
        # NYSE 需要 POST 请求
        payload = {
            "instrumentType": "EQUITY",
            "pageNumber": 1,
            "sortColumn": "NORMALIZED_TICKER",
            "sortOrder": "ASC",
            "maxResultsPerPage": 10000,
            "filterToken": ""
        }
        
        stocklist = pd.DataFrame()
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()        

        df = pd.DataFrame(data)        
        mask = df['normalizedTicker'].str.contains('p', case=True, na=False)
        filterdf = df[~mask]

        stocklist["SNO"] = filterdf["normalizedTicker"]
        stocklist["NAME"] = filterdf["instrumentName"]
        stocklist["EXCHANGE"] = filterdf["url"].str[27:31]

        XNYS = stocklist.query('EXCHANGE == "XNYS"')
        XNYS = XNYS.sort_values(by="SNO")

        XASE = stocklist.query('EXCHANGE == "XASE"')
        XASE = XASE.sort_values(by="SNO")

        XNGS = stocklist.query('EXCHANGE == "XNGS"')
        XNGS = XNGS.sort_values(by="SNO")

        XNMS = stocklist.query('EXCHANGE == "XNMS"')
        XNMS = XNMS.sort_values(by="SNO")

        XNCM = stocklist.query('EXCHANGE == "XNCM"')
        XNCM = XNCM.sort_values(by="SNO")


        XNYS.to_csv("Data/XNYS.csv", index=False)
        XASE.to_csv("Data/XASE.csv", index=False)
        XNGS.to_csv("Data/XNGS.csv", index=False)
        XNMS.to_csv("Data/XNMS.csv", index=False)
        XNCM.to_csv("Data/XNCM.csv", index=False)
        
    except Exception as e:
        print(f"get NYSE List Fail: {e}")        
    

if __name__ == "__main__":
    start = t.perf_counter()

    getStockList()
    
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')