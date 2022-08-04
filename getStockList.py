import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

def getStockNo(no):
    url = "http://www.aastocks.com/tc/stocks/market/industry/sector-industry-details.aspx?industrysymbol={}".format(no)

    sess = requests.session()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
        "Referer": "http://www.aastocks.com/tc/stocks/market/industry/sector-industry-details.aspx"
    }

    req = sess.get(url, headers=headers)
    soup = BeautifulSoup(req.text, features="html.parser")
    values =  soup.select("#tblTS2 .nls a.bmpLnk.cls")

    return values



def getIndustry():
    url = "http://www.aastocks.com/tc/stocks/market/industry/industry-performance.aspx"

    sess = requests.session()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
        "Referer": "http://www.aastocks.com/tc/stocks/market/industry/industry-performance.aspx"
    }

    req = sess.get(url, headers=headers)
    soup = BeautifulSoup(req.text, features="html.parser")
    values =  soup.select("#IndustyMain tr.indview_tr.nowrap a.a15.cls")

    return values


indlist = pd.DataFrame()
stocklist = pd.DataFrame()

indlist["no"] = [str.get("href").split("?industrysymbol=")[1].strip() for str in getIndustry()]
indlist["title"] = [str.text.strip() for str in getIndustry()]

indlist.to_csv("indlist.csv")


for index, row in indlist.iterrows():
    print(index)
    
    templist1 = pd.DataFrame()
    templist1["no"] = [str.text.strip() for str in getStockNo(row['no'])]
    templist1["indno"] = row.get("no")
    templist2 = stocklist

    stocklist = pd.concat([templist1,templist2],axis=0,ignore_index=True)

print(stocklist)
stocklist.to_csv("stocklist.csv")
