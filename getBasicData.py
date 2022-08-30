import pandas as pd
import numpy as np
import requests
import openpyxl
import os
from bs4 import BeautifulSoup
from datetime import date, timedelta

def getBData(sno):
    url = "http://www.aastocks.com/tc/stocks/analysis/company-fundamental/earnings-summary?symbol={}".format(sno)

    sess = requests.session()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
        "Referer": "http://www.aastocks.com/tc/stocks/analysis/company-fundamental/"
    }

    sess = requests.session()
    req = sess.get(url, headers=headers)
    soup = BeautifulSoup(req.text,features="html.parser")

    table =  soup.find('table',attrs={'id':'cnhk-list'})
    table_rows = table.find_all('tr')

    td = table_rows[0].find_all('td')
    headerlist = [val.text.strip() for val in td if val.text.strip()]

    if table:
        table_rows = table.find_all('tr')

        res_td = []

        for tr in table_rows[1:]:
            td = tr.find_all('td')
            row_td = [val.text.strip() for val in td if val.text.strip()]

            if row_td:
                res_td.append(row_td)

        
        df_td = pd.DataFrame(res_td, columns=headerlist[:-1])
        df_td = df_td.loc[:,~df_td.T.duplicated(keep='first')]

        df_td["sno"]=sno
        df_td.drop(9,inplace=True)
        df_td.insert(0,"sno",df_td.pop("sno"))

    return df_td


stocklist = pd.read_excel("Data/outputlist.xlsx")
snolist = stocklist["股票編號"]

resultlist = pd.DataFrame()

#tempdf = getBData("00462")
#print(tempdf)

# for sno in snolist:
#     print(sno.replace(".HK",""))
#     tempdf = getBData(sno.replace(".HK",""))
#     resultlist = pd.concat([resultlist, tempdf], ignore_index=True)

# resultlist.to_excel("BDlist.xlsx", index=False)


bdlist = pd.read_excel("Data/BDlist.xlsx",dtype=str)

snolist = bdlist.loc[bdlist["sno"]=="00001"]

snolist = snolist.dropna(axis=1)

templist = snolist.iloc[1].tail(3)



print(templist)


