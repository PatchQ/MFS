from email import header
import pandas as pd
import numpy as np
import requests
import openpyxl
from bs4 import BeautifulSoup
from tqdm import tqdm

def getStockNo(tab,indno):
    url = "http://www.aastocks.com/tc/stocks/market/industry/sector-industry-details.aspx?t={}".format(tab)+"&industrysymbol={}".format(indno)

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

    df_td = pd.DataFrame(res_td, columns=headerlist)
   
    nolist =  [("|".join(val.split("   "))).split("|")[1].split()[0] for val in df_td["名稱/  \r代號"] if val.strip()]
    df_td.insert(0,"股票編號",nolist)


    if (tab==1):
        namelist = [("|".join(val.split("   "))).split("|")[0].split()[0] for val in df_td["名稱/  \r代號"] if val.strip()]
        indnolist = [indno for val in df_td["名稱/  \r代號"]]
        df_td.insert(1,"股票名稱",namelist)
        df_td.insert(2,"行業編號",indnolist)
    else:
        df_td = df_td.drop(columns=["現價#"])

    df_td = df_td.drop(columns=["名稱/  \r代號"])
    #df_td = df_td.set_axis(headerlist[1:-1], axis=1, inplace=False)

    return df_td


indlist = pd.read_excel("Data/indlist.xlsx",dtype=str)
indnolist = indlist["行業編號"]

stocklist = pd.DataFrame()

for val in tqdm(indnolist):
    
    print(val)

    df1 = getStockNo(1,val)
    df4 = getStockNo(4,val)
    df6 = getStockNo(6,val)

    df = df1.join(df4,rsuffix='_df4').join(df6,rsuffix='_df6')

    stocklist = pd.concat([stocklist, df], ignore_index=True)


stocklist.to_excel("Data/stocklist.xlsx",index=False)



    
   
