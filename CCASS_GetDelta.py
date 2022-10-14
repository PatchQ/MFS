
import pandas as pd
import numpy as np
import requests
import openpyxl
import os
from bs4 import BeautifulSoup
from datetime import date, timedelta
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

PATH = "../SData/CCASS/"
OUTPATH = "../SData/CCASS_Delta/"
SDATE = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
EDATE = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
DATERANGE = pd.date_range(SDATE, EDATE)

BBLIST = pd.read_excel("Data/bigbrokerlist.xlsx",dtype=str)
BBLIST = BBLIST["No"]
SLIST = list(map(lambda s: s.replace(".xlsx", ""), os.listdir(PATH)))
SLIST = SLIST[:]


def getCCASSData(sno,date):

    url = "https://www3.hkexnews.hk/sdw/search/searchsdw_c.aspx"

    headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
            "Referer": "https://www3.hkexnews.hk"
        }

    sess = requests.session()
    req = sess.get(url, headers=headers)
    soup = BeautifulSoup(req.text,features="html.parser")

    payload = {item['name']:item.get('value','') for item in soup.select("input[name]")}
    payload['__EVENTTARGET'] = 'btnSearch'
    payload['txtStockCode'] = str(sno).replace(".HK","")
    payload['txtShareholdingDate'] = date

    req = sess.post(url,data=payload,headers=headers)
    soup_obj = BeautifulSoup(req.text,"html.parser")

    table =  soup_obj.find('table',attrs={'class':'table table-scroll table-sort table-mobile-list'})

    if table:
        table_rows = table.find_all('tr')

        th = table_rows[0].find_all('th')
        headerlist = [val.text.strip() for val in th if val.text.strip()]

        res_td = []

        for tr in table_rows:
            td = tr.find_all('td')
            row_td = [val.text.strip().split(":")[1].strip("\n") for val in td if val.text.strip()]

            if row_td:
                res_td.append(row_td)

        df_td = pd.DataFrame(res_td, columns=headerlist)
        df_td["sno"]=sno
        df_td["date"]=date

        df_td.drop(["地址"],axis=1,inplace=True)

        df_td = df_td.loc[df_td["參與者編號"].isin(BBLIST)]
        df_td["持股量"] = df_td["持股量"].apply(lambda s: s.replace(",","").replace(".","")).astype(float)

        df_td.sort_index(axis=1, inplace=True)

        return df_td

def GetDelta(sno):

    #get file modified date
    filemdate = date.fromtimestamp(os.path.getmtime(PATH+sno+".xlsx"))

    #if(True):
    if(filemdate.strftime("%Y%m%d")!=date.today().strftime("%Y%m%d")):
        df = pd.read_excel(PATH+sno+".xlsx")

        for valdate in DATERANGE:
            print(valdate)
            tempdf = df.query("date=="+valdate.strftime('%Y%m%d')+"")
            if(tempdf.size==0):
                temp = getCCASSData(sno,valdate.strftime("%Y%m%d"))
                df = pd.concat([df, temp], ignore_index=True)

        df["date"] = df["date"].astype(int)
        df = df.sort_values(by='date')
        df.to_excel(OUTPATH+sno+".xlsx",index=False)


for sno in tqdm(SLIST):
    GetDelta(sno)
