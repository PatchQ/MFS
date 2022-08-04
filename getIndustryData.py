from typing import Sequence
import pandas as pd
import numpy as np
import requests
import openpyxl
from bs4 import BeautifulSoup



url = "http://www.aastocks.com/tc/stocks/market/industry/industry-performance.aspx?&s=1&o=1"

sess = requests.session()

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
    "Referer": "http://www.aastocks.com/tc/stocks/market/industry/industry-performance.aspx"
}

req = sess.get(url, headers=headers)
soup = BeautifulSoup(req.text, features="html.parser")
table =  soup.find('table',attrs={'class':'indview_tbl'})
table_rows = table.find_all('tr')

res = []
indno = []

for tr in table_rows:
    alist = tr.select('.colFirst a.a15.cls')
    href = [val.get("href").split("?industrysymbol=")[1].strip() for val in alist if val.text.strip()]

    td = tr.find_all('td')
    row = [tr.text.strip() for tr in td if tr.text.strip()]

    if row:
        if row[4]!="0.00" and row[5]!="0.00":
            res.append(list(map(lambda x: x.replace('▼', '').replace('▲',''), row)))
            
            if href:
                indno.append(href)


df1 = pd.DataFrame(indno,columns=["行業編號"])

df = pd.DataFrame(res[1:],columns=res[0][:6])

df.insert(0,"行業編號",df1["行業編號"])

print(df)
df.to_excel("indlist.xlsx",index=False)

