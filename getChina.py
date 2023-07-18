import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import opencc

# 發送請求
url = "https://www.yidaiyilu.gov.cn/p/77298.html"
response = requests.get(url)

# 使用BeautifulSoup解析網頁內容
soup = BeautifulSoup(response.text,features="html.parser")

table =  soup.find('table')
table_rows = table.find_all('tr')
res_td = []
dropvalues = ['洲別','非洲','亞洲','歐洲','大洋洲','南美洲','北美洲']

for tr in table_rows:
    td = tr.find_all('td')
    row_td = [val.text.strip() for val in td if val.text.strip()]
    
    if row_td:
        res_td.append(opencc.OpenCC('s2hk').convert(row_td[0]))

df = pd.DataFrame(res_td,columns=['國家名'])
df = df[df.國家名.isin(dropvalues) == False]
df = df.reset_index()
df = df.drop(columns=['index'])

print(df)

df.to_excel("Data/China.xlsx", index=False)



    
