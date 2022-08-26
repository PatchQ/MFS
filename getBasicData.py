
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


    if table:
        table_rows = table.find_all('tr')

        res_td = []

        for tr in table_rows:
            td = tr.find_all('td')
            row_td = [val.text.strip() for val in td if val.text.strip()]

            if row_td:
                res_td.append(row_td)

        print(res_td)

getBData("02382")