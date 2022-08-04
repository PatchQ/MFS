import requests
from bs4 import BeautifulSoup

def getStockPrice(num):
    url = "http://www.aastocks.com/en/stocks/quote/detail-quote.aspx?symbol={}".format(num)

    sess = requests.session()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
        "Referer": "http://www.aastocks.com/tc/stocks/quote/detail-quote.aspx"
    }

    req = sess.get(url, headers=headers)
    soup = BeautifulSoup(req.text,features="html.parser")
    return soup.select(".content #labelLast span")[0].text.strip(" \xa0")

for stockNum in ["00700","03690","02382"]:
    print(stockNum+" : "+getStockPrice(stockNum))