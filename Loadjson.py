import requests
from bs4 import BeautifulSoup
import json
import re

# 發送請求
url = "http://www.aastocks.com/tc/stocks/market/industry/sector-industry-details.aspx?industrysymbol=702020&t=1&s=0&o=1&hk=0"
response = requests.get(url)

# 使用BeautifulSoup解析網頁內容
soup = BeautifulSoup(response.text,features="html.parser")

pattern = re.compile(r"var tsData *")
json_element = soup.find("script",text=pattern).string.replace("var tsData = ","|")
json_element = json_element.replace("}];","}]|")

json_data = json.dumps(json_element,ensure_ascii=False)
json_data = json_data.split("|")[1]

json_data = json_data.replace("\\","").replace("\'","").replace("</div>","|")
json_data = json_data.replace("d0:","\"d0\":").replace("d1:","\"d1\":").replace("d2:","\"d2\":").replace("d3:","\"d3\":").replace("d4:","\"d4\":").replace("d5:","\"d5\":")
json_data = json_data.replace("d6:","\"d6\":").replace("d7:","\"d7\":").replace("d8:","\"d8\":").replace("d9:","\"d9\":").replace("d10:","\"d10\":")

soup = BeautifulSoup(json_data,features="html.parser")

json_data = soup.get_text()
json_data = json.loads(json_data)

for data in json_data:
    print(data['d0'].replace("||","|").replace("||","|"))
