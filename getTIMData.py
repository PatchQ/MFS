import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime, timedelta

EVISALIST = ["E-visa", "e-visa"]
ETALIST = ["ETA", "eTA"]

TODAY = datetime.now().strftime("%Y%m%d")
ADATE = datetime.now().strftime("%Y-%m-%d")
DDATE = datetime.now().strftime("%Y-%m-%d")
EDATE = (datetime.now() + timedelta(days=int(365))).strftime("%Y-%m-%d")
NCODE = "MO"


MACAOPS = "MACAOSARCHINAPASSPORT"
PS = "PASSPORT"
MACAOTP = "TRAVELPERMITHKMO"
TP = "TRAVELPERMIT"

DOCTYPE = PS


def getTIMData(sno):
    url = "https://www.timaticweb2.com/integration/external?ref=cf2a4d29952f5bf6503e449037906178&country=/"+sno           

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
        "Referer": "https://www.timaticweb2.com/integration/"
    }

    with requests.Session() as s:
        
        returnval = ""
        r = s.get(url, headers=headers)

        payload = {                 
                    #"usrid": "19043", 
                    "ref": "cf2a4d29952f5bf6503e449037906178",                    
                    "pvh_destinationcountrycode":sno, 
                    "pvh_documenttype_1":DOCTYPE,
                    "pvh_residentcountrycode":NCODE,
                    "pvh_issuecountrycode_1":NCODE,                
                    "pvh_departairportcode":NCODE, 
                    "pvh_nationalitycode":NCODE,                                                
                    "pvh_arrivaldate":ADATE,
                    "pvh_departuredate":DDATE,                 
                    "pvh_expirydate_1":EDATE                                   
                }        
        
        r = s.post(url, headers=headers, data=payload)
        soup = BeautifulSoup(r.content, 'html.parser', from_encoding='utf-8')
        itemvisa =  soup.find('div',attrs={'class':'J-country-data-0'})

        returnval = ""

        if itemvisa != None:
            visatext =  itemvisa.find('div',attrs={'class':'content'})
            if visatext != None:
                returnval = visatext.text
        
        return returnval
    
def ExportTIMData(filename):
        
    countrydf = pd.read_excel(filename+".xlsx",dtype=str,keep_default_na=False)
    countrydf = countrydf[:]
    countrydf = countrydf.rename(columns={"TIM"+filename.split("_")[1]: "TIM"+TODAY})

    for i in tqdm(countrydf.index):
        if countrydf.loc[i, "國家代碼"] != "-":
            countrydf.loc[i, "old"]=countrydf.loc[i, "TIM"+TODAY].replace("\n"," ").strip()
            countrydf.loc[i, "new"]=getTIMData(countrydf.loc[i, "國家代碼"]).replace("\n"," ").strip()
            countrydf.loc[i, "TIM"+TODAY]=countrydf.loc[i, "new"]
            countrydf.loc[i, "diff"]=countrydf.loc[i, "old"].startswith(countrydf.loc[i, "new"])
            countrydf.loc[i, "e-visa"]=any(elem in countrydf.loc[i, "new"] for elem in EVISALIST)
            countrydf.loc[i, "ETA"]=any(elem in countrydf.loc[i, "new"] for elem in ETALIST)
            
    countrydf.to_excel("Result_TIM_"+TODAY+".xlsx",index=False)


if __name__ == '__main__':

    filename = "TIM_20251201"
    #print(getTIMData("TN"))
    ExportTIMData(filename)
