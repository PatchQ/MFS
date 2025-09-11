import pandas as pd
import numpy as np
import concurrent.futures as cf
import yfinance as yf
import os
from tqdm import tqdm


#get stock excel file from path
OUTPATH = "../SData/P_YFData/"
#EDATE = "2022-09-01"
EDATE = "2025-09-03"
SLIST = list(map(lambda s: s.replace(".xlsx", ""), os.listdir(OUTPATH)))
SLIST = SLIST[:]

#testing


def allvcpStock(sno):

    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    df = pd.read_excel(OUTPATH+sno+".xlsx",index_col=0)
    df = df.loc[df.index>=EDATE]
    df = df.loc[df["VCP"]]
    df = df.reset_index()

    return df


def allemaStock(sno):

    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    df = pd.read_excel(OUTPATH+sno+".xlsx",index_col=0)
    df = df.loc[df.index>=EDATE]
    df = df.loc[df["EMA"]]
    df = df.reset_index()

    return df


def vcpStock(sno):

    tempsno = str(sno).lstrip("0")
    tempsno = tempsno.zfill(7)

    df = pd.read_excel(OUTPATH+sno+".xlsx")

    if (df.iloc[df.shape[0]-1]["VCP"]):
        print(sno)



def findHStock():

    stocklist = pd.read_excel("Data/outputlist.xlsx")

    stocklist = stocklist.loc[stocklist["Close"] > stocklist["10SMA"]]
    #stocklist = stocklist.loc[stocklist["Close"] > stocklist["250SMA"]]
    #stocklist = stocklist.loc[stocklist["Close"] > stocklist["100SMA"]]
    #stocklist = stocklist.loc[stocklist["Close"] > stocklist["50SMA"]]

    stocklist = stocklist.loc[stocklist["10SMA"] > stocklist["20SMA"]]
    stocklist = stocklist.loc[stocklist["20SMA"] > stocklist["50SMA"]]
    stocklist = stocklist.loc[stocklist["50SMA"] > stocklist["100SMA"]]
    stocklist = stocklist.loc[stocklist["100SMA"] > stocklist["250SMA"]]

    #stocklist = stocklist.loc[stocklist["Volume"] > stocklist["V10"]]
    #stocklist = stocklist.loc[stocklist["Volume"] > stocklist["V20"]]
    #stocklist = stocklist.loc[stocklist["Volume"] > stocklist["V50"]]

    stocklist.to_excel("Data/findHStock.xlsx",index=False)



def findLStock():

    stocklist = pd.read_excel("Data/outputlist.xlsx")

    #stocklist = stocklist.loc[stocklist["Close"] < stocklist["10SMA"]]

    stocklist = stocklist.loc[stocklist["10SMA"] < stocklist["20SMA"]]
    stocklist = stocklist.loc[stocklist["20SMA"] < stocklist["50SMA"]]
    stocklist = stocklist.loc[stocklist["50SMA"] < stocklist["100SMA"]]
    stocklist = stocklist.loc[stocklist["100SMA"] < stocklist["250SMA"]]

    stocklist = stocklist.loc[stocklist["Volume"] > stocklist["V10"]]
    stocklist = stocklist.loc[stocklist["Volume"] > stocklist["V20"]]
    stocklist = stocklist.loc[stocklist["Volume"] > stocklist["V50"]]

    stocklist = stocklist.loc[stocklist["V10"] > stocklist["V20"]]
    stocklist = stocklist.loc[stocklist["V20"] > stocklist["V50"]]
    stocklist = stocklist.loc[stocklist["V50"] > stocklist["V100"]]
    stocklist = stocklist.loc[stocklist["V100"] > stocklist["V250"]]

    stocklist.to_excel("Data/findLStock.xlsx",index=False)




def main1():
    with cf.ProcessPoolExecutor(max_workers=17) as executor:
        list(tqdm(executor.map(vcpStock,SLIST,chunksize=2),total=len(SLIST)))
        #list(tqdm(executor.map(allvcpStock,slist,chunksize=2),total=len(slist)))
        #list(tqdm(executor.map(findHStock,slist,chunksize=2),total=len(slist)))
        #list(tqdm(executor.map(findLStock,slist,chunksize=2),total=len(slist)))

def mainEMA():
    allema = pd.DataFrame()

    with cf.ProcessPoolExecutor(max_workers=17) as executor:
        for tempdf in tqdm(executor.map(allemaStock,SLIST,chunksize=2),total=len(SLIST)):            
            allema = pd.concat([tempdf, allema], ignore_index=True)
            

        allema.to_excel("Data/allema.xlsx",index=False)
        print("Finish")        

def main2():
    allvcp = pd.DataFrame()

    with cf.ProcessPoolExecutor(max_workers=17) as executor:
        for tempdf in tqdm(executor.map(allvcpStock,SLIST,chunksize=2),total=len(SLIST)):
            allvcp = pd.concat([tempdf, allvcp], ignore_index=True)

        allvcp.to_excel("Data/allvcp.xlsx",index=False)
        print("Finish")        

def main3():
    #S = yf.Ticker("ES=F")
    S = yf.Ticker("^HSI")

    print(S.info)

    df = S.history(period="max")

    #print(S.actions)
    #print(S.financials)
    #print(S.major_holders)
    #print(S.institutional_holders)

    #show cashflow
    print(S.cashflow)
    print(S.quarterly_cashflow)

    #show earnings
    print(S.earnings)
    print(S.quarterly_earnings)

    print(df)


if __name__ == '__main__':
    #main1()
    #main2()
    #main3()
    mainEMA()

