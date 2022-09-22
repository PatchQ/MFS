import pandas as pd
import numpy as np
import openpyxl
import os
from tqdm import tqdm

def allvcpStock():
    #get stock excel file from path
    dir_path = "../YFData/"
    slist = list(map(lambda s: s.replace(".xlsx", ""), os.listdir(dir_path)))

    allvcp = pd.DataFrame()

    for sno in tqdm(slist[:]):
        tempsno = str(sno).lstrip("0")
        tempsno = tempsno.zfill(7)

        df = pd.read_excel(dir_path+"/"+sno+".xlsx")
        df = df.loc[df["VCP"]]
        allvcp = pd.concat([df, allvcp], ignore_index=True)

        
    allvcp.to_excel("Data/allvcp.xlsx",index=False)
    print("Finish")

        #if (df.iloc[df.shape[0]-1]["VCP"]):
        #    print(sno)

def vcpStock():
    #get stock excel file from path
    dir_path = "../YFData/"
    slist = list(map(lambda s: s.replace(".xlsx", ""), os.listdir(dir_path)))

    for sno in tqdm(slist[:]):
        tempsno = str(sno).lstrip("0")
        tempsno = tempsno.zfill(7)

        df = pd.read_excel(dir_path+"/"+sno+".xlsx")

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



#findHStock()
#findLStock()

#allvcpStock()
#vcpStock()
print(os.cpu_count())