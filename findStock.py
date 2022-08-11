import pandas as pd
import numpy as np
import openpyxl

stocklist = pd.read_excel("outputlist.xlsx")

stocklist = stocklist.loc[stocklist["10SMA"] > stocklist["20SMA"]]
stocklist = stocklist.loc[stocklist["20SMA"] > stocklist["50SMA"]]

stocklist = stocklist.loc[stocklist["50SMA"] > stocklist["100SMA"]]
stocklist = stocklist.loc[stocklist["100SMA"] > stocklist["250SMA"]]

stocklist = stocklist.loc[stocklist["Volume"] > stocklist["V10"]]
#stocklist = stocklist.loc[stocklist["Volume"] > stocklist["V20"]]
#stocklist = stocklist.loc[stocklist["Volume"] > stocklist["V50"]]

#stocklist = stocklist.loc[stocklist["V10"] > stocklist["V20"]]
#stocklist = stocklist.loc[stocklist["V20"] > stocklist["V50"]]

stocklist.to_excel("findstock.xlsx",index=False)



    
   
