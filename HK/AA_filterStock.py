import pandas as pd
import numpy as np
import openpyxl


stocklist = pd.read_excel("Data/stocklist.xlsx",dtype=str)
stocklist1 = pd.DataFrame()

stocklist["資格"] = stocklist["市值"].str.find("億")
#stocklist = stocklist.query("資格 >=5")
stocklist = stocklist.loc[stocklist["資格"] >= 5]
stocklist = stocklist.drop(["資格"], axis=1)
stocklist["數字市值"] = stocklist["市值"].apply(lambda s: s.replace("億","000000").replace(",","").replace(".","")).astype(float)
    
    
#只選出市值大於50億
stocklist = stocklist.query("數字市值 >= 5000000000")
#df_td = df_td.loc[df_td["數字市值"] >= 5000000000]

stocklist = stocklist.sort_values(by="股票編號")

# stocklist1 = stocklist.filter(items=["股票編號","股票名稱","行業編號","數字市值","市盈率",
#                         "市賬率","收益率","毛利率","派息比率","收入","年度收入增長","經營溢利"])

stocklist1 = stocklist.loc[:,["股票編號","股票名稱","行業編號","市值","數字市值","市盈率",
                         "市賬率","收益率","毛利率","派息比率","收入","年度收入增長","經營溢利"]]

stocklist1.to_excel("Data/filterstock1.xlsx",index=False)


#流動比率 = 流動資產 / 流動負債 (>=1)
stocklist["流動比率"] = stocklist["流動比率"].astype(float)
stocklist = stocklist.loc[stocklist["流動比率"] >= 1.00]

#速動比率 = 速動資產(變現性較好的資產) / 流動負債 (>=1)
stocklist["速動比率"] = stocklist["速動比率"].astype(float)
stocklist = stocklist.loc[stocklist["速動比率"] >= 1.00]

#股本回報率（ROE） ＝ 稅後淨利 ÷ 股東權益
#ROE 越高代表公司經營效率、獲利能力越好
stocklist["股本回報率"] = stocklist["股本回報率"].apply(lambda s: s.replace("%","")).astype(float)
stocklist = stocklist.loc[stocklist["股本回報率"] >= 0.00]

#資產回報率（ROA）＝ 總資產報酬率 ＝ 稅後淨利 ÷ 總資產
#ROA 越高代表公司越能善用資產創造收入
stocklist["資產回報率"] = stocklist["資產回報率"].apply(lambda s: s.replace("%","")).astype(float)
stocklist = stocklist.loc[stocklist["資產回報率"] >= 0.00]

#毛利率 ＝（營業收入 - 營業成本）/ 營業收入 x 100%
#毛利率低不一定不好，但太低一定不好 (例如低於5%)
stocklist["毛利率"] = stocklist["毛利率"].apply(lambda s: s.replace("%","")).astype(float)
stocklist = stocklist.loc[stocklist["毛利率"] >= 5.00]

#邊際利潤（Profit Margin）是衡量一家公司賺錢能力的比率，數字愈高，代表其控制成本的能力愈佳
#邊際利潤率 = (銷售收入 - 變動成本) / 銷售收入 x 100%
stocklist["邊際利潤率"] = stocklist["邊際利潤率"].apply(lambda s: s.replace("%","").replace(",","")).astype(float)
stocklist = stocklist.loc[stocklist["邊際利潤率"] >= 0]


stocklist = stocklist.sort_values(by="股票編號")

stocklist.to_excel("Data/filterstock.xlsx",index=False)



    
   
