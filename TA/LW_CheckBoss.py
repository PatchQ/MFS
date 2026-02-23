import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

try:
    from LW_Calindicator import *    
except ImportError:
    from TA.LW_Calindicator import *    

def checkBoss(df, sno, stype, swing_analysis):

    #print(sno)    
    df.index = pd.to_datetime(df.index)

    swing_analysis = swing_analysis.reset_index()
    swing_analysis['Date'] = swing_analysis['Date'].dt.strftime("%Y-%m-%d")

    swing_analysis['PATTERN'] = ""
    swing_analysis['LLLow'] = 0
    swing_analysis['LLDate'] = ""
    swing_analysis['HHClose'] = 0
    swing_analysis['HHDate'] = ""
    swing_analysis['HHHigh'] = 0
    swing_analysis['sno'] = sno
    swing_analysis['stype'] = stype
    
    df['classification'] = ""
    df['BOSS_PATTERN'] = ""
    df['LLLow'] = 0
    df['LLDate'] = ""
    df['HHClose'] = 0
    df['HHDate'] = ""
    df['HHHigh'] = 0
    df['VOLATILITY'] = 0
    df['22DLow'] = 0
    df['33DLow'] = 0
    df['BOSS_STATUS'] = ""
    df['HHEMA1'] = False
    df['HHEMA2'] = False
    df['HHEMA3'] = False
    df['BOSSB'] = False
    df['BOSSTP1'] = False
    df['BOSSTP2'] = False
    df['BOSSTP3'] = False
    df['BOSSCL1'] = False
    df['BOSSCL2'] = False
    df['BOSSTU1'] = False
    df['BOSSTU2'] = False
    

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(swing_analysis) - 2):
            templist = list(swing_analysis['Classification'].iloc[i:i+3])
            swing_analysis['PATTERN'].iloc[i] = ''.join(templist)

            swing_analysis['LLLow'].iloc[i] = swing_analysis['Price'].iloc[i+1]
            swing_analysis['LLDate'].iloc[i] = swing_analysis['Date'].iloc[i+1]
            swing_analysis['HHClose'].iloc[i] = swing_analysis['Close'].iloc[i+2]            
            swing_analysis['HHDate'].iloc[i] = swing_analysis['Date'].iloc[i+2]
            swing_analysis['HHHigh'].iloc[i] = swing_analysis['Price'].iloc[i+2]

            sadate = pd.to_datetime(swing_analysis['Date'].iloc[i])

            date_match = (df.index == sadate)            
            df.loc[date_match, "classification"] = swing_analysis["Classification"].iloc[i]
            df.loc[date_match, "LLLow"] = swing_analysis["LLLow"].iloc[i]
            df.loc[date_match, "LLDate"] = swing_analysis["LLDate"].iloc[i]
            df.loc[date_match, "HHClose"] = swing_analysis["HHClose"].iloc[i]
            df.loc[date_match, "HHDate"] = swing_analysis["HHDate"].iloc[i]
            df.loc[date_match, "HHHigh"] = swing_analysis["HHHigh"].iloc[i]
            df.loc[date_match, "BOSS_PATTERN"] = swing_analysis["PATTERN"].iloc[i]
            df.loc[date_match, "VOLATILITY"] = round(((swing_analysis["HHHigh"].iloc[i] - swing_analysis["LLLow"].iloc[i]) / swing_analysis["LLLow"].iloc[i]),2)

            hhdate = pd.to_datetime(swing_analysis["HHDate"].iloc[i])
            
            ema_values = df.loc[df.index == hhdate, "EMA1"]
            df.loc[date_match, "HHEMA1"] = ema_values.iloc[0] if len(ema_values) > 0 else np.nan
            ema_values = df.loc[df.index == hhdate, "EMA2"]
            df.loc[date_match, "HHEMA2"] = ema_values.iloc[0] if len(ema_values) > 0 else np.nan
            ema_values = df.loc[df.index == hhdate, "EMA3"]
            df.loc[date_match, "HHEMA3"] = ema_values.iloc[0] if len(ema_values) > 0 else np.nan

            etempdate = pd.to_datetime(swing_analysis["LLDate"].iloc[i])
            stempdate = etempdate - timedelta(days=22)
            df.loc[date_match, "22DLow"] = df.loc[(df.index>=stempdate) & (df.index<etempdate), "Low"].min()
            stempdate = etempdate - timedelta(days=33)
            df.loc[date_match, "33DLow"] = df.loc[(df.index>=stempdate) & (df.index<etempdate), "Low"].min()
            

    BOSS1Rule1 = (df['BOSS_PATTERN']=="LHLLHH") | (df['BOSS_PATTERN']=="HHLLHH")
    BOSS1Rule2 = df['HHClose']>df['High']
    BOSS1Rule3 = df['VOLATILITY']>=0.14
    
    df["BOSS1"] = (BOSS1Rule1 & BOSS1Rule2 & BOSS1Rule3) # & df["HHEMA3"])    
    
    tempdf = df.loc[df["BOSS1"]]    
    #tempdf = tempdf.reset_index()


    df['bullish_ratio'] = 0.00
    df['bullish_count'] = 0
    df['strong_bullish'] = 0
    df['medium_bullish'] = 0
    df['weak_bullish'] = 0
    
    
    for i in range(len(tempdf)):
        sdate = pd.to_datetime(tempdf["LLDate"].iloc[i])
        edate = pd.to_datetime(tempdf["HHDate"].iloc[i])
        fdf = df.loc[(df.index>=sdate) & (df.index<=edate)]

        bullish_count, bullish_ratio = calCandleStick(fdf)
        strong_bullish, medium_bullish, weak_bullish = calCandleStickBody(fdf)
         

        date_match = (df.index == tempdf.index[i])
        df.loc[date_match, "bullish_count"] = bullish_count
        df.loc[date_match, "bullish_ratio"] = bullish_ratio
        df.loc[date_match, "strong_bullish"] = strong_bullish
        df.loc[date_match, "medium_bullish"] = medium_bullish
        df.loc[date_match, "weak_bullish"] = weak_bullish        
    
    BOSS2Rule1 = df['LLLow']<=df['22DLow'] 
    BOSS2Rule2 = df["bullish_ratio"]>=0.65
    BOSS2Rule3 = df["strong_bullish"]>=1
    BOSS2Rule4 = df["bullish_count"]>=4    

    df["BOSS2"] = (df["BOSS1"] & BOSS2Rule1 & BOSS2Rule2 & BOSS2Rule3 & BOSS2Rule4)  
    
    df.loc[df["BOSS2"], "buy_price"] = round(((df["HHHigh"] + df["LLLow"]) / 2),2)
    df.loc[df["BOSS2"], "cl_price"] = df["LLLow"]
    df.loc[df["BOSS2"], "tp1_price"] = df["HHHigh"]
    df.loc[df["BOSS2"], "tp2_price"] = df["buy_price"] + (df["HHHigh"] - df["buy_price"]) * 2 
    df.loc[df["BOSS2"], "tp3_price"] = df["buy_price"] + (df["HHHigh"] - df["buy_price"]) * 3 
    df.loc[df["BOSS2"], "BOSS_STATUS"] = "SB1-"+df.loc[df["BOSS2"]].index.strftime("%Y/%m/%d")

    tempdf = df.loc[df["BOSS2"]]    
    #tempdf = tempdf.reset_index()

    for i in range(len(tempdf)):

        lastbuydate = pd.to_datetime("1900-01-01") 
        lastcl1date = pd.to_datetime("1900-01-01") 
        lasttp1date = pd.to_datetime("1900-01-01") 
        lastcl2date = pd.to_datetime("1900-01-01") 
        lasttp2date = pd.to_datetime("1900-01-01") 
        lasttp3date = pd.to_datetime("1900-01-01") 
        tp1 = False
        tp2 = False
        tp3 = False
        cl1 = False
        cl2 = False
        buy = False    

        buy_price = tempdf["buy_price"].iloc[i]
        cl_price = tempdf["cl_price"].iloc[i]        
        tp1_price = tempdf["tp1_price"].iloc[i]
        tp2_price = tempdf["tp2_price"].iloc[i]
        tp3_price = tempdf["tp3_price"].iloc[i]
        hh_price = tp1_price

        hhdate = pd.to_datetime(tempdf["HHDate"].iloc[i])

        try:
            buydeadline = df[df.index >= hhdate].index[22]
        except IndexError:            
            buydeadline = df[df.index >= hhdate].index[-1]

        startbossdate = tempdf.index[i].strftime("%Y/%m/%d")               


        buydate_mask = (df.index < buydeadline) & (df.index > hhdate) & (buy_price>=df["Low"]*0.995) & df["EMA3"]
        buydates = df[buydate_mask].index

        if len(buydates)!=0:
            buy = True            
            lastbuydate = buydates[0]
        
        highdate_mask = (df.index <= lastbuydate) & (df.index > hhdate) & (df["High"]>hh_price)
        highdates = df[highdate_mask].index

        if len(highdates)!=0:
            buy = False            

        if buy:
            df.loc[lastbuydate,'BOSS_STATUS'] = "BY1-"+startbossdate
            df.loc[lastbuydate,'BOSSB'] = True
            df.loc[lastbuydate,'buy_price'] = buy_price
            df.loc[lastbuydate,'cl_price'] = cl_price
            df.loc[lastbuydate,'tp1_price'] = tp1_price
            df.loc[lastbuydate,'tp2_price'] = tp2_price
            df.loc[lastbuydate,'tp3_price'] = tp3_price
            #print("BUYDate : "+lastbuydate.strftime("%Y-%m-%d"))
            try:
                tp1deadline = df[df.index>=lastbuydate].index[30]
            except IndexError:            
                tp1deadline = df[df.index>=lastbuydate].index[-1]

            tp1date_mask = (df.index < tp1deadline) & (df.index >= lastbuydate) & (df["High"]>=tp1_price*0.995)
            #tp1date_mask = (df.index >= lastbuydate) & (df["High"]>=tp1_price*0.995)
            tp1dates = df[tp1date_mask].index
            
            cl1date_mask = (df.index < tp1deadline) & (df.index >= lastbuydate) & (df["Close"]<cl_price)
            #cl1date_mask = (df.index >= lastbuydate) & (df["Close"]<cl_price)
            cl1dates = df[cl1date_mask].index            

            if len(tp1dates)!=0:
                tp1=True
                lasttp1date = tp1dates[0]

            if len(cl1dates)!=0:                
                cl1=True
                lastcl1date = cl1dates[0]

            if (tp1deadline < pd.Timestamp(datetime.now().date())):
                if (cl1==False and tp1==False):                    
                    if (round(((df.loc[tp1deadline,'Low'] - buy_price) / buy_price),2)>=0.01):
                        df.loc[tp1deadline,'BOSSTU1'] = True       
                        df.loc[tp1deadline,'BOSS_STATUS'] = "TU1-"+startbossdate                               
                    else:
                        df.loc[tp1deadline,'BOSSTU2'] = True                    
                        df.loc[tp1deadline,'BOSS_STATUS'] = "TU2-"+startbossdate                  

            if cl1:
                if tp1:
                    if (lasttp1date>=lastcl1date):
                        df.loc[lastcl1date,'BOSS_STATUS'] = "CL1-"+startbossdate
                        df.loc[lastcl1date,'BOSSCL1'] = True                  
                        tp1=False      
                        #print("CL1 : "+lastcl1date.strftime("%Y-%m-%d"))
                    else:
                        df.loc[lasttp1date,'BOSS_STATUS'] = "TP1-"+startbossdate
                        df.loc[lasttp1date,'BOSSTP1'] = True
                        tp1=True
                        #print("TP1 : "+lasttp1date.strftime("%Y-%m-%d"))                             
                else:
                    df.loc[lastcl1date,'BOSS_STATUS'] = "CL1-"+startbossdate
                    df.loc[lastcl1date,'BOSSCL1'] = True
                    #print("CL1 : "+lastcl1date.strftime("%Y-%m-%d"))
            
            if tp1:
                df.loc[lasttp1date,'BOSS_STATUS'] = "TP1-"+startbossdate
                df.loc[lasttp1date,'BOSSTP1'] = True
                #print("TP1 : "+lasttp1date.strftime("%Y-%m-%d"))     

                try:
                    tp2deadline = df[df.index>=lasttp1date].index[30]
                except IndexError:            
                    tp2deadline = df[df.index>=lasttp1date].index[-1]
                
                tp2date_mask = (df.index < tp2deadline) & (df.index >= lasttp1date) & (df["High"]>=tp2_price*0.99)
                #tp2date_mask = (df.index >= lasttp1date) & (df["High"]>=tp2_price*0.99)
                tp2dates = df[tp2date_mask].index        
                            
                cl2date_mask = (df.index < tp2deadline) & (df.index >= lasttp1date) & (df["Close"]<cl_price)
                #cl2date_mask = (df.index >= lasttp1date) & (df["Close"]<cl_price)
                cl2dates = df[cl2date_mask].index

                if len(tp2dates)!=0:
                    tp2 = True                
                    lasttp2date = tp2dates[0]

                if len(cl2dates)!=0:
                    cl2 = True
                    lastcl2date = cl2dates[0]

                if cl2:
                    if tp2:
                        if (lasttp2date>=lastcl2date):
                            df.loc[lastcl2date,'BOSS_STATUS'] = "CL2-"+startbossdate
                            df.loc[lastcl2date,'BOSSCL2'] = True
                            tp2=False
                            #print("CL2 : "+lastcl2date.strftime("%Y-%m-%d")+ " : "+startbossdate)
                        else:
                            df.loc[lasttp2date,'BOSS_STATUS'] = "TP2-"+startbossdate
                            df.loc[lasttp2date,'BOSSTP2'] = True
                            tp2=True
                            #print("TP2 : "+lasttp2date.strftime("%Y-%m-%d")+ " : "+startbossdate)
                    else:
                        df.loc[lastcl2date,'BOSS_STATUS'] = "CL2-"+startbossdate
                        df.loc[lastcl2date,'BOSSCL2'] = True
                        #print("CL2 : "+lastcl2date.strftime("%Y-%m-%d")+ " : "+startbossdate)                               
                
                if tp2:
                    df.loc[lasttp2date,'BOSS_STATUS'] = "TP2-"+startbossdate
                    df.loc[lasttp2date,'BOSSTP2'] = True
                    #print("TP2 : "+lasttp2date.strftime("%Y-%m-%d")+ " : "+startbossdate)
                    try:
                        tp3deadline = df[df.index>=lasttp2date].index[30]
                    except IndexError:            
                        tp3deadline = df[df.index>=lasttp2date].index[-1]

                    tp3date_mask = (df.index < tp3deadline) & (df.index >= lasttp2date) & (df["High"]>=tp3_price*0.99)
                    #tp2date_mask = (df.index >= lasttp1date) & (df["High"]>=tp2_price*0.99)
                    tp3dates = df[tp3date_mask].index        

                    if len(tp3dates)!=0:
                        tp3 = True                
                        lasttp3date = tp3dates[0]

                    if tp3:
                        df.loc[lasttp3date,'BOSS_STATUS'] = "TP3-"+startbossdate
                        df.loc[lasttp3date,'BOSSTP3'] = True


    return df
