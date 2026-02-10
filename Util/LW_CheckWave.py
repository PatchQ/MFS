import pandas as pd
import warnings

    
def checkWave(df, sno, stype, swing_analysis):

    #print(sno)    
    df.index = pd.to_datetime(df.index)

    swing_analysis = swing_analysis.reset_index()
    swing_analysis['Date'] = swing_analysis['Date'].dt.strftime("%Y-%m-%d")

    swing_analysis['PATTERN'] = ""
    swing_analysis['WLow'] = 0
    swing_analysis['WLDate'] = ""
    swing_analysis['WHigh'] = 0
    swing_analysis['WHDate'] = ""
    swing_analysis['sno'] = sno
    swing_analysis['stype'] = stype
    
    df['classification'] = ""
    df['PATTERN'] = ""
    df['WLow'] = 0
    df['WLDate'] = ""        
    df['WHigh'] = 0
    df['WHDate'] = ""        
    df['HHHL'] = False
    

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") #HH-HL-HH-HL
        for i in range(len(swing_analysis) - 4):
            templist = list(swing_analysis['Classification'].iloc[i:i+5])
            swing_analysis['PATTERN'].iloc[i] = ''.join(templist)

            swing_analysis['WLow'].iloc[i] = swing_analysis['Price'].iloc[i+2]
            swing_analysis['WLDate'].iloc[i] = swing_analysis['Date'].iloc[i+2]
            swing_analysis['WHigh'].iloc[i] = swing_analysis['Price'].iloc[i+3]
            swing_analysis['WHDate'].iloc[i] = swing_analysis['Date'].iloc[i+3]
            

            sadate = pd.to_datetime(swing_analysis['Date'].iloc[i])

            date_match = (df.index == sadate)
            df.loc[date_match, "classification"] = swing_analysis["Classification"].iloc[i]
            df.loc[date_match, "WLow"] = swing_analysis["WLow"].iloc[i]
            df.loc[date_match, "WLDate"] = swing_analysis["WLDate"].iloc[i]            
            df.loc[date_match, "WHigh"] = swing_analysis["WHigh"].iloc[i]
            df.loc[date_match, "WHDate"] = swing_analysis["WHDate"].iloc[i]        
            df.loc[date_match, "PATTERN"] = swing_analysis["PATTERN"].iloc[i]
            #df.loc[date_match, "VOLATILITY"] = round(((swing_analysis["HHHigh"].iloc[i] - swing_analysis["HLLow"].iloc[i]) / swing_analysis["HLLow"].iloc[i]),2)            

    WAVERule1 = df['PATTERN']=="LLHHHLHHHL"
    WAVERule2 = df['PATTERN']=="HLHHHLHHHL"
    
    df["HHHL"] = (WAVERule1 | WAVERule2)

    return df