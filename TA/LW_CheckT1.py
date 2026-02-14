
def checkT1(df, days, threshold=0.1):
    try:

        if len(df) < days:
            print(f"數據不足，無法計算 {days} 天波動")
            df["T1_"+str(days)] = False
            return df
        
        # 取最近N天的數據
        #recent_data = df.tail(days)
        
        # 計算最高價、最低價
        highest = df['High'].rolling(window=days).max()
        lowest = df['Low'].rolling(window=days).min()
                
        # 計算波動幅度
        volatility = (highest - lowest) / lowest
        
        # 檢查波動是否在閾值內
        df["T1_"+str(days)] = volatility <= threshold

        return df

    except Exception as e:
        print(f"Indicator error: {str(e)}")
        df["T1_"+str(days)] = False
        return df