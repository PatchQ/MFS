import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
import concurrent.futures as cf
import os
from tqdm import tqdm
import time as t
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN

from LW_Calindicator import *

PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/" 

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SwingPointAnalyzer:
    def __init__(self, symbol, period="6mo", interval="1d"):
        """
        初始化分析器
        
        Parameters:
        -----------
        symbol : str
            股票代號 (如: "2330.TW")
        period : str
            資料期間 (如: "6mo", "1y", "2y")
        interval : str
            資料間隔 (如: "1d", "1h", "1wk")
        """
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.df = None
        self.swing_points = []  # 儲存擺動點 (index, price, type)
        self.HH_HL_LH_LL = []   # 儲存 HH, HL, LH, LL 點
        
    def fetch_data(self):
        """從 yfinance 下載股票數據"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.df = ticker.history(period=self.period, interval=self.interval)
            
            if self.df.empty:
                print(f"無法下載 {self.symbol} 的數據")
                return False
            
            print(f"下載完成: {self.symbol}, 資料筆數: {len(self.df)}")
            return True
            
        except Exception as e:
            print(f"下載數據時發生錯誤: {e}")
            return False
    
    def calculate_daily_volatility(self, window=20):
        """
        計算每日波幅的平均值
        
        Parameters:
        -----------
        window : int
            計算平均波幅的視窗大小
        """
        if self.df is None or self.df.empty:
            print("請先下載數據")
            return None
        
        # 計算每日波幅 (當日最高價 - 當日最低價)
        self.df['Daily_Range'] = self.df['High'] - self.df['Low']
        
        # 計算波幅的平均值 (使用滾動平均)
        self.df['Avg_Range'] = self.df['Daily_Range'].rolling(window=window).mean()
        
        # 計算轉彎閾值 (平均波幅 + 2%)
        self.df['Turn_Threshold'] = self.df['Avg_Range'] + 2
        
        return self.df['Avg_Range'].iloc[-1]  # 返回最新的平均波幅
    
    def find_swing_points(self, window=20):
        """
        找出擺動點 (高點和低點)
        使用拋物線轉向概念: 價格變化超過閾值且出現轉向時標記為擺動點
        """
        if self.df is None or self.df.empty:
            print("請先下載數據")
            return
        
        # 確保有計算波幅
        if 'Turn_Threshold' not in self.df.columns:
            self.calculate_daily_volatility()
        
        # 初始化變數
        swing_points = []
        trend = None  # 趨勢: 'up' 或 'down'
        last_extreme = None  # 上一個極值點 (高點或低點)
        last_extreme_idx = -1
        
        # 遍歷數據 (從第20天開始，確保有平均波幅)
        for i in range(window, len(self.df)):
            current_price = self.df['Close'].iloc[i]
            current_high = self.df['High'].iloc[i]
            current_low = self.df['Low'].iloc[i]
            threshold = self.df['Turn_Threshold'].iloc[i]

            print(threshold)
            
            # 如果是第一個點，初始化
            if last_extreme is None:
                # 判斷初始趨勢
                if i > window and self.df['Close'].iloc[i] > self.df['Close'].iloc[i-1]:
                    trend = 'up'
                    last_extreme = current_high
                    last_extreme_idx = i
                    swing_points.append((i, last_extreme, 'high'))
                else:
                    trend = 'down'
                    last_extreme = current_low
                    last_extreme_idx = i
                    swing_points.append((i, last_extreme, 'low'))
                continue
            
            if trend == 'up':
                # 上升趨勢中，更新最高點
                if current_high > last_extreme:
                    last_extreme = current_high
                    last_extreme_idx = i
                    # 更新最後一個擺動點
                    if swing_points[-1][2] == 'high':
                        swing_points[-1] = (i, last_extreme, 'high')
                    else:
                        swing_points.append((i, last_extreme, 'high'))
                
                # 檢查是否轉向: 從高點下跌超過閾值
                elif last_extreme - current_low > threshold:
                    trend = 'down'
                    last_extreme = current_low
                    last_extreme_idx = i
                    swing_points.append((i, last_extreme, 'low'))
                    
            else:  # trend == 'down'
                # 下降趨勢中，更新最低點
                if current_low < last_extreme:
                    last_extreme = current_low
                    last_extreme_idx = i
                    # 更新最後一個擺動點
                    if swing_points[-1][2] == 'low':
                        swing_points[-1] = (i, last_extreme, 'low')
                    else:
                        swing_points.append((i, last_extreme, 'low'))
                
                # 檢查是否轉向: 從低點上漲超過閾值
                elif current_high - last_extreme > threshold:
                    trend = 'up'
                    last_extreme = current_high
                    last_extreme_idx = i
                    swing_points.append((i, last_extreme, 'high'))
        
        self.swing_points = swing_points
        print(f"找到 {len(swing_points)} 個擺動點")
        return swing_points
    
    def identify_HH_HL_LH_LL(self):
        """
        識別 HH, HL, LH, LL 點
        規則:
        - HH: 高點且比前一個高點高
        - HL: 低點且比前一個低點高
        - LH: 高點且比前一個高點低
        - LL: 低點且比前一個低點低
        """
        if not self.swing_points:
            print("請先找出擺動點")
            return
        
        HH_HL_LH_LL = []
        last_high = None
        last_low = None
        
        for i, (idx, price, point_type) in enumerate(self.swing_points):
            classification = None
            
            if point_type == 'high':
                if last_high is not None:
                    if price > last_high:
                        classification = 'HH'  # 高高低
                    else:
                        classification = 'LH'  # 低高高
                else:
                    classification = 'First_High'  # 第一個高點
                last_high = price
                
            else:  # point_type == 'low'
                if last_low is not None:
                    if price > last_low:
                        classification = 'HL'  # 高低高
                    else:
                        classification = 'LL'  # 低低低
                else:
                    classification = 'First_Low'  # 第一個低點
                last_low = price
            
            # 將結果添加到列表中
            HH_HL_LH_LL.append({
                'Index': idx,
                'Date': self.df.index[idx],
                'Price': price,
                'Type': point_type,
                'Classification': classification
            })
        
        self.HH_HL_LH_LL = HH_HL_LH_LL
        
        # 統計各種類型的數量
        classifications = [point['Classification'] for point in HH_HL_LH_LL]
        stats = pd.Series(classifications).value_counts()
        print("分類統計:")
        print(stats)
        
        return HH_HL_LH_LL
    
    def visualize_results(self):
        """可視化結果"""
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.rcParams['font.sans-serif'] = [
            'DejaVu Sans',     
            'Arial Unicode MS', 
            'Microsoft YaHei',  
            'SimHei',           
            'sans-serif'
        ]

        matplotlib.rcParams['axes.unicode_minus'] = False
        
        if self.df is None or not self.HH_HL_LH_LL:
            print("沒有數據可視化")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 圖1: 價格走勢和擺動點
        ax1.plot(self.df.index, self.df['Close'], label='Close Price', alpha=0.5)
        
        # 繪製擺動點
        for point in self.swing_points:
            idx, price, point_type = point
            color = 'red' if point_type == 'high' else 'green'
            marker = 'v' if point_type == 'high' else '^'
            ax1.scatter(self.df.index[idx], price, color=color, marker=marker, s=100)
        
        ax1.set_title(f'{self.symbol} - 價格走勢和擺動點')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('價格')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 圖2: HH, HL, LH, LL 分類
        ax2.plot(self.df.index, self.df['Close'], label='Close Price', alpha=0.5)
        
        # 不同分類使用不同顏色和標記
        colors = {
            'HH': 'red',      # 高高低 - 紅色向上箭頭
            'HL': 'green',    # 高低高 - 綠色向上箭頭
            'LH': 'orange',   # 低高高 - 橙色向下箭頭
            'LL': 'blue',     # 低低低 - 藍色向下箭頭
            'First_High': 'purple',
            'First_Low': 'brown'
        }
        
        markers = {
            'HH': '^', 'HL': '^', 
            'LH': 'v', 'LL': 'v',
            'First_High': 'o', 'First_Low': 'o'
        }
        
        for point in self.HH_HL_LH_LL:
            if point['Classification'] in colors:
                ax2.scatter(
                    point['Date'], point['Price'],
                    color=colors[point['Classification']],
                    marker=markers[point['Classification']],
                    s=100,
                    label=point['Classification'] if point['Classification'] not in 
                    [p.get_label() for p in ax2.collections] else ""
                )
        
        ax2.set_title(f'{self.symbol} - HH/HL/LH/LL 分類')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('價格')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_summary(self):
        """獲取分析摘要"""
        if not self.HH_HL_LH_LL:
            return None
        
        summary = {
            '股票代號': self.symbol,
            '數據期間': self.period,
            '數據間隔': self.interval,
            '總擺動點數': len(self.swing_points),
            'HH數量': len([p for p in self.HH_HL_LH_LL if p['Classification'] == 'HH']),
            'HL數量': len([p for p in self.HH_HL_LH_LL if p['Classification'] == 'HL']),
            'LH數量': len([p for p in self.HH_HL_LH_LL if p['Classification'] == 'LH']),
            'LL數量': len([p for p in self.HH_HL_LH_LL if p['Classification'] == 'LL']),
        }
        
        # 計算趨勢強度 (HH+HL vs LH+LL)
        bullish_points = summary['HH數量'] + summary['HL數量']
        bearish_points = summary['LH數量'] + summary['LL數量']
        
        if bullish_points + bearish_points > 0:
            summary['多頭強度'] = bullish_points / (bullish_points + bearish_points)
            summary['空頭強度'] = bearish_points / (bullish_points + bearish_points)
        else:
            summary['多頭強度'] = 0
            summary['空頭強度'] = 0
        
        return summary

# 使用範例
def main():
    # 創建分析器
    analyzer = SwingPointAnalyzer(
        symbol="0011.HK",  # 台積電
        period="2y",      # 6個月數據
        interval="1d"      # 日線數據
    )
    
    # 下載數據
    if not analyzer.fetch_data():
        return
    
    # 計算波幅
    avg_volatility = analyzer.calculate_daily_volatility(window=60)
    print(f"平均每日波幅: {avg_volatility:.2f}")
    
    # 找出擺動點
    swing_points = analyzer.find_swing_points(window=60)
    
    # 識別 HH, HL, LH, LL
    classifications = analyzer.identify_HH_HL_LH_LL()
    
    # 顯示結果
    summary = analyzer.get_summary()
    print("\n分析摘要:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # 可視化結果
    analyzer.visualize_results()
    
    # 輸出詳細的 HH/HL/LH/LL 點
    print("\n詳細的擺動點分類:")
    df_results = pd.DataFrame(classifications)
    print(df_results.to_string(index=False))

def main2():    
    analyzer = SwingPointAnalyzer(symbol="0011.HK", period="5y", interval="1d")
    analyzer.fetch_data()
    analyzer.calculate_daily_volatility(window=20)
    analyzer.find_swing_points(window=20)
    classifications = analyzer.identify_HH_HL_LH_LL()

    summary = analyzer.get_summary()
    print("\n分析摘要:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # 輸出詳細的 HH/HL/LH/LL 點
    print("\n詳細的擺動點分類:")
    df_results = pd.DataFrame(classifications)
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main2()


# # 使用示例
# def calHHLL(df,window):

#     stock_data = df.copy()    
#     stock_data.index = pd.to_datetime(stock_data.index,utc=True).tz_convert('Asia/Shanghai')         
#     stock_data = extendData(stock_data)       
        
#     # 创建分析器
#     analyzer = RobustSwingPointAnalyzer()
    
#     # 执行综合分析
#     swing_df = analyzer.comprehensive_swing_analysis(window,stock_data['High'],stock_data['Low'],
#                          stock_data['Close'],stock_data['Volume'])    
    
#     # 计算置信度
#     swing_df = analyzer.calculate_confidence_score(swing_df)
    
#     # 过滤高置信度的摆动点
#     if len(swing_df)>0:
#         swing_df = swing_df[swing_df['confidence_score'] >= 40]            
#         resultdf = swing_df.reset_index().drop(['level_0','index'],axis=1)
#     else:
#         resultdf = pd.DataFrame()    
    
#     return resultdf


# def AnalyzeData(sno,stype):
       
#     df = pd.read_csv(PATH+"/"+stype+"/"+sno+".csv",index_col=0)
#     df = convertData(df)

#     #print(sno)
#     df = calEMA(df)

#     #tempdf1 = calHHLL(df,3)
#     tempdf = calHHLL(df,5)
    
#     #tempdf = pd.concat([tempdf1,tempdf2])

#     if len(tempdf)>0:
#         tempdf = tempdf.sort_values('date').drop_duplicates(subset=['date'], keep='last')
#         df = checkLHHHLL(df, sno, stype, tempdf)        
#         df = calT1(df,50)        
#         df = df.reset_index()
#         df.to_csv(OUTPATH+"/"+stype+"/P_"+sno+".csv",index=False)


# def YFprocessData(stype):

#     snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(PATH+"/"+stype)))
#     SLIST = pd.DataFrame(snolist, columns=["sno"])
#     SLIST = SLIST.assign(stype=stype+"")
#     SLIST = SLIST[7:8]

#     with cf.ProcessPoolExecutor(max_workers=5) as executor:
#         list(tqdm(executor.map(AnalyzeData,SLIST["sno"],SLIST["stype"],chunksize=1),total=len(SLIST)))


# if __name__ == '__main__':
#     start = t.perf_counter()

#     YFprocessData("L")    
#     #YFprocessData("M")
#     #YFprocessData("S")

#     finish = t.perf_counter()
#     print(f'It took {round(finish-start,2)} second(s) to finish.')
    