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

from ..Util.LW_Calindicator import *

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
        self.df['Avg_Range'] = self.simple_robust_average()
        #self.df['Avg_Range'] = self.df['Daily_Range'].rolling(window=window).mean()
        
        # 計算轉彎閾值 (平均波幅 x 1.5-2)
        self.df['Turn_Threshold'] = self.df['Avg_Range'] * 1.5
        
        return self.df['Avg_Range'].iloc[-1]  # 返回最新的平均波幅
    
    def simple_robust_average(self, window=20, threshold=0.5):
        """
        簡化版本：只排除比前值大超過threshold%的單個值
        """
        avg_values = []
        
        for i in range(len(self.df['Daily_Range'])):
            if i < window - 1:
                avg_values.append(np.nan)
                continue
            
            # 獲取窗口數據
            window_data = self.df['Daily_Range'].iloc[i-window+1:i+1]
            
            # 計算每個值相對於前一個值的變化
            changes = window_data.pct_change().fillna(0)
            
            # 找出異常值索引
            outlier_indices = changes[changes > threshold].index
            
            if len(outlier_indices) > 0:
                # 排除異常值
                filtered_data = window_data.drop(outlier_indices)
                # 計算平均
                avg = filtered_data.mean() if len(filtered_data) > 0 else np.nan
            else:
                # 沒有異常值，正常計算
                avg = window_data.mean()
            
            avg_values.append(avg)
        
        return pd.Series(avg_values, index=self.df['Daily_Range'].index)              
    
    def find_swing_points(self, window=20, min_trend_length=20, lookback_multiplier=3):
        """
        找出擺動點，並在長期趨勢確立後重新調整之前的點
        增加趨勢確認機制：需要突破前一個LH才確認上升趨勢
        
        Parameters:
        -----------
        window : int
            計算波動率的窗口
        min_trend_length : int
            最小趨勢長度（交易日），超過此長度認為是長期趨勢
        lookback_multiplier : int
            回看倍數，用於判斷趨勢突破（回看前N個擺動點）
        """
        if self.df is None or self.df.empty:
            print("請先下載數據")
            return
        
        # 確保有計算波幅
        if 'Turn_Threshold' not in self.df.columns:
            self.calculate_daily_volatility()
        
        # 初始化變數
        swing_points = []  # 格式: (idx, price, type)
        trend = None  # 當前趨勢: 'up'/'down'/'consolidation'
        last_extreme = None  # 上一個極值點價格
        last_extreme_idx = -1
        last_extreme_type = None  # 上一個極值點類型
        
        # 趨勢結構變數
        trend_start_idx = None  # 當前趨勢開始的索引
        last_swing_high = None  # 最後一個擺動高點
        last_swing_low = None  # 最後一個擺動低點
        trend_swings = []  # 當前趨勢內的擺動點
        
        # 遍歷數據
        for i in range(window, len(self.df)):
            current_price = self.df['Close'].iloc[i]
            current_high = self.df['High'].iloc[i]
            current_low = self.df['Low'].iloc[i]
            threshold = self.df['Turn_Threshold'].iloc[i]

            print(str(current_price)+" : "+str(threshold))            
            
            # 初始化第一個擺動點
            if last_extreme is None:
                # 判斷初始趨勢
                if i > window and self.df['Close'].iloc[i] > self.df['Close'].iloc[i-1]:
                    trend = 'up'
                    last_extreme = current_high
                    last_extreme_idx = i
                    last_extreme_type = 'high'
                    last_swing_high = (i, current_high)
                    swing_points.append((i, last_extreme, 'high'))
                else:
                    trend = 'down'
                    last_extreme = current_low
                    last_extreme_idx = i
                    last_extreme_type = 'low'
                    last_swing_low = (i, current_low)
                    swing_points.append((i, last_extreme, 'low'))
                trend_start_idx = i
                continue
            
            # 檢查是否出現長期趨勢
            is_long_trend = False
            if trend_start_idx is not None and (i - trend_start_idx) >= min_trend_length:
                # 檢查價格變動幅度是否顯著
                start_price = self.df['Close'].iloc[trend_start_idx]
                price_change = abs(current_price - start_price) / start_price
                
                # 如果價格變動超過平均波動率的2倍，認為是顯著趨勢
                avg_volatility = self.df['Turn_Threshold'].iloc[trend_start_idx:i].mean()
                if price_change > avg_volatility * 2:
                    is_long_trend = True
                    
                    # 如果是長期下跌趨勢，重新評估之前的擺動點
                    if trend == 'down' and len(swing_points) >= 4:
                        self._adjust_swings_in_downtrend(swing_points, i, current_price)
            
            # 主趨勢邏輯
            if trend == 'up':
                # 上升趨勢中，更新最高點
                if current_high > last_extreme:
                    last_extreme = current_high
                    last_extreme_idx = i
                    last_extreme_type = 'high'
                    
                    # 更新最後一個擺動點
                    if swing_points[-1][2] == 'high':
                        swing_points[-1] = (i, last_extreme, 'high')
                    else:
                        swing_points.append((i, last_extreme, 'high'))
                        last_swing_high = (i, current_high)
                
                # 檢查是否轉向: 從高點下跌超過閾值
                elif last_extreme - current_low > threshold:
                    # 在長期下跌趨勢中，需要突破前一個LH才確認轉向
                    if is_long_trend and trend == 'down':
                        if last_swing_high is not None and current_high > last_swing_high[1]:
                            # 突破前一個LH，確認轉為上升
                            trend = 'up'
                            last_extreme = current_high
                            last_extreme_idx = i
                            last_extreme_type = 'high'
                            swing_points.append((i, last_extreme, 'high'))
                            last_swing_high = (i, current_high)
                            trend_start_idx = i
                        else:
                            # 繼續下跌趨勢，只記錄LL
                            last_extreme = current_low
                            last_extreme_idx = i
                            last_extreme_type = 'low'
                            swing_points.append((i, last_extreme, 'low'))
                            last_swing_low = (i, current_low)
                    else:
                        # 正常轉向
                        trend = 'down'
                        last_extreme = current_low
                        last_extreme_idx = i
                        last_extreme_type = 'low'
                        swing_points.append((i, last_extreme, 'low'))
                        last_swing_low = (i, current_low)
                        trend_start_idx = i
                        
            else:  # trend == 'down'
                # 下降趨勢中，更新最低點
                if current_low < last_extreme:
                    last_extreme = current_low
                    last_extreme_idx = i
                    last_extreme_type = 'low'
                    
                    # 更新最後一個擺動點
                    if swing_points[-1][2] == 'low':
                        swing_points[-1] = (i, last_extreme, 'low')
                    else:
                        swing_points.append((i, last_extreme, 'low'))
                        last_swing_low = (i, current_low)
                
                # 檢查是否轉向: 從低點上漲超過閾值
                elif current_high - last_extreme > threshold:
                    # 在長期上升趨勢後下跌，需要跌破前一個HL才確認轉向
                    if is_long_trend and trend == 'up':
                        if last_swing_low is not None and current_low < last_swing_low[1]:
                            trend = 'down'
                            last_extreme = current_low
                            last_extreme_idx = i
                            last_extreme_type = 'low'
                            swing_points.append((i, last_extreme, 'low'))
                            last_swing_low = (i, current_low)
                            trend_start_idx = i
                        else:
                            # 繼續上升趨勢，只記錄HH
                            last_extreme = current_high
                            last_extreme_idx = i
                            last_extreme_type = 'high'
                            swing_points.append((i, last_extreme, 'high'))
                            last_swing_high = (i, current_high)
                    else:
                        # 正常轉向
                        trend = 'up'
                        last_extreme = current_high
                        last_extreme_idx = i
                        last_extreme_type = 'high'
                        swing_points.append((i, last_extreme, 'high'))
                        last_swing_high = (i, current_high)
                        trend_start_idx = i
        
        # 後處理：過濾掉不符合趨勢結構的擺動點
        swing_points = self._filter_swing_points_by_structure(swing_points)
        
        self.swing_points = swing_points
        print(f"找到 {len(swing_points)} 個擺動點")
        return swing_points

    def _adjust_swings_in_downtrend(self, swing_points, current_idx, current_price):
        """
        調整下跌趨勢中的擺動點，確保只包含LH和LL
        """
        if len(swing_points) < 4:
            return swing_points
        
        # 找出最近的幾個擺動點
        recent_swings = swing_points[-6:]  # 看最近6個點
        
        # 確保高點遞減，低點遞減
        for j in range(1, len(recent_swings)):
            current_swing = recent_swings[j]
            prev_swing = recent_swings[j-1]
            
            if current_swing[2] == 'high' and prev_swing[2] == 'high':
                # 高點應該遞減
                if current_swing[1] > prev_swing[1]:
                    # 當前高點高於前一個高點，需要調整
                    # 刪除當前高點或前一個高點中較低的那個
                    pass
            
            elif current_swing[2] == 'low' and prev_swing[2] == 'low':
                # 低點應該遞減
                if current_swing[1] > prev_swing[1]:
                    # 當前低點高於前一個低點，需要調整
                    pass
        
        return swing_points

    def _filter_swing_points_by_structure(self, swing_points):
        """
        根據趨勢結構過濾擺動點
        確保在明確趨勢中只保留符合趨勢方向的擺動點
        """
        if len(swing_points) < 3:
            return swing_points
        
        filtered_points = [swing_points[0]]  # 保留第一個點
        
        i = 1
        while i < len(swing_points):
            current_point = swing_points[i]
            prev_point = swing_points[i-1]
            
            # 檢查擺動點序列是否合理
            # 1. 相鄰的點應該是高低交替
            if current_point[2] == prev_point[2]:
                # 如果類型相同，保留價格更極端的那個
                if current_point[2] == 'high':
                    if current_point[1] > prev_point[1]:
                        # 當前高點更高，替換前一個高點
                        filtered_points[-1] = current_point
                    # 否則忽略當前高點
                else:  # low
                    if current_point[1] < prev_point[1]:
                        # 當前低點更低，替換前一個低點
                        filtered_points[-1] = current_point
                    # 否則忽略當前低點
            else:
                # 高低交替，直接加入
                filtered_points.append(current_point)
            
            i += 1
        
        return filtered_points

    def identify_trend_structure(self, swing_points, lookback=10):
        """
        識別趨勢結構
        返回: 'uptrend', 'downtrend', 'consolidation'
        """
        if len(swing_points) < 4:
            return 'consolidation'
        
        # 取最近的幾個擺動點
        recent_swings = swing_points[-lookback:] if len(swing_points) > lookback else swing_points
        
        highs = [p for p in recent_swings if p[2] == 'high']
        lows = [p for p in recent_swings if p[2] == 'low']
        
        if len(highs) >= 2 and len(lows) >= 2:
            # 檢查是否形成高點遞增、低點遞增（上升趨勢）
            highs_sorted = sorted(highs, key=lambda x: x[0])
            lows_sorted = sorted(lows, key=lambda x: x[0])
            
            # 檢查最後兩個高點
            if highs_sorted[-1][1] > highs_sorted[-2][1]:
                # 檢查最後兩個低點
                if lows_sorted[-1][1] > lows_sorted[-2][1]:
                    return 'uptrend'
            
            # 檢查是否形成高點遞減、低點遞減（下跌趨勢）
            if highs_sorted[-1][1] < highs_sorted[-2][1]:
                if lows_sorted[-1][1] < lows_sorted[-2][1]:
                    return 'downtrend'
        
        return 'consolidation'    
    
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
    analyzer = SwingPointAnalyzer(symbol="1209.HK", period="2y", interval="1d")
    analyzer.fetch_data()
    analyzer.calculate_daily_volatility(window=20)
    swing_points = analyzer.find_swing_points(window=20)
    classifications = analyzer.identify_HH_HL_LH_LL()
    

    summary = analyzer.get_summary()
    print("\n分析摘要:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # 輸出詳細的 HH/HL/LH/LL 點
    print("\n詳細的擺動點分類:")
    df_results = pd.DataFrame(classifications)
    print(df_results.to_string(index=False))

    trend_structure = analyzer.identify_trend_structure(swing_points, lookback=10)
    print(f"當前趨勢結構: {trend_structure}")    


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
    