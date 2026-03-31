import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 設定中文字型以防繪圖亂碼
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

class HHLL:
    def __init__(self, stockdata, symbol="Stock"):               
        self.df = stockdata.copy() # 使用 copy 避免 SettingWithCopyWarning
        self.symbol = symbol       # 新增 symbol 屬性供繪圖使用
        self.swing_points = []     # 儲存擺動點 (index, price, type)
        self.HH_HL_LH_LL = []      # 儲存 HH, HL, LH, LL 點
    
    def calculate_daily_volatility(self, window=20):
        """
        計算每日波幅的平均值與轉向閾值 (使用向量化運算提升效能)
        """
        if self.df is None or self.df.empty:
            print("請先提供數據")
            return None
        
        # 1. 計算每日波幅
        self.df['Daily_Range'] = self.df['High'] - self.df['Low']
        
        # 2. 穩健平均：使用滾動中位數 (Rolling Median) 來自然排除極端異常值，效能遠勝 for 迴圈
        self.df['Avg_Range'] = self.df['Daily_Range'].rolling(window=window).median()
        
        # 3. 向量化計算轉向閾值 (np.where 效能遠高於 apply)
        # 邏輯: 收盤價 <= 50 則 threshold = Avg_Range * 1.5，否則為 Avg_Range * 2.0
        self.df['Turn_Threshold'] = np.where(
            self.df['Close'] <= 50,
            self.df['Avg_Range'] * 1.5,
            self.df['Avg_Range'] * 2.0
        )
        
        # 填補前期的 NaN 值，避免無法計算
        self.df['Turn_Threshold'] = self.df['Turn_Threshold'].bfill()
        
        return self.df['Turn_Threshold'].iloc[-1]
    
    def find_swing_points(self):
        """
        利用動態閾值找出擺動點 (使用狀態機邏輯，確保高低點交替)
        """
        if 'Turn_Threshold' not in self.df.columns:
            self.calculate_daily_volatility()
            
        swing_points = []
        
        # 狀態變數
        looking_for = None  # 'high' 或 'low'
        extreme_price = None
        extreme_idx = None
        
        prices_high = self.df['High'].values
        prices_low = self.df['Low'].values
        thresholds = self.df['Turn_Threshold'].values
        
        # 初始化第一個尋找方向
        for i in range(1, len(self.df)):
            if prices_high[i] > prices_high[i-1]:
                looking_for = 'high'
                extreme_price = prices_high[i]
                extreme_idx = i
                break
            elif prices_low[i] < prices_low[i-1]:
                looking_for = 'low'
                extreme_price = prices_low[i]
                extreme_idx = i
                break
                
        if looking_for is None:
            return []

        # 遍歷數據尋找轉折點
        for i in range(extreme_idx + 1, len(self.df)):
            current_high = prices_high[i]
            current_low = prices_low[i]
            threshold = thresholds[i]
            
            if looking_for == 'high':
                # 如果創新高，更新極值
                if current_high > extreme_price:
                    extreme_price = current_high
                    extreme_idx = i
                # 如果從高點回落超過閾值，確認前一個高點為 Swing High，並開始找 Swing Low
                elif extreme_price - current_low > threshold:
                    swing_points.append((extreme_idx, extreme_price, 'high'))
                    looking_for = 'low'
                    extreme_price = current_low
                    extreme_idx = i
                    
            elif looking_for == 'low':
                # 如果創新低，更新極值
                if current_low < extreme_price:
                    extreme_price = current_low
                    extreme_idx = i
                # 如果從低點反彈超過閾值，確認前一個低點為 Swing Low，並開始找 Swing High
                elif current_high - extreme_price > threshold:
                    swing_points.append((extreme_idx, extreme_price, 'low'))
                    looking_for = 'high'
                    extreme_price = current_high
                    extreme_idx = i
                    
        # 將最後一個未確認的極點也加入（可選，視策略而定）
        if extreme_idx is not None:
            swing_points.append((extreme_idx, extreme_price, looking_for))
            
        self.swing_points = swing_points
        return swing_points

    def identify_HH_HL_LH_LL(self):
        """
        識別 HH, HL, LH, LL 點
        """
        if not self.swing_points:
            return []
        
        HH_HL_LH_LL = []
        last_high = None
        last_low = None
        
        for idx, price, point_type in self.swing_points:
            classification = None
            
            if point_type == 'high':
                if last_high is not None:
                    classification = 'HH' if price > last_high else 'LH'
                else:
                    classification = 'First_High'
                last_high = price
                
            else:  # point_type == 'low'
                if last_low is not None:
                    classification = 'HL' if price > last_low else 'LL'
                else:
                    classification = 'First_Low'
                last_low = price
            
            HH_HL_LH_LL.append({
                'Index': idx,
                'Date': self.df.index[idx],
                'Price': price,
                'Close': self.df['Close'].iloc[idx],
                'Type': point_type,
                'Classification': classification
            })
        
        self.HH_HL_LH_LL = HH_HL_LH_LL
        return pd.DataFrame(HH_HL_LH_LL)
    
    def visualize_results(self):
        """可視化結果"""
        if self.df is None or not self.HH_HL_LH_LL:
            print("沒有數據可視化")
            return
            
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # 繪製收盤價折線
        ax.plot(self.df.index, self.df['Close'], label='Close Price', color='gray', alpha=0.5)
        
        # 將 DataFrame 轉換為方便迴圈操作的結構
        points_df = pd.DataFrame(self.HH_HL_LH_LL)
        
        # 定義繪圖屬性
        styles = {
            'HH': {'color': 'red', 'marker': '^'},
            'HL': {'color': 'green', 'marker': '^'},
            'LH': {'color': 'orange', 'marker': 'v'},
            'LL': {'color': 'blue', 'marker': 'v'},
            'First_High': {'color': 'purple', 'marker': 'o'},
            'First_Low': {'color': 'brown', 'marker': 'o'}
        }
        
        # 繪製散點與連線
        prev_date, prev_price = None, None
        for _, row in points_df.iterrows():
            cls = row['Classification']
            if cls in styles:
                ax.scatter(row['Date'], row['Price'], 
                           color=styles[cls]['color'], marker=styles[cls]['marker'], 
                           s=100, zorder=5)
                # 標註文字
                ax.annotate(cls, (row['Date'], row['Price']), 
                            textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
                
                # 畫出擺動連線 (ZigZag線)
                if prev_date is not None:
                    ax.plot([prev_date, row['Date']], [prev_price, row['Price']], 'k--', alpha=0.3)
                prev_date, prev_price = row['Date'], row['Price']
                
        ax.set_title(f'{self.symbol} - 市場結構 (HH/HL/LH/LL)')
        ax.set_xlabel('日期')
        ax.set_ylabel('價格')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def calHHLL(df):
    """
    對外接口函式
    """
    analyzer = HHLL(stockdata=df)
    analyzer.calculate_daily_volatility(window=20)
    analyzer.find_swing_points()
    results_df = analyzer.identify_HH_HL_LH_LL()
    return results_df
