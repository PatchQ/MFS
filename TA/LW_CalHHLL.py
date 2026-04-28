import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 設定中文字型以防繪圖亂碼
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


class SwingPoint:
    """
    Swing Point 類 - 確保經過驗證才能確認
    """
    def __init__(self, idx, price, point_type, confirmed=False):
        self.idx = idx
        self.price = price
        self.point_type = point_type  # 'high' or 'low'
        self.confirmed = confirmed
        self.bar_count = 0  # 形成前的K線數
    
    def confirm(self):
        self.confirmed = True
    
    def to_dict(self):
        return {
            'idx': self.idx,
            'price': self.price,
            'point_type': self.point_type,
            'confirmed': self.confirmed,
            'bar_count': self.bar_count
        }


class HHLL:
    def __init__(self, stockdata, symbol="Stock"):               
        self.df = stockdata.copy()  # 使用 copy 避免 SettingWithCopyWarning
        self.symbol = symbol        # 新增 symbol 屬性供繪圖使用
        self.swing_points = []      # 儲存 SwingPoint 對象
        self.HH_HL_LH_LL = []       # 儲存 HH, HL, LH, LL 點
    
    def calculate_daily_volatility(self, window=20):
        """
        計算每日波幅的 ATR 與動態閾值 (使用向量化運算提升效能)
        
        優化點：
        1. 使用 ATR (True Range) 取代 Daily Range，更準確反映波動率
        2. 使用 Wilder's 平滑 (EWM) 計算 ATR
        3. 根據波動率百分位數動態調整倍數
        """
        if self.df is None or self.df.empty:
            print("請先提供數據")
            return None
        
        # 1. 計算 True Range
        self.df['TR'] = np.maximum(
            self.df['High'] - self.df['Low'],
            np.maximum(
                abs(self.df['High'] - self.df['Close'].shift(1)),
                abs(self.df['Low'] - self.df['Close'].shift(1))
            )
        )
        
        # 2. ATR 使用 Wilder's 平滑 (alpha = 1/window)
        self.df['ATR'] = self.df['TR'].ewm(alpha=1/window, min_periods=window).mean()
        
        # 3. 波動率標準化：用 ATR/Close 百分位數決定倍數
        self.df['Volatility_Pct'] = self.df['ATR'] / self.df['Close']
        
        # 計算滾動75%百分位數
        self.df['Vol_Percentile'] = self.df['Volatility_Pct'].rolling(window=window).quantile(0.75)
        
        # 4. 動態倍數：波動率高時放大閾值
        self.df['Multiplier'] = np.where(
            self.df['Volatility_Pct'] > self.df['Vol_Percentile'],
            2.5,  # 高波動區間
            np.where(self.df['Close'] <= 50, 1.5, 2.0)
        )
        
        self.df['Turn_Threshold'] = self.df['ATR'] * self.df['Multiplier']
        
        # 填補前期的 NaN 值
        self.df['Turn_Threshold'] = self.df['Turn_Threshold'].bfill()
        
        return self.df['Turn_Threshold'].iloc[-1]
    
    def find_swing_points(self, confirm_bars=3):
        """
        利用動態閾值找出 Swing Points (使用狀態機邏輯，確保高低點交替)
        
        優化點：
        1. 使用 N 根K線確認趨勢方向作為初始化
        2. 最後極點需經過確認機制驗證
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
        
        # 初始化第一個尋找方向 - 使用 confirm_bars 根K線確認趨勢
        for i in range(confirm_bars, len(self.df)):
            recent_highs = prices_high[i-confirm_bars:i+1]
            recent_lows = prices_low[i-confirm_bars:i+1]
            
            # 檢查是否形成明確上升趨勢 (每根都高於前一根)
            if all(recent_highs[j] > recent_highs[j-1] for j in range(1, confirm_bars+1)):
                looking_for = 'high'
                extreme_price = prices_high[i]
                extreme_idx = i
                break
            # 檢查是否形成明確下降趨勢 (每根都低於前一根)
            elif all(recent_lows[j] < recent_lows[j-1] for j in range(1, confirm_bars+1)):
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
                # 如果從高點回落超過閾值，確認前一個高點為 Swing High
                elif extreme_price - current_low > threshold:
                    swing_point = SwingPoint(extreme_idx, extreme_price, 'high', confirmed=True)
                    swing_points.append(swing_point)
                    looking_for = 'low'
                    extreme_price = current_low
                    extreme_idx = i
                    
            elif looking_for == 'low':
                # 如果創新低，更新極值
                if current_low < extreme_price:
                    extreme_price = current_low
                    extreme_idx = i
                # 如果從低點反彈超過閾值，確認前一個低點為 Swing Low
                elif current_high - extreme_price > threshold:
                    swing_point = SwingPoint(extreme_idx, extreme_price, 'low', confirmed=True)
                    swing_points.append(swing_point)
                    looking_for = 'high'
                    extreme_price = current_high
                    extreme_idx = i
        
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
        
        for sp in self.swing_points:
            idx = sp.idx
            price = sp.price
            point_type = sp.point_type
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
    
    def get_signals(self):
        """
        輸出市場結構信號 (回測接口)
        
        Returns:
            DataFrame containing trend reversal signals with columns:
            - date: Signal date
            - type: Signal type (BEARISH_REVERSAL, BULLISH_REVERSAL, etc.)
            - strength: Signal strength (STRONG, WEAK)
            - price: Price at signal
        """
        if not self.HH_HL_LH_LL:
            return None
        
        signals = []
        for i in range(1, len(self.HH_HL_LH_LL)):
            prev = self.HH_HL_LH_LL[i-1]
            curr = self.HH_HL_LH_LL[i]
            
            # 趨勢反轉信號
            if prev['Classification'] == 'HH' and curr['Classification'] == 'LL':
                signals.append({
                    'date': curr['Date'],
                    'type': 'BEARISH_REVERSAL',
                    'strength': 'STRONG',
                    'price': curr['Price']
                })
            elif prev['Classification'] == 'LL' and curr['Classification'] == 'HH':
                signals.append({
                    'date': curr['Date'],
                    'type': 'BULLISH_REVERSAL',
                    'strength': 'STRONG',
                    'price': curr['Price']
                })
            # 趨勢延續信號
            elif prev['Classification'] == 'HH' and curr['Classification'] == 'HL':
                signals.append({
                    'date': curr['Date'],
                    'type': 'BULLISH_CONTINUATION',
                    'strength': 'WEAK',
                    'price': curr['Price']
                })
            elif prev['Classification'] == 'LL' and curr['Classification'] == 'LH':
                signals.append({
                    'date': curr['Date'],
                    'type': 'BEARISH_CONTINUATION',
                    'strength': 'WEAK',
                    'price': curr['Price']
                })
            # 突破信號
            elif prev['Classification'] == 'HL' and curr['Classification'] == 'HH':
                signals.append({
                    'date': curr['Date'],
                    'type': 'BULLISH_BREAKOUT',
                    'strength': 'STRONG',
                    'price': curr['Price']
                })
            elif prev['Classification'] == 'LH' and curr['Classification'] == 'LL':
                signals.append({
                    'date': curr['Date'],
                    'type': 'BEARISH_BREAKOUT',
                    'strength': 'STRONG',
                    'price': curr['Price']
                })
        
        return pd.DataFrame(signals)
    
    def visualize_results(self):
        """
        可視化結果 (向量化版本 - 移除 iterrows)
        """
        if self.df is None or not self.HH_HL_LH_LL:
            print("沒有數據可視化")
            return
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # 繪製收盤價折線
        ax.plot(self.df.index, self.df['Close'], label='Close Price', color='gray', alpha=0.5)
        
        # 將 DataFrame 轉換為陣列操作
        points_df = pd.DataFrame(self.HH_HL_LH_LL)
        
        if points_df.empty:
            plt.close()
            return
        
        # 向量化定義繪圖屬性
        colors = points_df['Classification'].map({
            'HH': 'red', 'HL': 'green', 'LH': 'orange', 'LL': 'blue',
            'First_High': 'purple', 'First_Low': 'brown'
        })
        markers = points_df['Classification'].map({
            'HH': '^', 'HL': '^', 'LH': 'v', 'LL': 'v',
            'First_High': 'o', 'First_Low': 'o'
        })
        
        # 向量化繪製散點
        ax.scatter(points_df['Date'], points_df['Price'], 
                   c=colors, marker=markers, s=100, zorder=5)
        
        # 向量化標註文字
        for cls in points_df['Classification'].unique():
            cls_mask = points_df['Classification'] == cls
            cls_dates = points_df.loc[cls_mask, 'Date']
            cls_prices = points_df.loc[cls_mask, 'Price']
            for date, price in zip(cls_dates, cls_prices):
                ax.annotate(cls, (date, price), 
                            textcoords="offset points", xytext=(0, 10), 
                            ha='center', fontsize=9)
        
        # 向量化繪製 ZigZag 連線
        dates = points_df['Date'].values
        prices = points_df['Price'].values
        if len(dates) > 1:
            ax.plot(dates, prices, 'k--', alpha=0.3, zorder=4)
        
        ax.set_title(f'{self.symbol} - 市場結構 (HH/HL/LH/LL)')
        ax.set_xlabel('日期')
        ax.set_ylabel('價格')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def calHHLL(df, confirm_bars=3):
    """
    對外接口函式
    
    Args:
        df: 股票數據 DataFrame，需包含 High, Low, Close 欄位
        confirm_bars: 確認趨勢方向的K線數量 (默認3)
    
    Returns:
        DataFrame containing HH, HL, LH, LL points
    """
    analyzer = HHLL(stockdata=df)
    analyzer.calculate_daily_volatility(window=20)
    analyzer.find_swing_points(confirm_bars=confirm_bars)
    results_df = analyzer.identify_HH_HL_LH_LL()
    return results_df
