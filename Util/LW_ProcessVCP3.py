import pandas as pd
import numpy as np
import concurrent.futures as cf
import os
from tqdm import tqdm
import time as t
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from LW_Calindicator import *
    from LW_CalHHHL import *
    from LW_BossSkill import *

except ImportError:

    from UTIL.LW_Calindicator import *
    from UTIL.LW_CalHHHL import *
    from UTIL.LW_BossSkill import *    


PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/" 


class DailyStrategyScanner:
    def __init__(self, df):
        """
        初始化掃描器
        
        參數:
        df: 包含以下欄位的DataFrame:
            - Date: 日期
            - Open: 開盤價
            - High: 最高價
            - Low: 最低價
            - Close: 收盤價
            - Volume: 成交量
        """
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
    def calculate_all_indicators(self):
        """計算所有技術指標"""            
        # 1. 基礎移動平均線
        self.df['MA_50'] = self.df['Close'].rolling(window=50, min_periods=1).mean()
        self.df['MA_20'] = self.df['Close'].rolling(window=20, min_periods=1).mean()
        
        # 2. 價格漲幅 (從50天前到現在)
        self.df['Price_Increase_50D'] = ((self.df['Close'] - self.df['Close'].shift(50)) / 
                                         self.df['Close'].shift(50) * 100)
        
        # 3. 計算ADX, +DI, -DI
        self.calculate_adx_daily()
        
        # 4. 計算RSI
        self.calculate_rsi_daily()
        
        # 5. 計算Keltner Channels (用於波動性)
        self.calculate_keltner_channels_daily()
        
        # 6. 計算成交量指標
        self.calculate_volume_indicators_daily()
        
        # 7. 計算異常分數
        self.calculate_anomaly_score_daily()
        
        # 8. 計算支撐阻力位
        self.calculate_support_resistance_daily()
        
        # 9. 計算收縮結構 (需要較複雜的計算)
        self.identify_contractions_daily()
                
        return self.df
    
    def calculate_adx_daily(self, period=14):
        """每日計算ADX, +DI, -DI"""
        # 計算True Range
        df = self.df.copy()
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        
        # 計算方向移動
        df['DMplus'] = np.where(
            (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
            np.maximum(df['High'] - df['High'].shift(1), 0), 0
        )
        df['DMminus'] = np.where(
            (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
            np.maximum(df['Low'].shift(1) - df['Low'], 0), 0
        )
        
        # 平滑計算
        df['TR_smooth'] = df['TR'].rolling(window=period, min_periods=1).mean()
        df['DMplus_smooth'] = df['DMplus'].rolling(window=period, min_periods=1).mean()
        df['DMminus_smooth'] = df['DMminus'].rolling(window=period, min_periods=1).mean()
        
        # 計算方向指標
        df['DIplus'] = 100 * (df['DMplus_smooth'] / df['TR_smooth'])
        df['DIminus'] = 100 * (df['DMminus_smooth'] / df['TR_smooth'])
        
        # 計算ADX
        df['DI_diff'] = abs(df['DIplus'] - df['DIminus'])
        df['DI_sum'] = df['DIplus'] + df['DIminus']
        df['DX'] = 100 * (df['DI_diff'] / df['DI_sum'].replace(0, np.nan))
        df['ADX'] = df['DX'].rolling(window=period, min_periods=1).mean()
        
        # 更新到主數據框
        self.df['ADX'] = df['ADX']
        self.df['DIplus'] = df['DIplus']
        self.df['DIminus'] = df['DIminus']
        
        return self.df
    
    def calculate_rsi_daily(self, period=14):
        """每日計算RSI"""        
        
        delta = self.df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        return self.df
    
    def calculate_keltner_channels_daily(self, period=20, atr_multiplier=2):
        """每日計算Keltner Channels"""        
        
        # 中線: EMA
        self.df['KC_middle'] = self.df['Close'].ewm(span=period, min_periods=1).mean()
        
        # 計算ATR
        high_low = self.df['High'] - self.df['Low']
        high_close = abs(self.df['High'] - self.df['Close'].shift())
        low_close = abs(self.df['Low'] - self.df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period, min_periods=1).mean()
        
        # 上下通道
        self.df['KC_upper'] = self.df['KC_middle'] + atr_multiplier * atr
        self.df['KC_lower'] = self.df['KC_middle'] - atr_multiplier * atr
        
        # 計算KC寬度 (波動性指標)
        self.df['KC_width'] = (self.df['KC_upper'] - self.df['KC_lower']) / self.df['KC_middle'].replace(0, np.nan) * 100
        
        return self.df
    
    def calculate_volume_indicators_daily(self):
        """每日計算成交量指標"""        
        
        # 成交量移動平均
        self.df['Volume_MA_20'] = self.df['Volume'].rolling(window=20, min_periods=1).mean()
        
        # 成交量變化率
        self.df['Volume_Change'] = self.df['Volume'].pct_change()
        
        # 成交量相對於平均的倍數
        self.df['Volume_Ratio'] = self.df['Volume'] / self.df['Volume_MA_20'].replace(0, np.nan)
        
        return self.df
    
    def calculate_anomaly_score_daily(self, window=20):
        """每日計算異常分數"""        
        
        # 計算價格和成交量的標準化z-score
        price_mean = self.df['Close'].rolling(window=window, min_periods=1).mean()
        price_std = self.df['Close'].rolling(window=window, min_periods=1).std()
        self.df['Price_zscore'] = abs((self.df['Close'] - price_mean) / price_std.replace(0, np.nan))
        
        volume_mean = self.df['Volume'].rolling(window=window, min_periods=1).mean()
        volume_std = self.df['Volume'].rolling(window=window, min_periods=1).std()
        self.df['Volume_zscore'] = abs((self.df['Volume'] - volume_mean) / volume_std.replace(0, np.nan))
        
        # 異常分數: 如果價格或成交量的z-score超過3，則標記為異常
        self.df['Anomaly_Score'] = np.where(
            (self.df['Price_zscore'] > 3) | (self.df['Volume_zscore'] > 3), 0, 1
        )
        
        return self.df
    
    def calculate_support_resistance_daily(self, window=20):
        """每日計算支撐阻力位"""        
        
        # 動態阻力計算 (最近20日的最高價，不包括當天)
        self.df['Resistance'] = self.df['High'].rolling(window=window, min_periods=1).max().shift(1)
        
        # 動態支撐計算 (最近20日的最低價，不包括當天)
        self.df['Support'] = self.df['Low'].rolling(window=window, min_periods=1).min().shift(1)
        
        # 當前價格相對於阻力位的距離
        self.df['Resistance_Distance'] = (
            (self.df['Close'] - self.df['Resistance']) / self.df['Resistance'].replace(0, np.nan) * 100
        )
        
        # 突破標誌
        self.df['Breakout'] = self.df['Close'] > self.df['Resistance']
        
        return self.df
    
    def identify_contractions_daily(self, lookback_period=60, contraction_window=20):
        """
        每日識別收縮結構
        
        參數:
        lookback_period: 用於尋找局部高點低點的回顧期間
        contraction_window: 檢查收縮的窗口期
        """        
        
        # 初始化結果列
        self.df['Has_Contractions'] = False
        self.df['Contraction_Count'] = 0
        self.df['Volatility_Decreasing'] = False
        self.df['Volume_Decrease'] = False
        
        # 對於每一行數據（從第lookback_period天開始）
        for i in range(lookback_period, len(self.df)):
            # 獲取過去lookback_period天的數據
            recent_data = self.df.iloc[i-lookback_period:i+1].copy()
            
            # 尋找局部高點和低點
            recent_data['High_Peak'] = recent_data['High'].rolling(window=5, center=True, min_periods=1).max()
            recent_data['Low_Trough'] = recent_data['Low'].rolling(window=5, center=True, min_periods=1).min()
            
            # 識別高點和低點
            highs = recent_data[recent_data['High'] == recent_data['High_Peak']]
            lows = recent_data[recent_data['Low'] == recent_data['Low_Trough']]
            
            # 至少需要3個高點和3個低點來分析收縮
            if len(highs) < 3 or len(lows) < 3:
                continue
            
            # 計算回撤幅度 (從高點到下一個低點)
            retracements = []
            high_prices = []
            low_prices = []
            
            for j in range(min(len(highs), len(lows))-1):
                high_price = highs.iloc[j]['High']
                low_price = lows.iloc[j+1]['Low']
                retracement_pct = (high_price - low_price) / high_price * 100
                retracements.append(retracement_pct)
                high_prices.append(high_price)
                low_prices.append(low_price)
            
            # 檢查收縮條件
            contractions = []
            volatility_decrease = True
            
            for k in range(1, len(retracements)):
                if retracements[k] <= retracements[k-1] * 0.5:  # 每個收縮 ≤ 前一個回撤的50%
                    contractions.append({
                        'retracement': retracements[k],
                        'prev_retracement': retracements[k-1],
                        'contraction_ratio': retracements[k] / retracements[k-1]
                    })
                    
                    # 檢查波動性減少
                    high_idx = highs.index[k]
                    prev_high_idx = highs.index[k-1]
                    
                    # 確保索引在數據範圍內
                    if high_idx in recent_data.index and prev_high_idx in recent_data.index:
                        kc_width_current = recent_data.loc[high_idx, 'KC_width'] if 'KC_width' in recent_data.columns else 0
                        kc_width_prev = recent_data.loc[prev_high_idx, 'KC_width'] if 'KC_width' in recent_data.columns else 0
                        
                        if kc_width_current >= kc_width_prev:
                            volatility_decrease = False
                            break
            
            # 檢查是否有至少2個連續收縮
            has_contractions = len(contractions) >= 2
            
            # 檢查收縮期間成交量減少
            volume_decrease = False
            if has_contractions and len(highs) >= 3:
                # 獲取最近幾個高點期間的成交量
                recent_highs = highs.tail(3)
                if len(recent_highs) >= 3:
                    vol1 = recent_data.loc[recent_highs.index[0], 'Volume'] if recent_highs.index[0] in recent_data.index else 0
                    vol2 = recent_data.loc[recent_highs.index[1], 'Volume'] if recent_highs.index[1] in recent_data.index else 0
                    vol3 = recent_data.loc[recent_highs.index[2], 'Volume'] if recent_highs.index[2] in recent_data.index else 0
                    
                    # 檢查成交量是否遞減
                    if vol1 > vol2 > vol3:
                        volume_decrease = True
            
            # 更新當前行的結果
            self.df.at[i, 'Has_Contractions'] = has_contractions
            self.df.at[i, 'Contraction_Count'] = len(contractions)
            self.df.at[i, 'Volatility_Decreasing'] = volatility_decrease
            self.df.at[i, 'Volume_Decrease'] = volume_decrease
        
        return self.df
    
    def check_breakout_volume(self, window=5):
        """檢查突破點的成交量尖峰"""        
        
        # 計算成交量尖峰
        self.df['Volume_Spike'] = False
        
        for i in range(window, len(self.df)):
            # 如果當前是突破點
            if self.df.at[i, 'Breakout']:
                # 獲取過去window天的平均成交量
                avg_volume = self.df['Volume'].iloc[i-window:i].mean()
                current_volume = self.df.at[i, 'Volume']
                
                # 如果當前成交量是平均的1.5倍以上，則認為有成交量尖峰
                if avg_volume > 0 and current_volume > avg_volume * 1.5:
                    self.df.at[i, 'Volume_Spike'] = True
        
        return self.df
    
    def apply_all_conditions_daily(self):
        """每日應用所有條件並標記Passed_All"""        
        
        # 確保所有指標都已計算
        if 'MA_50' not in self.df.columns:
            self.calculate_all_indicators()
        
        # 檢查突破成交量
        if 'Volume_Spike' not in self.df.columns:
            self.check_breakout_volume()
        
        # 創建條件檢查列
        conditions = []
        
        # 條件1: 上升趨勢需求
        condition1 = (self.df['Price_Increase_50D'] >= 30) & (self.df['Close'] > self.df['MA_50'])
        conditions.append(condition1)
        self.df['Condition1_Uptrend'] = condition1
        
        # 條件2: 收縮結構
        condition2 = self.df['Has_Contractions'] & self.df['Volatility_Decreasing']
        conditions.append(condition2)
        self.df['Condition2_Contraction'] = condition2
        
        # 條件3: 成交量信號
        condition3 = self.df['Volume_Decrease'] & self.df['Volume_Spike']
        conditions.append(condition3)
        self.df['Condition3_Volume'] = condition3
        
        # 條件4: 動量確認
        condition4 = (self.df['ADX'] > 25) & (self.df['DIplus'] > self.df['DIminus']) & (self.df['RSI'] < 70)
        conditions.append(condition4)
        self.df['Condition4_Momentum'] = condition4
        
        # 條件5: 異常過濾
        condition5 = self.df['Anomaly_Score'] == 1
        conditions.append(condition5)
        self.df['Condition5_Anomaly'] = condition5
        
        # 條件6: 突破驗證
        condition6 = self.df['Breakout']
        conditions.append(condition6)
        self.df['Condition6_Breakout'] = condition6
        
        # 所有條件都必須滿足
        all_conditions = pd.Series(True, index=self.df.index)
        for condition in conditions:
            all_conditions = all_conditions & condition
        
        # 添加Passed_All列
        self.df['Passed_All'] = all_conditions
        
        # 計算通過的天數
        passed_count = self.df['Passed_All'].sum()
        total_days = len(self.df)
        passed_percentage = (passed_count / total_days * 100) if total_days > 0 else 0
        
        print(f"完成! 總天數: {total_days}, 通過天數: {passed_count}, 通過率: {passed_percentage:.2f}%")
        
        return self.df
    
    def get_daily_signals(self):
        """獲取每日信號"""
        self.apply_all_conditions_daily()
        return self.df
    
    def get_summary_statistics(self):
        """獲取摘要統計"""
        if 'Passed_All' not in self.df.columns:
            self.apply_all_conditions_daily()
        
        summary = {
            '總天數': len(self.df),
            '通過天數': self.df['Passed_All'].sum(),
            '通過率': self.df['Passed_All'].mean() * 100,
            '首個通過日期': self.df.loc[self.df['Passed_All'], 'Date'].min() if self.df['Passed_All'].any() else None,
            '最後通過日期': self.df.loc[self.df['Passed_All'], 'Date'].max() if self.df['Passed_All'].any() else None,
            '連續通過最長天數': self.get_longest_streak(),
        }
        
        # 各條件通過率
        for col in self.df.columns:
            if col.startswith('Condition'):
                summary[f'{col}_通過率'] = self.df[col].mean() * 100
        
        return summary
    
    def get_longest_streak(self):
        """計算最長的連續通過天數"""
        if 'Passed_All' not in self.df.columns:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for passed in self.df['Passed_All']:
            if passed:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def get_passed_dates(self):
        """獲取所有通過的日期"""
        if 'Passed_All' not in self.df.columns:
            self.apply_all_conditions_daily()
        
        passed_dates = self.df.loc[self.df['Passed_All'], ['Date', 'Close']].copy()
        return passed_dates
    
    def export_results(self, filename='daily_strategy_results.csv'):
        """導出結果到CSV文件"""
        if 'Passed_All' not in self.df.columns:
            self.apply_all_conditions_daily()
        
        # 只導出重要欄位
        export_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Passed_All']
        
        # 添加條件欄位
        condition_columns = [col for col in self.df.columns if col.startswith('Condition')]
        export_columns.extend(condition_columns)
        
        # 添加指標欄位
        indicator_columns = ['MA_50', 'Price_Increase_50D', 'ADX', 'DIplus', 'DIminus', 'RSI', 
                            'Resistance', 'Breakout', 'Volume_Spike']
        export_columns.extend([col for col in indicator_columns if col in self.df.columns])
        
        # 確保所有欄位都存在
        existing_columns = [col for col in export_columns if col in self.df.columns]
        
        export_df = self.df[existing_columns].copy()
        export_df.to_csv(filename, index=False)        
        
        return export_df

def AnalyzeData(sno,stype):
    df = pd.read_csv(PATH+"/"+stype+"/"+sno+".csv") 
    scanner = DailyStrategyScanner(df)
        
    results = scanner.get_daily_signals()
    
    results = results.reset_index()
    results.to_csv(OUTPATH+"/"+stype+"/P_"+sno+".csv",index=False)  



def ProcessVCP(stype):

    snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(PATH+"/"+stype)))
    SLIST = pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST[:]

    with cf.ProcessPoolExecutor(max_workers=5) as executor:
        list(tqdm(executor.map(AnalyzeData,SLIST["sno"],SLIST["stype"],chunksize=1),total=len(SLIST)))


if __name__ == '__main__':
    start = t.perf_counter()

    ProcessVCP("L")
    ProcessVCP("M")
    #ProcessBOSS("S")

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')

