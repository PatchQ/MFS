import pandas as pd
import numpy as np
import concurrent.futures as cf
import os
import platform
from tqdm import tqdm
import time as t
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

try:
    from LW_Calindicator import *
except ImportError:
    from TA.LW_Calindicator import *

PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/" 

class VCPScanner:
    def __init__(self, df):
        """
        初始化完整策略掃描器（寬鬆版本）
        
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

        if len(self.df)!=0:
            self.df['Date'] = pd.to_datetime(self.df.index)
            self.df = self.df.sort_values('Date').reset_index(drop=True)
        
    def calculate_adx_di(self, period=14):
        """計算ADX, +DI, -DI"""        
        
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
    
    def calculate_rsi(self, period=14):
        """計算RSI"""        
        
        delta = self.df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        return self.df
    
    def calculate_kc_width(self, period=20, atr_multiplier=2):
        """計算Keltner Channels寬度"""        
        
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
        self.df['KC_Width'] = (self.df['KC_upper'] - self.df['KC_lower']) / self.df['KC_middle'].replace(0, np.nan) * 100
        
        return self.df
    
    def calculate_anomaly_score(self, window=20, threshold=4):
        """計算異常分數（寬鬆版本：z-score閾值從3提高到4）"""        
        
        # 計算價格和成交量的標準化z-score
        price_mean = self.df['Close'].rolling(window=window, min_periods=1).mean()
        price_std = self.df['Close'].rolling(window=window, min_periods=1).std()
        self.df['Price_zscore'] = abs((self.df['Close'] - price_mean) / price_std.replace(0, np.nan))
        
        volume_mean = self.df['Volume'].rolling(window=window, min_periods=1).mean()
        volume_std = self.df['Volume'].rolling(window=window, min_periods=1).std()
        self.df['Volume_zscore'] = abs((self.df['Volume'] - volume_mean) / volume_std.replace(0, np.nan))
        
        # 異常分數: 如果價格或成交量的z-score超過threshold，則標記為異常
        # 寬鬆版本：使用更高的閾值（4而不是3）
        self.df['Anomaly_Score'] = np.where(
            (self.df['Price_zscore'] > threshold) | (self.df['Volume_zscore'] > threshold), 0, 1
        )
        
        return self.df
    
    def calculate_moving_averages(self):
        """計算移動平均線"""        
        
        self.df['MA_50'] = self.df['Close'].rolling(window=50, min_periods=1).mean()
        self.df['MA_20'] = self.df['Close'].rolling(window=20, min_periods=1).mean()
        
        # 計算價格高於50日MA
        self.df['Price_Above_MA50'] = self.df['Close'] > self.df['MA_50']
        
        # 計算50天價格漲幅（寬鬆版本：從30%降低到20%）
        self.df['Price_Increase_50D'] = ((self.df['Close'] - self.df['Close'].shift(50)) / 
                                         self.df['Close'].shift(50).replace(0, np.nan) * 100)
        
        # 添加30天價格漲幅作為額外參考
        self.df['Price_Increase_30D'] = ((self.df['Close'] - self.df['Close'].shift(30)) / 
                                         self.df['Close'].shift(30).replace(0, np.nan) * 100)
        
        return self.df
    
    def calculate_resistance_levels(self, window=20):
        """計算阻力位（寬鬆版本：允許最近5天內的最高價作為阻力）"""        
        
        # 計算滾動阻力 (過去window天最高價，不包括當天)
        self.df['Resistance'] = self.df['High'].rolling(window=window, min_periods=1).max().shift(1)
        
        # 計算是否突破阻力位（寬鬆版本：允許1%以內的突破）
        resistance_with_buffer = self.df['Resistance'] * 1.01  # 允許1%緩衝
        self.df['Above_Resistance'] = self.df['Close'] > resistance_with_buffer
        
        # 計算突破幅度
        self.df['Breakout_Percentage'] = ((self.df['Close'] - self.df['Resistance']) / 
                                          self.df['Resistance'].replace(0, np.nan) * 100)
        
        return self.df
    
    def find_contractions_for_day(self, df_slice, current_index, lookback_days=200):
        """
        針對特定日期尋找收縮結構（寬鬆版本）
        
        主要放寬的條件：
        1. 收縮判定閾值從0.6提高到0.7（更容易識別為收縮）
        2. 收縮比例從0.5提高到0.6（允許更大的回撤）
        3. KC寬度條件從嚴格小於放寬為小於等於
        """
        # 確保有足夠的數據
        if current_index < 40:  # 從40天開始而不是50天
            return {
                'has_contractions': False,
                'contraction_count': 0,
                'contractions': [],
                'valid_contractions': False,
                'kc_contraction': False,
                'volume_contraction': False,
                'volume_breakout_spike': False
            }
        
        # 獲取回顧期間的數據
        start_idx = max(0, current_index - lookback_days)
        window_data = df_slice.iloc[start_idx:current_index+1].copy()
        
        # 重置索引以便使用位置索引
        window_data = window_data.reset_index(drop=True)
        current_idx_in_window = len(window_data) - 1
        
        # 計算近期成交量平均值 (過去15天而不是10天，更平滑)
        recent_volume = 0
        lookback_volume = 15  # 從10天增加到15天
        if current_idx_in_window >= lookback_volume:
            recent_volume = window_data['Volume'].iloc[current_idx_in_window-lookback_volume+1:current_idx_in_window+1].mean()
        elif current_idx_in_window > 0:
            recent_volume = window_data['Volume'].iloc[:current_idx_in_window+1].mean()
        
        # 初始化變量
        contractions = []
        closes = window_data['Close'].values
        i = current_idx_in_window
        contraction_count = 0
        
        # 向前尋找收縮（寬鬆版本：更容易識別收縮）
        while i > 0 and contraction_count < 8:  # 從6個增加到8個
            # 放寬收縮識別條件：允許更小的跌幅
            if closes[i] < closes[i-1] * 0.995:  # 允許0.5%以內的下跌
                start = i
                # 放寬連續下跌的識別條件
                while i > 0 and closes[i] < closes[i-1] * 0.998:  # 允許0.2%以內的下跌
                    i -= 1
                end = i
                
                # 確保有有效的區間
                if start > end:
                    # 計算回撤幅度
                    high = window_data['High'].iloc[end:start+1].max()
                    low = window_data['Low'].iloc[end:start+1].min()
                    
                    # 計算該收縮期間的平均成交量
                    contraction_volume = window_data['Volume'].iloc[end:start+1].mean()
                    
                    # 避免除以零
                    if high > 0:
                        retracement = (high - low) / high
                        
                        # 只考慮回撤幅度大於2%的收縮（避免噪音）
                        if retracement < 0.02:  # 小於2%的回撤忽略
                            i -= 1
                            continue
                        
                        # 計算平均KC寬度
                        kc_width = window_data['KC_Width'].iloc[end:start+1].mean()
                        
                        # 檢查是否符合收縮條件（寬鬆版本：閾值從0.6提高到0.7）
                        if contractions:
                            # 每個收縮 ≤ 前一個回撤的60%（原為50%）
                            if retracement > contractions[-1]['retracement'] * 0.7:  # 從0.6提高到0.7
                                break
                        
                        contractions.append({
                            'retracement': retracement,
                            'kc_width': kc_width,
                            'avg_volume': contraction_volume,
                            'start_date': window_data['Date'].iloc[end],
                            'end_date': window_data['Date'].iloc[start]
                        })
                        
                        contraction_count += 1
            
            i -= 1
        
        # 分析收縮結構
        has_contractions = len(contractions) >= 2
        
        # 檢查收縮是否連續且符合條件（寬鬆版本）
        valid_contractions = False
        kc_contraction = False
        volume_contraction = False
        
        if len(contractions) >= 2:
            # 檢查每個收縮 ≤ 前一個回撤的60%（原為50%）
            valid_contractions = all(
                contractions[i]['retracement'] <= contractions[i-1]['retracement'] * 0.6  # 從0.5提高到0.6
                for i in range(1, len(contractions))
            )
            
            # 檢查波動率是否隨每次收縮而降低（寬鬆版本：允許持平）
            kc_contraction = all(
                contractions[i]['kc_width'] <= contractions[i-1]['kc_width'] * 1.05  # 允許5%以內的增加
                for i in range(1, len(contractions))
            )
            
            # 檢查成交量收縮條件（寬鬆版本）
            if len(contractions) >= 2:
                # 取最近的兩個收縮的成交量平均值
                contraction_volumes = [c['avg_volume'] for c in contractions[:2]]
                avg_contraction_volume = np.mean(contraction_volumes) if contraction_volumes else 0
                
                # 檢查成交量收縮: 收縮期間的平均成交量 < 近期平均成交量的80%（原為70%）
                if recent_volume > 0:
                    volume_contraction = avg_contraction_volume < recent_volume * 0.8  # 從0.7提高到0.8
        
        # 檢查突破點的成交量尖峰（寬鬆版本）
        volume_breakout_spike = False
        if current_idx_in_window > 0:
            current_volume = window_data['Volume'].iloc[current_idx_in_window]
            volume_ma_20 = window_data['Volume'].rolling(window=20, min_periods=1).mean().iloc[current_idx_in_window]
            
            # 成交量尖峰: 當日成交量 > 20日平均成交量的1.3倍（原為1.5倍）
            if volume_ma_20 > 0:
                volume_breakout_spike = current_volume > volume_ma_20 * 1.3  # 從1.5降低到1.3
        
        return {
            'has_contractions': has_contractions,
            'contraction_count': len(contractions),
            'contractions': contractions,
            'valid_contractions': valid_contractions,
            'kc_contraction': kc_contraction,
            'volume_contraction': volume_contraction,
            'volume_breakout_spike': volume_breakout_spike,
            'recent_volume': recent_volume,
            'avg_contraction_volume': avg_contraction_volume if 'avg_contraction_volume' in locals() else 0
        }
    
    def calculate_volatility_decrease(self):
        """計算波動率下降指標（寬鬆版本）"""        
        
        self.df['Volatility_Decrease'] = 'N/A'
        self.df['Volatility_Decrease_Pct'] = 0.0
        
        for i in range(50, len(self.df)):  # 從60天減少到50天
            if i >= 50:
                # 計算過去50-20天的平均KC寬度（原為60-30天）
                kc_width_older = self.df['KC_Width'].iloc[i-50:i-20].mean()
                
                # 計算最近10天的平均KC寬度
                kc_width_recent = self.df['KC_Width'].iloc[i-10:i].mean()
                
                # 避免除以零
                if kc_width_recent > 0:
                    volatility_pct = (kc_width_older / kc_width_recent - 1) * 100
                    
                    # 格式化為百分比字符串
                    self.df.at[i, 'Volatility_Decrease'] = f"{volatility_pct:.1f}%"
                    self.df.at[i, 'Volatility_Decrease_Pct'] = volatility_pct
                else:
                    self.df.at[i, 'Volatility_Decrease'] = 'N/A'
                    self.df.at[i, 'Volatility_Decrease_Pct'] = 0.0
        
        return self.df
    
    def calculate_daily_conditions(self, lookback_days=200):
        """
        為每一天計算所有條件（寬鬆版本）
        
        主要放寬的條件：
        1. 上升趨勢需求：價格漲幅從30%降低到20%
        2. ADX強度：從25降低到20
        3. RSI條件：從70提高到75
        4. 突破條件：允許1%緩衝區
        """        
        
        # 確保所有基礎指標已計算
        if 'ADX' not in self.df.columns:
            self.calculate_adx_di()
        if 'RSI' not in self.df.columns:
            self.calculate_rsi()
        if 'KC_Width' not in self.df.columns:
            self.calculate_kc_width()
        if 'Anomaly_Score' not in self.df.columns:
            self.calculate_anomaly_score()
        if 'MA_50' not in self.df.columns:
            self.calculate_moving_averages()
        if 'Resistance' not in self.df.columns:
            self.calculate_resistance_levels()
        
        # 計算波動率下降指標
        self.calculate_volatility_decrease()
        
        # 初始化結果列
        self.df['Contraction_Structure'] = False
        self.df['Contraction_Count'] = 0
        self.df['Valid_Contractions'] = False
        self.df['KC_Contraction'] = False
        self.df['Volume_Contraction'] = False
        self.df['Volume_Spike'] = False
        self.df['Breakout_Detected'] = False
        
        # 技術指標條件（寬鬆版本）
        self.df['ADX_Strength'] = False
        self.df['DI_Bullish'] = False
        self.df['RSI_Value'] = 0.0
        self.df['Anomaly_Free'] = False
        
        # 為每一天計算條件
        total_days = len(self.df)
        
        for i in range(50, total_days):  # 從第50天開始而不是60天
            # 獲取截至當前日期的數據
            df_until_today = self.df.iloc[:i+1].copy()
            
            # 尋找收縮結構和成交量分析
            contraction_results = self.find_contractions_for_day(df_until_today, i, lookback_days)
            
            # 更新收縮結構相關欄位
            self.df.at[i, 'Contraction_Structure'] = contraction_results['has_contractions']
            self.df.at[i, 'Contraction_Count'] = contraction_results['contraction_count']
            self.df.at[i, 'Valid_Contractions'] = contraction_results['valid_contractions']
            self.df.at[i, 'KC_Contraction'] = contraction_results['kc_contraction']
            self.df.at[i, 'Volume_Contraction'] = contraction_results['volume_contraction']
            self.df.at[i, 'Volume_Spike'] = contraction_results['volume_breakout_spike']
            
            # 更新技術指標條件（寬鬆版本）
            # ADX強度：從25降低到20
            self.df.at[i, 'ADX_Strength'] = self.df.at[i, 'ADX'] > 20 if pd.notna(self.df.at[i, 'ADX']) else False
            
            # DI看漲條件：允許DIplus稍微領先即可
            di_plus = self.df.at[i, 'DIplus'] if pd.notna(self.df.at[i, 'DIplus']) else 0
            di_minus = self.df.at[i, 'DIminus'] if pd.notna(self.df.at[i, 'DIminus']) else 0
            self.df.at[i, 'DI_Bullish'] = di_plus > di_minus * 0.95  # 允許DIplus略低於DIminus
            
            self.df.at[i, 'RSI_Value'] = round(self.df.at[i, 'RSI'], 1) if pd.notna(self.df.at[i, 'RSI']) else 0.0
            self.df.at[i, 'Anomaly_Free'] = self.df.at[i, 'Anomaly_Score'] == 1 if pd.notna(self.df.at[i, 'Anomaly_Score']) else False
            
            # 計算突破檢測（寬鬆版本）
            resistance = self.df.at[i, 'Resistance'] if pd.notna(self.df.at[i, 'Resistance']) else 0
            current_close = self.df.at[i, 'Close']
            volume_spike = contraction_results['volume_breakout_spike']
            
            # 突破條件: 收盤價高於阻力位（允許1%緩衝）且有成交量尖峰
            # 或者只要有強烈的成交量尖峰，即使突破不明顯
            breakout_threshold = resistance * 0.99  # 允許1%以下的差距
            #strong_volume_spike = current_volume > volume_ma_20 * 1.8 if 'volume_ma_20' in locals() else False
            
            self.df.at[i, 'Breakout_Detected'] = ((current_close > breakout_threshold) and volume_spike)            
                   
        return self.df
    

def checkVCP(df):

    conditions = []        
    # df = pd.read_csv(PATH+"/"+stype+"/"+sno+".csv")         
    # df = convertData(df)

    scanner = VCPScanner(df)        
    df = scanner.calculate_daily_conditions(lookback_days=250)

    # 1. Uptrend Requirement:
    #30%+ price increase  價格上漲20%以上
    #Price above 50-day MA  價格高於 50 日均線    
    condition1 = (df['Price_Increase_50D'] >= 20) & (df['Price_Above_MA50'])
    conditions.append(condition1)
    df['Uptrend_Requirement'] = condition1
    
    # 2. Contraction Structure: 收縮結構
    #At least 2 successive contractions
    #Each contraction ≤ 50% of previous retracement每次收縮幅度≤前一次回檔幅度的50%
    #Volatility (KC Width) must decrease with each contraction波動率（KC 寬度）必須隨著每次收縮而降低。            
         
    # 收縮結構 (至少有2個連續收縮且符合所有收縮條件)
    condition2 = df['Contraction_Structure'] & df['Valid_Contractions'] & df['KC_Contraction']
    conditions.append(condition2)
    df['Contraction_Pattern'] = condition2
    
    # 成交量信號 (成交量收縮和突破成交量尖峰)
    condition3 = df['Volume_Contraction'] | df['Volume_Spike']
    conditions.append(condition3)
    df['Volume_Signature'] = condition3
    
    # 動量確認 (ADX > 20, +DI > -DI, RSI < 75)
    condition4 = df['ADX_Strength'] & df['DI_Bullish'] & (df['RSI_Value'] < 75)
    conditions.append(condition4)
    df['Momentum_Confirmation'] = condition4
    
    # 突破驗證
    condition5 = df['Breakout_Detected']
    conditions.append(condition5)

    # 異常過濾
    condition6 = df['Anomaly_Free']
    conditions.append(condition6)
    
    # 所有條件都必須滿足
    all_conditions = pd.Series(True, index=df.index)
    for condition in conditions:
        all_conditions = all_conditions & condition

    # Final decision    
    df['VCP'] = all_conditions    

    if len(df)!=0:
        df = df.set_index('Date')
        df.index.name = 'index'    
    # df.to_csv(OUTPATH+"/"+stype+"/P_"+sno+".csv")

    return df





def ProcessVCP(stype):

    snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(PATH+"/"+stype)))
    SLIST = pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST[:]

    if platform.system()=="Windows":
        executor = cf.ProcessPoolExecutor(max_workers=5)
    elif platform.system()=="Darwin":
        executor = cf.ThreadPoolExecutor(max_workers=4)

    with executor:
        list(tqdm(executor.map(checkVCP,SLIST["sno"],SLIST["stype"],chunksize=1),total=len(SLIST)))


if __name__ == '__main__':
    start = t.perf_counter()

    ProcessVCP("L")    
    ProcessVCP("M")
    #ProcessBOSS("S")

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
        