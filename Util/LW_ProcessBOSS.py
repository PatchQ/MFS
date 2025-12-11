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

class RobustSwingPointAnalyzer:
    def __init__(self):
        self.results = {}
    
    def multi_window_swing_detection(self, high_series, low_series, close_series, 
                                   windows=[5, 7, 10], consensus_threshold=0.6):
        """
        多窗口摆动点检测，通过共识机制提高准确性
        
        参数:
        high_series: 最高价序列
        low_series: 最低价序列  
        close_series: 收盘价序列
        windows: 多个窗口大小
        consensus_threshold: 共识阈值 (0-1)，表示需要多少个窗口同意
        
        返回:
        consensus_highs: 共识高点
        consensus_lows: 共识低点
        """
        all_highs = []
        all_lows = []
        
        # 使用不同窗口大小检测摆动点
        for window in windows:
            highs, lows = self.find_swing_points_scipy(high_series, low_series, window)
            all_highs.extend([(idx, price, window) for idx, price in highs])
            all_lows.extend([(idx, price, window) for idx, price in lows])
        
        # 基于共识过滤摆动点
        consensus_highs = self.filter_by_consensus(all_highs, len(windows), consensus_threshold)
        consensus_lows = self.filter_by_consensus(all_lows, len(windows), consensus_threshold)
        
        return consensus_highs, consensus_lows
    
    def find_swing_points_scipy(self, high_series, low_series, window=5):
        """
        使用scipy的信号处理函数识别摆动点（更准确）
        """
        # 识别局部高点和低点
        high_indices = argrelextrema(high_series.values, np.greater, order=window)[0]
        low_indices = argrelextrema(low_series.values, np.less, order=window)[0]
        
        highs = [(idx, high_series.iloc[idx]) for idx in high_indices]
        lows = [(idx, low_series.iloc[idx]) for idx in low_indices]
        
        return highs, lows
    
    def filter_by_consensus(self, points, total_windows, threshold):
        """
        基于共识机制过滤摆动点
        """
        if not points:
            return []
        
        # 按索引分组
        index_groups = {}
        for idx, price, window in points:
            if idx not in index_groups:
                index_groups[idx] = []
            index_groups[idx].append((price, window))
        
        # 计算每个索引的共识度
        consensus_points = []
        for idx, price_windows in index_groups.items():
            consensus_ratio = len(price_windows) / total_windows
            
            if consensus_ratio >= threshold:
                # 取平均价格作为共识价格
                avg_price = np.mean([price for price, _ in price_windows])
                consensus_points.append((idx, avg_price))
        
        return consensus_points
    
    def cluster_nearby_points(self, points, time_threshold=3, price_threshold=0.02):
        """
        使用聚类算法合并接近的摆动点
        """
        if len(points) < 2:
            return points
        
        # 准备聚类数据 [索引, 价格]
        X = np.array([[idx, price] for idx, price in points])
        
        # 使用DBSCAN聚类，考虑时间和价格距离
        # 调整eps参数来控制聚类的紧密程度
        eps_time = time_threshold
        eps_price = np.std(X[:, 1]) * price_threshold
        
        # 由于时间和价格单位不同，需要标准化或调整权重
        # 这里简单处理，主要基于时间聚类
        clustering = DBSCAN(eps=eps_time, min_samples=1).fit(X[:, 0].reshape(-1, 1))
        
        clustered_points = []
        for cluster_id in set(clustering.labels_):
            cluster_points = X[clustering.labels_ == cluster_id]
            
            if len(cluster_points) > 0:
                # 取时间居中的点，价格取最高/最低（取决于后续分类）
                center_idx = int(np.median(cluster_points[:, 0]))
                # 对于高点取最高价，对于低点取最低价
                extreme_price = np.max(cluster_points[:, 1])  # 后续会根据类型调整
                clustered_points.append((center_idx, extreme_price))
        
        return clustered_points
    
    def adaptive_swing_detection(self, base_window, high_series, low_series, close_series, 
                               volatility_lookback=22, adaptive_factor=1.5):
        """
        自适应摆动点检测，根据市场波动率调整窗口大小
        """
        # 计算波动率
        volatility = close_series.pct_change().rolling(volatility_lookback).std()
        
        # 根据波动率动态调整窗口大小        
        adaptive_window = (base_window * (1 + adaptive_factor * volatility)).fillna(base_window)
        adaptive_window = adaptive_window.astype(int).clip(2, 10)  # 限制窗口范围
                
        # 使用自适应窗口检测摆动点
        all_highs = []
        all_lows = []
        
        for i in range(len(high_series)):
            if i < max(adaptive_window):
                continue
                
            current_window = adaptive_window.iloc[i]            
            
            # 检查是否为局部高点
            if (high_series.iloc[i] == high_series.iloc[i-current_window:i+1].max() and
                high_series.iloc[i] == high_series.iloc[i:i+current_window+1].max()):
                all_highs.append((i, high_series.iloc[i]))
            
            # 检查是否为局部低点
            if (low_series.iloc[i] == low_series.iloc[i-current_window:i+1].min() and
                low_series.iloc[i] == low_series.iloc[i:i+current_window+1].min()):
                all_lows.append((i, low_series.iloc[i]))
        
        return all_highs, all_lows
    
    def classify_swing_points_robust(self, high_points, low_points, close_series, 
                                   price_tolerance=0.001, volume_confirmation=True, 
                                   volume_series=None, volume_threshold=1.2):
        """
        稳健的摆动点分类，结合多种确认因素
        """
        # 合并所有点并按时间排序
        all_points = []
        for idx, price in high_points:
            all_points.append({'index': idx, 'price': price, 'type': 'high', 'close':close_series.iloc[idx], 
                              'date': close_series.index[idx] if idx < len(close_series.index) else None})
        
        for idx, price in low_points:
            all_points.append({'index': idx, 'price': price, 'type': 'low', 'close':close_series.iloc[idx], 
                              'date': close_series.index[idx] if idx < len(close_series.index) else None})
        
        # 按索引排序
        all_points.sort(key=lambda x: x['index'])
        
        # 分离高点和低点序列
        high_sequence = [p for p in all_points if p['type'] == 'high']
        low_sequence = [p for p in all_points if p['type'] == 'low']
        
        # 分类高点序列
        for i, point in enumerate(high_sequence):
            if i == 0:
                point['classification'] = 'Start_H'
                continue
            
            prev_point = high_sequence[i-1]
            current_price = point['price']
            prev_price = prev_point['price']
            
            # 计算价格变化
            price_change = (current_price - prev_price) / prev_price
            
            # 考虑成交量确认
            volume_confirm = True
            if volume_confirmation and volume_series is not None:
                current_volume = volume_series.iloc[point['index']] if point['index'] < len(volume_series) else 0
                avg_volume = volume_series.rolling(20).mean().iloc[point['index']] if point['index'] >= 20 else current_volume
                volume_confirm = current_volume > avg_volume * volume_threshold
            
            if abs(price_change) <= price_tolerance:
                point['classification'] = '-H'
            elif price_change > 0 and volume_confirm:
                point['classification'] = 'HH'
            elif price_change > 0:
                point['classification'] = 'HH'  # 弱确认HH
            else:
                point['classification'] = 'LH'
        
        # 分类低点序列
        for i, point in enumerate(low_sequence):
            if i == 0:
                point['classification'] = 'Start_L'
                continue
            
            prev_point = low_sequence[i-1]
            current_price = point['price']
            prev_price = prev_point['price']
            
            # 计算价格变化
            price_change = (current_price - prev_price) / prev_price
            
            # 考虑成交量确认
            volume_confirm = True
            if volume_confirmation and volume_series is not None:
                current_volume = volume_series.iloc[point['index']] if point['index'] < len(volume_series) else 0
                avg_volume = volume_series.rolling(20).mean().iloc[point['index']] if point['index'] >= 20 else current_volume
                volume_confirm = current_volume > avg_volume * volume_threshold
            
            if abs(price_change) <= price_tolerance:
                point['classification'] = '-L'
            elif price_change < 0 and volume_confirm:
                point['classification'] = 'LL'
            elif price_change < 0:
                point['classification'] = 'LL'  # 弱确认LL
            else:
                point['classification'] = 'HL'
        
        # 合并结果
        classified_points = high_sequence + low_sequence
        classified_points.sort(key=lambda x: x['index'])
        
        return pd.DataFrame(classified_points)
    
    def trend_confirmation(self, swing_df, close_series, ma_short=22, ma_long=50):
        """
        使用移动平均线确认趋势方向
        """        
        if swing_df.empty:
            return swing_df
        
        # 计算移动平均线
        ma_short_series = close_series.rolling(ma_short).mean()
        ma_long_series = close_series.rolling(ma_long).mean()
        
        # 为每个摆动点添加趋势确认
        for i, row in swing_df.iterrows():
            idx = row['index']
            if idx < len(close_series):
                # 确定趋势方向
                if ma_short_series.iloc[idx] > ma_long_series.iloc[idx]:
                    trend = 'uptrend'
                else:
                    trend = 'downtrend'
                
                # 根据趋势确认摆动点分类
                current_class = row['classification']
                if trend == 'uptrend' and current_class in ['HH', 'HL']:
                    swing_df.at[i, 'trend_confirmed'] = True
                elif trend == 'downtrend' and current_class in ['LL', 'LH']:
                    swing_df.at[i, 'trend_confirmed'] = True
                else:
                    swing_df.at[i, 'trend_confirmed'] = False
            else:
                swing_df.at[i, 'trend_confirmed'] = False
        
        return swing_df
    
    def fibonacci_retracement_confirmation(self, swing_df, retracement_levels=[0.5, 0.7]):
        """
        使用斐波那契回撤确认摆动点的重要性 retracement_levels=[0.382, 0.5, 0.618, 0.705, 0.786]
        """
        if len(swing_df) < 2:
            return swing_df
        
        # 识别主要摆动点
        major_swings = swing_df[swing_df['classification'].isin(['HH', 'LL'])]
        
        if len(major_swings) < 2:
            return swing_df
        
        # 计算斐波那契回撤水平
        for i in range(1, len(major_swings)):
            prev_swing = major_swings.iloc[i-1]
            current_swing = major_swings.iloc[i]
            
            if prev_swing['classification'] == 'HH' and current_swing['classification'] == 'LL':
                # 下跌趋势中的回撤
                high = prev_swing['price']
                low = current_swing['price']
                diff = high - low
                
                # 计算回撤水平
                for level in retracement_levels:
                    retracement_price = high - diff * level
                    # 标记接近回撤水平的摆动点
                    mask = (abs(swing_df['price'] - retracement_price) / retracement_price) < 0.01
                    swing_df.loc[mask, 'fibonacci_level'] = level
            
            elif prev_swing['classification'] == 'LL' and current_swing['classification'] == 'HH':
                # 上升趋势中的回撤
                low = prev_swing['price']
                high = current_swing['price']
                diff = high - low
                
                # 计算回撤水平
                for level in retracement_levels:
                    retracement_price = low + diff * level
                    # 标记接近回撤水平的摆动点
                    mask = (abs(swing_df['price'] - retracement_price) / retracement_price) < 0.01
                    swing_df.loc[mask, 'fibonacci_level'] = level
        
        return swing_df
    
    def comprehensive_swing_analysis(self, base_window, high_series, low_series, close_series, volume_series=None):
        """
        综合摆动点分析，结合多种方法提高准确性
        """
        #print("开始综合摆动点分析...")
        
        # 方法1: 多窗口共识检测
        #print("1. 多窗口共识检测...")
        # consensus_highs, consensus_lows = self.multi_window_swing_detection(
        #      high_series, low_series, close_series, windows=[5,7,10]
        # )
        
        # 方法2: 自适应窗口检测
        #print("2. 自适应窗口检测...")
        adaptive_highs, adaptive_lows = self.adaptive_swing_detection(
            base_window, high_series, low_series, close_series
        )
        
        # 合并所有检测结果
        # all_highs = consensus_highs + adaptive_highs
        # all_lows = consensus_lows + adaptive_lows

        all_highs = adaptive_highs
        all_lows = adaptive_lows
        
        # 聚类接近的点
        #print("3. 聚类接近的摆动点...")
        clustered_highs = self.cluster_nearby_points(all_highs)
        clustered_lows = self.cluster_nearby_points(all_lows)
        
        # 分类摆动点
        #print("4. 分类摆动点...")
        swing_df = self.classify_swing_points_robust(
            clustered_highs, clustered_lows, close_series, 
            volume_series=volume_series
        )
       
        # 趋势确认
        #print("5. 趋势确认...")
        swing_df = self.trend_confirmation(swing_df, close_series)
        
        # 斐波那契确认
        #print("6. 斐波那契确认...")
        swing_df = self.fibonacci_retracement_confirmation(swing_df)
        
        #print("分析完成!")
        return swing_df
    
    def calculate_confidence_score(self, swing_df):
        """
        计算每个摆动点的置信度分数
        """
        if swing_df.empty:
            return swing_df
        
        confidence_scores = []
        
        for _, row in swing_df.iterrows():
            score = 0
            
            # 基础分数
            if row['classification'] in ['HH', 'LL']:
                score += 30
            elif row['classification'] in ['HL', 'LH']:
                score += 20
            else:  # -H, -L
                score += 10
            
            # 趋势确认加分
            if row.get('trend_confirmed', False):
                score += 20
            
            # 斐波那契确认加分
            if pd.notna(row.get('fibonacci_level')):
                score += 25
            
            # 成交量确认 (如果可用)
            if hasattr(row, 'volume_confirm') and row['volume_confirm']:
                score += 25
            
            confidence_scores.append(min(score, 100))  # 限制最大分数为100
        
        swing_df['confidence_score'] = confidence_scores
        return swing_df

# 使用示例
def calHHLL(df,window):

    stock_data = df.copy()    
    stock_data.index = pd.to_datetime(stock_data.index,utc=True).tz_convert('Asia/Shanghai')         
    stock_data = extendData(stock_data)       
        
    # 创建分析器
    analyzer = RobustSwingPointAnalyzer()
    
    # 执行综合分析
    swing_df = analyzer.comprehensive_swing_analysis(window,stock_data['High'],stock_data['Low'],
                         stock_data['Close'],stock_data['Volume'])    
    
    # 计算置信度
    swing_df = analyzer.calculate_confidence_score(swing_df)
    
    # 过滤高置信度的摆动点
    if len(swing_df)>0:
        swing_df = swing_df[swing_df['confidence_score'] >= 40]            
        resultdf = swing_df.reset_index().drop(['level_0','index'],axis=1)
    else:
        resultdf = pd.DataFrame()    
    
    return resultdf


def AnalyzeData(sno,stype):
       
    df = pd.read_csv(PATH+"/"+stype+"/"+sno+".csv",index_col=0)
    df = convertData(df)

    #print(sno)
    df = calEMA(df)

    tempdf1 = calHHLL(df,3)
    tempdf2 = calHHLL(df,5)
    
    tempdf = pd.concat([tempdf1,tempdf2])

    if len(tempdf)>0:
        tempdf = tempdf.sort_values('date').drop_duplicates(subset=['date'], keep='last')
        df = checkLHHHLL(df, sno, stype, tempdf)        
        df = calT1(df,50)        
        df = df.reset_index()
        df.to_csv(OUTPATH+"/"+stype+"/P_"+sno+".csv",index=False)


def YFprocessData(stype):

    snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(PATH+"/"+stype)))
    SLIST = pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST[:]

    with cf.ProcessPoolExecutor(max_workers=5) as executor:
        list(tqdm(executor.map(AnalyzeData,SLIST["sno"],SLIST["stype"],chunksize=1),total=len(SLIST)))


if __name__ == '__main__':
    start = t.perf_counter()

    YFprocessData("L")    
    YFprocessData("M")
    #YFprocessData("S")

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    