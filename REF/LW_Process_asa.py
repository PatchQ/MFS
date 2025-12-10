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
import warnings
warnings.filterwarnings('ignore')

from YFData_Calindicator import *

PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/" 

class AdvancedSwingAnalyzer:
    def __init__(self):
        self.results = {}
    
    def multi_timeframe_consensus(self, high_series, low_series, close_series, volume_series=None):
        """
        多时间框架共识分析，结合不同窗口大小
        """
        # 定义多个窗口大小
        windows = [3,5]
        all_swing_points = []
        
        for window in windows:
            swings = self.find_swings_with_window(high_series, low_series, close_series, 
                                                 volume_series, window)
            all_swing_points.extend(swings)
        
        # 应用共识机制
        consensus_swings = self.apply_consensus_mechanism(all_swing_points, len(windows))
        
        return consensus_swings
    
    def find_swings_with_window(self, high_series, low_series, close_series, volume_series, window):
        """
        使用特定窗口大小寻找摆动点
        """
        # 使用scipy识别极值点
        high_indices = argrelextrema(high_series.values, np.greater, order=window)[0]
        low_indices = argrelextrema(low_series.values, np.less, order=window)[0]
        
        swings = []
        
        # 处理高点
        for idx in high_indices:
            if idx < len(close_series):
                swings.append({
                    'index': idx,
                    'price': high_series.iloc[idx],
                    'type': 'high',
                    'date': high_series.index[idx],
                    'window': window,
                    'close': close_series.iloc[idx],
                    'volume': volume_series.iloc[idx] if volume_series is not None else 0
                })
        
        # 处理低点
        for idx in low_indices:
            if idx < len(close_series):
                swings.append({
                    'index': idx,
                    'price': low_series.iloc[idx],
                    'type': 'low',
                    'date': low_series.index[idx],
                    'window': window,
                    'close': close_series.iloc[idx],
                    'volume': volume_series.iloc[idx] if volume_series is not None else 0
                })
        
        return swings
    
    def apply_consensus_mechanism(self, all_swings, total_windows, min_consensus=0.5):
        """
        应用共识机制，过滤被多个窗口确认的摆动点
        """
        # 按索引和类型分组
        swing_groups = {}
        
        for swing in all_swings:
            key = (swing['index'], swing['type'])
            if key not in swing_groups:
                swing_groups[key] = []
            swing_groups[key].append(swing)
        
        # 过滤共识点
        consensus_points = []
        for key, group in swing_groups.items():
            consensus_ratio = len(group) / total_windows
            
            if consensus_ratio >= min_consensus:
                # 取平均价格
                avg_price = np.mean([s['price'] for s in group])
                # 使用最早检测到的窗口的日期
                earliest_date = min([s['date'] for s in group])
                # 使用最大窗口作为代表性窗口
                max_window = max([s['window'] for s in group])
                
                consensus_points.append({
                    'index': key[0],
                    'price': avg_price,
                    'type': key[1],
                    'date': earliest_date,
                    'consensus_ratio': consensus_ratio,
                    'window': max_window,
                    'close': group[0]['close'],
                    'volume': group[0]['volume']
                })
        
        # 按索引排序
        consensus_points.sort(key=lambda x: x['index'])
        return consensus_points
    
    def enhanced_classification(self, swings_df, price_tolerance=0.005, volume_threshold=1.2):
        """
        增强的分类逻辑，特别改进LH和HL的识别
        """
        if len(swings_df) < 2:
            return pd.DataFrame(swings_df)
        
        # 转换为DataFrame以便处理
        df = pd.DataFrame(swings_df)
        df = df.sort_values('index').reset_index(drop=True)
        
        # 分离高点和低点
        highs = df[df['type'] == 'high'].copy().reset_index(drop=True)
        lows = df[df['type'] == 'low'].copy().reset_index(drop=True)
        
        # 分类高点序列
        high_classifications = []
        for i in range(len(highs)):
            if i == 0:
                high_classifications.append('Start_H')
                continue
            
            current = highs.iloc[i]
            prev = highs.iloc[i-1]
            
            price_change = (current['price'] - prev['price']) / prev['price']
            
            # 增强的LH识别逻辑
            if price_change < -price_tolerance:
                # 确认是真正的LH而不仅仅是噪音
                if self.confirm_lh_pattern(highs, i, lows):
                    high_classifications.append('LH')
                else:
                    # 可能是噪音，标记为待定
                    high_classifications.append('LH_potential')
            elif abs(price_change) <= price_tolerance:
                high_classifications.append('-H')
            else:
                # 确认是真正的HH
                if self.confirm_hh_pattern(highs, i, lows):
                    high_classifications.append('HH')
                else:
                    high_classifications.append('HH_potential')
        
        highs['classification'] = high_classifications
        
        # 分类低点序列
        low_classifications = []
        for i in range(len(lows)):
            if i == 0:
                low_classifications.append('Start_L')
                continue
            
            current = lows.iloc[i]
            prev = lows.iloc[i-1]
            
            price_change = (current['price'] - prev['price']) / prev['price']
            
            # 增强的HL识别逻辑
            if price_change > price_tolerance:
                # 确认是真正的HL而不仅仅是噪音
                if self.confirm_hl_pattern(lows, i, highs):
                    low_classifications.append('HL')
                else:
                    # 可能是噪音，标记为待定
                    low_classifications.append('HL_potential')
            elif abs(price_change) <= price_tolerance:
                low_classifications.append('-L')
            else:
                # 确认是真正的LL
                if self.confirm_ll_pattern(lows, i, highs):
                    low_classifications.append('LL')
                else:
                    low_classifications.append('LL_potential')
        
        lows['classification'] = low_classifications
        
        # 合并结果
        result_df = pd.concat([highs, lows], ignore_index=True)
        result_df = result_df.sort_values('index').reset_index(drop=True)
        
        # 最终确认逻辑
        result_df = self.final_confirmation(result_df)
        
        return result_df
    
    def confirm_lh_pattern(self, highs, current_idx, lows_df):
        """
        确认LH模式的增强逻辑
        """
        if current_idx < 1:
            return False
        
        current_high = highs.iloc[current_idx]
        prev_high = highs.iloc[current_idx-1]
        
        # 条件1: 价格必须显著低于前一个高点
        price_decline = (prev_high['price'] - current_high['price']) / prev_high['price']
        if price_decline < 0.01:  # 至少下跌1%
            return False
        
        # 条件2: 检查是否有相应的低点支撑这个LH
        # 找到当前高点前后的低点
        current_index = current_high['index']
        prev_lows = lows_df[lows_df['index'] < current_index]
        next_lows = lows_df[lows_df['index'] > current_index]
        
        if len(prev_lows) > 0 and len(next_lows) > 0:
            # 检查低点序列是否支持下降趋势
            recent_low = prev_lows.iloc[-1]
            if len(next_lows) > 0:
                next_low = next_lows.iloc[0]
                # 如果后续低点更低，支持LH判断
                if next_low['price'] < recent_low['price']:
                    return True
        
        # 条件3: 成交量确认（如果可用）
        if current_high['volume'] > 0:
            #avg_volume = np.mean([hs['volume'] for hs in highs.iloc[max(0, current_idx-5):current_idx+1]])
            avg_volume = highs['volume'].iloc[max(0, current_idx-5):current_idx+1].mean()
            if current_high['volume'] < avg_volume * 0.8:
                # LH形成时成交量通常萎缩
                return True
        
        return False
    
    def confirm_hl_pattern(self, lows, current_idx, highs_df):
        """
        确认HL模式的增强逻辑
        """
        if current_idx < 1:
            return False
        
        current_low = lows.iloc[current_idx]
        prev_low = lows.iloc[current_idx-1]
        
        # 条件1: 价格必须显著高于前一个低点
        price_increase = (current_low['price'] - prev_low['price']) / prev_low['price']
        if price_increase < 0.01:  # 至少上涨1%
            return False
        
        # 条件2: 检查是否有相应的高点支撑这个HL
        current_index = current_low['index']
        prev_highs = highs_df[highs_df['index'] < current_index]
        next_highs = highs_df[highs_df['index'] > current_index]
        
        if len(prev_highs) > 0 and len(next_highs) > 0:
            # 检查高点序列是否支持上升趋势
            recent_high = prev_highs.iloc[-1]
            if len(next_highs) > 0:
                next_high = next_highs.iloc[0]
                # 如果后续高点更高，支持HL判断
                if next_high['price'] > recent_high['price']:
                    return True
        
        # 条件3: 成交量确认（如果可用）
        if current_low['volume'] > 0:
            #avg_volume = np.mean([l['volume'] for l in lows.iloc[max(0, current_idx-5):current_idx+1]])
            avg_volume = lows['volume'].iloc[max(0, current_idx-5):current_idx+1].mean()
            if current_low['volume'] > avg_volume * 1.2:
                # HL形成时成交量通常放大
                return True
        
        return False
    
    def confirm_hh_pattern(self, highs, current_idx, lows_df):
        """
        确认HH模式的逻辑
        """
        if current_idx < 1:
            return True  # 第一个高点默认为HH
        
        current_high = highs.iloc[current_idx]
        prev_high = highs.iloc[current_idx-1]
        
        # 价格必须显著高于前一个高点
        price_increase = (current_high['price'] - prev_high['price']) / prev_high['price']
        return price_increase >= 0.01  # 至少上涨1%
    
    def confirm_ll_pattern(self, lows, current_idx, highs_df):
        """
        确认LL模式的逻辑
        """
        if current_idx < 1:
            return True  # 第一个低点默认为LL
        
        current_low = lows.iloc[current_idx]
        prev_low = lows.iloc[current_idx-1]
        
        # 价格必须显著低于前一个低点
        price_decrease = (prev_low['price'] - current_low['price']) / prev_low['price']
        return price_decrease >= 0.01  # 至少下跌1%
    
    def final_confirmation(self, swings_df):
        """
        最终确认逻辑，解决潜在冲突
        """
        # 解决连续的LH或HL
        swings_df = self.resolve_consecutive_patterns(swings_df)
        
        # 基于上下文确认模式
        swings_df = self.context_based_confirmation(swings_df)
        
        # 移除潜在标记，只保留确认的模式
        swings_df['classification'] = swings_df['classification'].replace({
            'LH_potential': 'LH',
            'HL_potential': 'HL',
            'HH_potential': 'HH',
            'LL_potential': 'LL'
        })
        
        return swings_df
    
    def resolve_consecutive_patterns(self, swings_df):
        """
        解决连续的LH或HL模式
        """
        i = 0
        while i < len(swings_df) - 2:
            current = swings_df.iloc[i]
            next1 = swings_df.iloc[i+1]
            next2 = swings_df.iloc[i+2]
            
            # 处理连续LH
            if (current['classification'] == 'LH' and 
                next1['classification'] == 'LH' and
                current['type'] == 'high' and next1['type'] == 'high'):
                
                # 保留价格较高的那个作为LH
                if current['price'] >= next1['price']:
                    swings_df.at[i+1, 'classification'] = 'noise'
                else:
                    swings_df.at[i, 'classification'] = 'noise'
            
            # 处理连续HL
            if (current['classification'] == 'HL' and 
                next1['classification'] == 'HL' and
                current['type'] == 'low' and next1['type'] == 'low'):
                
                # 保留价格较低的那个作为HL
                if current['price'] <= next1['price']:
                    swings_df.at[i+1, 'classification'] = 'noise'
                else:
                    swings_df.at[i, 'classification'] = 'noise'
            
            i += 1
        
        # 移除标记为noise的点
        swings_df = swings_df[swings_df['classification'] != 'noise'].reset_index(drop=True)
        
        return swings_df
    
    def context_based_confirmation(self, swings_df):
        """
        基于上下文确认模式
        """
        for i in range(2, len(swings_df)):
            if i >= len(swings_df):
                break
                
            current = swings_df.iloc[i]
            prev1 = swings_df.iloc[i-1]
            prev2 = swings_df.iloc[i-2]
            
            # 检查LH-LL-HH模式
            if (prev2['classification'] == 'LH' and
                prev1['classification'] == 'LL' and
                current['classification'] == 'HH'):
                # 这是一个有效的LH-LL-HH模式，确认这些点
                pass
            
            # 如果LH后面直接是HH，可能需要重新分类
            if (prev1['classification'] == 'LH' and
                current['classification'] == 'HH' and
                prev1['type'] == 'high' and current['type'] == 'high'):
                
                price_change = (current['price'] - prev1['price']) / prev1['price']
                if price_change > 0.02:  # 如果涨幅显著，前一个可能不是真正的LH
                    # 检查前一个点是否应该被重新分类
                    if i-2 >= 0:
                        prev2 = swings_df.iloc[i-2]
                        if prev2['type'] == 'high' and prev2['price'] > prev1['price']:
                            # prev1可能是一个回调低点，不是LH
                            swings_df.at[i-1, 'classification'] = 'HL'
        
        return swings_df
    
    def find_lh_ll_hh_patterns(self, swings_df, max_gap=20):
        patterns = []
        
        # 转换为列表以便处理
        swings_list = swings_df.to_dict('records')
        
        i = 0
        while i < len(swings_list) - 2:
            point1 = swings_list[i]
            point2 = swings_list[i+1] if i+1 < len(swings_list) else None
            point3 = swings_list[i+2] if i+2 < len(swings_list) else None
            
            if point2 is None or point3 is None:
                break
                        
            if (point1['classification'] == 'LH' and
                point2['classification'] == 'LL' and
                point3['classification'] == 'HH'):
                
                # 检查时间间隔是否合理
                time_gap1 = point2['index'] - point1['index']
                time_gap2 = point3['index'] - point2['index']
                
                if time_gap1 <= max_gap and time_gap2 <= max_gap:
                    patterns.append({
                        'pattern': 'LH-LL-HH',
                        'lh_index': point1['index'],
                        'll_index': point2['index'],
                        'hh_index': point3['index'],
                        'lh_price': point1['price'],
                        'll_price': point2['price'],
                        'hh_price': point3['price'],
                        'lh_date': point1['date'],
                        'll_date': point2['date'],
                        'hh_date': point3['date'],
                        'time_gap1': time_gap1,
                        'time_gap2': time_gap2,
                        'total_bars': point3['index'] - point1['index']
                    })

                    i += 3
                else:
                    i += 1
            else:
                i += 1
        
        return patterns
    
    def comprehensive_analysis(self, high_series, low_series, close_series, volume_series):
        """
        综合分析流程
        """
        print("开始综合分析...")
        
        # 1. 多时间框架共识
        print("1. 多时间框架共识分析...")
        consensus_swings = self.multi_timeframe_consensus(
            high_series, low_series, close_series, volume_series
        )
        
        # 2. 增强分类
        print("2. 增强模式分类...")
        classified_swings = self.enhanced_classification(consensus_swings)
        
        # 3. 寻找LH-LL-HH模式
        print("3. 寻找LH-LL-HH模式...")
        patterns = self.find_lh_ll_hh_patterns(classified_swings)
        
        print("分析完成!")
        return classified_swings, patterns

# 使用示例和可视化
def analyze_stock_for_patterns(sno, stype):
    """
    分析股票寻找LH-LL-HH模式
    """
    # 下载数据
    df = pd.read_csv(PATH+"/"+stype+"/"+sno+".csv",index_col=0)
    stock_data = convertData(df)
    stock_data.index = pd.to_datetime(stock_data.index,utc=True).tz_convert('Asia/Shanghai')         
    stock_data = extendData(stock_data)      

    # 创建分析器
    analyzer = AdvancedSwingAnalyzer()
    
    # 执行分析
    swings_df, patterns = analyzer.comprehensive_analysis(
        stock_data['High'],
        stock_data['Low'],
        stock_data['Close'],
        stock_data['Volume']
    )
    
    print(f"\n=== {sno} 分析结果 ===")
    print(f"总摆动点: {len(swings_df)}")
    print(f"找到的LH-LL-HH模式: {len(patterns)}")
    
    # 显示模式详情
    for i, pattern in enumerate(patterns):
        print(f"\nMode {i+1}:")
        print(f"  LH: {pd.to_datetime(pattern['lh_date']).strftime('%Y-%m-%d')} (price: {pattern['lh_price']:.2f})")
        print(f"  LL: {pd.to_datetime(pattern['ll_date']).strftime('%Y-%m-%d')} (price: {pattern['ll_price']:.2f})")
        print(f"  HH: {pd.to_datetime(pattern['hh_date']).strftime('%Y-%m-%d')} (price: {pattern['hh_price']:.2f})")
        print(f"  Time Period: {pattern['total_bars']} Bars")
    
    return swings_df, patterns, stock_data


def AnalyzeData(sno,stype):
       
    df = pd.read_csv(PATH+"/"+stype+"/"+sno+".csv",index_col=0)
    df = convertData(df)

    #print(sno)
    df = calEMA(df)

    #tempdf = calHHLL(df)
    tempdf = pd.DataFrame()
    
    if len(tempdf)>0:
        df = checkLHHHLL(df, sno, stype, tempdf)        
        df = calT1(df,150)
        # df = calT1(df,50)            
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

    #YFprocessData("L")
    #YFprocessData("M")
    #YFprocessData("S")

    sno = "0011.HK"
    
    swings_df, patterns, stock_data = analyze_stock_for_patterns(sno, "L")
    
    if patterns:
        swings_df.to_csv(f"Data\{sno}_swing_analysis.csv", index=False)
        print(f"\n结果已保存到: {sno}_swing_analysis.csv")

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
    
