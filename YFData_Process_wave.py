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
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from YFData_Calindicator import *

PATH = "../SData/YFData/"
OUTPATH = "../SData/P_YFData/" 

class YFinanceWaveAnalyzer:
    def __init__(self):
        self.data = None
        self.patterns = []
    
    def fetch_data(self, stype: str, symbol: str, period: int = 250, interval: str = "1d") -> pd.DataFrame:
        """
        从yfinance获取股票数据
        """
        try:
            ticker = yf.Ticker(symbol)
            self.data = pd.read_csv(PATH+"/"+stype+"/"+symbol+".csv",index_col=0).tail(period) 
            
            # 确保数据包含必要的列
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in self.data.columns for col in required_columns):
                raise ValueError("数据缺少必要的列")
                
            print(f"成功获取 {symbol} 数据: {len(self.data)} 条记录")
            return self.data
        except Exception as e:
            print(f"获取数据失败: {e}")
            return pd.DataFrame()
    
    def find_extremes_with_volume_confirmation(self, window: int = 5) -> Tuple[List[int], List[int]]:
        """
        使用价格和成交量确认的高低点识别
        """
        if self.data is None or len(self.data) < window * 2:
            return [], []
            
        highs, lows = [], []
        high_prices = self.data['High'].values
        low_prices = self.data['Low'].values
        close_prices = self.data['Close'].values
        volumes = self.data['Volume'].values
        n = len(high_prices)
        
        # 计算成交量的移动平均作为参考
        volume_ma = pd.Series(volumes).rolling(window=5).mean().values
        
        for i in range(window, n - window):
            current_high = high_prices[i]
            current_low = low_prices[i]
            current_volume = volumes[i]
            
            # 检查高点 - 结合价格和成交量
            high_condition = (
                all(current_high >= high_prices[i-j] for j in range(1, window+1)) and
                all(current_high >= high_prices[i+j] for j in range(1, window+1)) and
                current_volume > volume_ma[i] * 0.8  # 成交量不低于平均的80%
            )
            
            # 检查低点
            low_condition = (
                all(current_low <= low_prices[i-j] for j in range(1, window+1)) and
                all(current_low <= low_prices[i+j] for j in range(1, window+1)) and
                current_volume > volume_ma[i] * 0.7  # 对低点的成交量要求稍低
            )
            
            if high_condition:
                highs.append(i)
            if low_condition:
                lows.append(i)
        
        return self.filter_extremes(highs, lows, close_prices)
    
    def filter_extremes(self, highs: List[int], lows: List[int], prices: List[float]) -> Tuple[List[int], List[int]]:
        """
        过滤掉太接近的极值点
        """
        min_distance = 10  # 最小距离3个周期
        
        # 过滤高点
        filtered_highs = []
        for i, idx in enumerate(highs):
            if i == 0:
                filtered_highs.append(idx)
            else:
                if idx - filtered_highs[-1] >= min_distance:
                    filtered_highs.append(idx)
                else:
                    # 如果太接近，保留价格更高的点
                    if prices[idx] > prices[filtered_highs[-1]]:
                        filtered_highs[-1] = idx
        
        # 过滤低点
        filtered_lows = []
        for i, idx in enumerate(lows):
            if i == 0:
                filtered_lows.append(idx)
            else:
                if idx - filtered_lows[-1] >= min_distance:
                    filtered_lows.append(idx)
                else:
                    # 如果太接近，保留价格更低的点
                    if prices[idx] < prices[filtered_lows[-1]]:
                        filtered_lows[-1] = idx
        
        return filtered_highs, filtered_lows
    
    def find_extremes_multi_timeframe(self) -> Tuple[List[int], List[int]]:
        """
        多时间窗口综合分析
        """
        windows = [5,8,10]  # 使用更短的窗口来捕捉更多细节
        all_highs, all_lows = [], []
        
        for window in windows:
            highs, lows = self.find_extremes_with_volume_confirmation(window)
            all_highs.extend(highs)
            all_lows.extend(lows)
        
        # 统计每个位置被识别为高低点的次数
        from collections import Counter
        high_counter = Counter(all_highs)
        low_counter = Counter(all_lows)
        
        # 保留被至少2个窗口确认的点
        # confirmed_highs = [idx for idx, count in high_counter.items() if count >= 2]
        # confirmed_lows = [idx for idx, count in low_counter.items() if count >= 2]

        confirmed_highs = [idx for idx, count in high_counter.items() if count >= 1]
        confirmed_lows = [idx for idx, count in low_counter.items() if count >= 1]
        
        # 按位置排序
        confirmed_highs.sort()
        confirmed_lows.sort()
        
        return confirmed_highs, confirmed_lows
    
    def classify_wave_points(self, highs: List[int], lows: List[int]) -> List[Dict]:
        """
        分类波浪点：HH, HL, LH, LL
        """
        if self.data is None:
            return []
            
        close_prices = self.data['Close'].values
        extremes = []
        
        # 合并所有极值点并按位置排序
        all_points = []
        for h in highs:
            all_points.append(('high', h, close_prices[h]))
        for l in lows:
            all_points.append(('low', l, close_prices[l]))
        
        all_points.sort(key=lambda x: x[1])
        
        if len(all_points) < 3:
            return extremes
        
        # 分类极值点
        for i in range(2, len(all_points)):
            prev2_type, prev2_idx, prev2_price = all_points[i-2]
            prev1_type, prev1_idx, prev1_price = all_points[i-1]
            curr_type, curr_idx, curr_price = all_points[i]
            
            # 确保交替出现高低点
            if prev2_type == prev1_type:
                continue
                
            if prev1_type == 'high' and curr_type == 'low':
                # HL 或 LL 判断
                if curr_price > prev2_price:
                    wave_type = 'HL'
                else:
                    wave_type = 'LL'
                extremes.append({
                    'type': wave_type, 
                    'index': curr_idx, 
                    'price': curr_price,
                    'date': self.data.index[curr_idx] if hasattr(self.data.index, '__getitem__') else None
                })
                    
            elif prev1_type == 'low' and curr_type == 'high':
                # HH 或 LH 判断
                if curr_price > prev2_price:
                    wave_type = 'HH'
                else:
                    wave_type = 'LH'
                extremes.append({
                    'type': wave_type,
                    'index': curr_idx,
                    'price': curr_price,
                    'date': self.data.index[curr_idx] if hasattr(self.data.index, '__getitem__') else None
                })
        
        return extremes
    
    def enhanced_lh_hl_detection(self, extremes: List[Dict]) -> List[Dict]:
        """
        加强的LH和HL检测
        """
        enhanced_extremes = []
        
        for i, extreme in enumerate(extremes):
            if extreme['type'] in ['LH', 'HL']:
                idx = extreme['index']
                
                # 检查价格行为确认
                if self.confirm_reversal_pattern(idx, extreme['type']):
                    enhanced_extremes.append(extreme)
                else:
                    # 如果不确认，可能重新分类
                    new_type = self.reclassify_extreme(extreme, extremes, i)
                    extreme['type'] = new_type
                    enhanced_extremes.append(extreme)
            else:
                enhanced_extremes.append(extreme)
        
        return enhanced_extremes
    
    def confirm_reversal_pattern(self, idx: int, extreme_type: str) -> bool:
        """
        确认反转模式
        """
        if self.data is None:
            return False
            
        close_prices = self.data['Close'].values
        lookback = 5
        lookforward = 3
        
        if idx < lookback or idx > len(close_prices) - lookforward - 1:
            return True  # 边界情况，默认确认
        
        # 检查前后的价格行为
        prior_trend = np.mean(np.diff(close_prices[idx-lookback:idx]))
        future_trend = np.mean(np.diff(close_prices[idx:idx+lookforward]))
        
        if extreme_type == 'LH':
            # LH点应该前有上升趋势，后有下降趋势
            return prior_trend > 0 and future_trend < 0
        elif extreme_type == 'HL':
            # HL点应该前有下降趋势，后有上升趋势
            return prior_trend < 0 and future_trend > 0
        
        return True
    
    def reclassify_extreme(self, extreme: Dict, extremes: List[Dict], current_idx: int) -> str:
        """
        重新分类不确定的极值点
        """
        if current_idx == 0 or current_idx == len(extremes) - 1:
            return extreme['type']
            
        prev_extreme = extremes[current_idx - 1]
        next_extreme = extremes[current_idx + 1] if current_idx < len(extremes) - 1 else None
        
        # 基于相邻点重新分类
        if extreme['type'] == 'LH':
            # 如果LH点价格实际上比前高还高，可能是HH
            if extreme['price'] > prev_extreme['price'] and prev_extreme['type'] in ['HH', 'LH']:
                return 'HH'
        elif extreme['type'] == 'HL':
            # 如果HL点价格实际上比前低还低，可能是LL
            if extreme['price'] < prev_extreme['price'] and prev_extreme['type'] in ['LL', 'HL']:
                return 'LL'
        
        return extreme['type']
    
    def find_lh_ll_hh_patterns(self) -> List[Dict]:
        """
        寻找LH-LL-HH模式
        """
        if self.data is None:
            return []
            
        # 获取极值点
        highs, lows = self.find_extremes_multi_timeframe()
        extremes = self.classify_wave_points(highs, lows)
        
        # 加强LH/HL检测
        extremes = self.enhanced_lh_hl_detection(extremes)
        
        patterns = []
        close_prices = self.data['Close'].values
        
        # 寻找连续的模式
        for i in range(len(extremes) - 2):
            first = extremes[i]
            second = extremes[i+1]
            third = extremes[i+2]
            
            # LH-LL-HH 模式
            if (first['type'] == 'LH' and 
                second['type'] == 'LL' and 
                third['type'] == 'HH'):
                
                pattern = {
                    'pattern': 'LH-LL-HH',
                    'points': [first, second, third],
                    'start_index': first['index'],
                    'end_index': third['index'],
                    'start_date': first.get('date'),
                    'end_date': third.get('date'),
                    'price_change': close_prices[third['index']] - close_prices[first['index']],
                    'confidence': self.calculate_pattern_confidence(close_prices, [first, second, third])
                }
                patterns.append(pattern)
            
            # LH-HL-HH 模式
            elif (first['type'] == 'LH' and 
                  second['type'] == 'HL' and 
                  third['type'] == 'HH'):
                
                pattern = {
                    'pattern': 'LH-HL-HH',
                    'points': [first, second, third],
                    'start_index': first['index'],
                    'end_index': third['index'],
                    'start_date': first.get('date'),
                    'end_date': third.get('date'),
                    'price_change': close_prices[third['index']] - close_prices[first['index']],
                    'confidence': self.calculate_pattern_confidence(close_prices, [first, second, third])
                }
                patterns.append(pattern)
        
        return patterns
    
    def calculate_pattern_confidence(self, prices: List[float], points: List[Dict]) -> float:
        """
        计算模式置信度
        """
        if len(points) < 3:
            return 0.0
        
        confidence = 1.0
        
        # 基于价格变化幅度
        price_changes = []
        for i in range(1, len(points)):
            change = abs(prices[points[i]['index']] - prices[points[i-1]['index']])
            price_changes.append(change)
        
        avg_change = np.mean(price_changes) if price_changes else 0
        total_range = max(prices) - min(prices)
        
        if total_range > 0:
            # 价格变化占整体范围的比例越大，置信度越高
            confidence *= min(avg_change / total_range * 10, 1.0)
        
        # 基于模式完整性
        expected_types = ['LH', 'LL', 'HH'] if points[0]['type'] == 'LH' else ['LH', 'HL', 'HH']
        actual_types = [p['type'] for p in points]
        
        if actual_types == expected_types:
            confidence *= 1.2  # 完整模式加分
        
        return min(confidence, 1.0)
    
    def analyze_stock(self, stype: str, symbol: str, period: int = 250) -> Dict:
        """
        综合分析股票波浪模式
        """
        # 获取数据
        data = self.fetch_data(stype, symbol, period)
        if data.empty:
            return {}
        
        # 寻找模式
        patterns = self.find_lh_ll_hh_patterns()
        
        # 获取极值点用于显示
        highs, lows = self.find_extremes_multi_timeframe()
        extremes = self.classify_wave_points(highs, lows)
        
        return {
            'symbol': symbol,
            'data': self.data,
            'highs': highs,
            'lows': lows,
            'extremes': extremes,
            'patterns': patterns,
            'pattern_count': len(patterns)
        }
    
    def plot_analysis(self, results: Dict, save_path: Optional[str] = None):
        """
        绘制分析结果
        """
        if self.data is None or not results:
            print("没有数据可绘制")
            return
            
        plt.figure(figsize=(15, 10))
        
        # 绘制价格
        plt.subplot(2, 1, 1)
        plt.plot(self.data.index, self.data['Close'], label='Close Price', linewidth=1)
        
        # 标记高低点
        highs = results['highs']
        lows = results['lows']
        
        plt.scatter(self.data.index[highs], self.data['Close'].iloc[highs], 
                   color='red', marker='v', s=50, label='Highs', zorder=5)
        plt.scatter(self.data.index[lows], self.data['Close'].iloc[lows], 
                   color='green', marker='^', s=50, label='Lows', zorder=5)
        
        # 标记模式
        for pattern in results['patterns']:
            points = pattern['points']
            for point in points:
                plt.scatter(point.get('date'), point['price'], 
                           color='orange', s=80, edgecolors='black', zorder=6)
            
            # 连接模式点
            pattern_dates = [p.get('date') for p in points]
            pattern_prices = [p['price'] for p in points]
            plt.plot(pattern_dates, pattern_prices, 'o-', color='purple', 
                    linewidth=2, label=f"{pattern['pattern']} Pattern")
        
        plt.title(f"{results['symbol']} - Wave Analysis")
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # 绘制成交量
        plt.subplot(2, 1, 2)
        plt.bar(self.data.index, self.data['Volume'], alpha=0.3, color='gray')
        plt.title('Volume')
        plt.ylabel('Volume')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        plt.show()

# 使用示例
def demo_analysis():
    analyzer = YFinanceWaveAnalyzer()
    
    # 分析苹果股票

    stype = "L"
    sno = "0011.HK"
    period = 1500

    results = analyzer.analyze_stock(stype, sno, period)
    
    if results:
        print(f"\n=== {results['symbol']} 波浪分析结果 ===")
        print(f"找到的高点数量: {len(results['highs'])}")
        print(f"找到的低点数量: {len(results['lows'])}")
        print(f"识别的极值点类型: {[e['type'] for e in results['extremes']]}")
        print(f"找到的LH-LL-HH模式数量: {results['pattern_count']}")
        
        for pattern in results['patterns']:
            print(f"\n模式: {pattern['pattern']}")
            print(f"  置信度: {pattern['confidence']:.2f}")
            print(f"  价格变化: {pattern['price_change']:.2f}")
            print(f"  时间范围: {pattern['start_date']} 到 {pattern['end_date']}")
        
        # 绘制图表
        #analyzer.plot_analysis(results)


if __name__ == "__main__":
    demo_analysis()

