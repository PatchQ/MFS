# TA/LW_CalHHLL.py 優化計劃

## 問題摘要

| 問題 | 嚴重性 | 影響 |
|------|--------|------|
| 閾值邏輯過於簡化 | 🔴 高 | 誤判Swing Point |
| 初始化邏輯有盲點 | 🔴 高 | 起始方向判斷錯誤 |
| 最後極點未確認就添加 | 🔴 高 | 產生假信號 |
| 高低點交替驗證不足 | 🟡 中 | 魯棒性差 |
| 使用 iterrows 繪圖 | 🟡 中 | 效能差 |
| 缺少回測接口 | 🟡 中 | 無法直接對接策略 |

---

## 優化方案

### 1. 增強閾值計算 (calculate_daily_volatility)

**目標**：用 ATR 取代 Daily Range，並實現動態閾值

```python
def calculate_daily_volatility(self, window=20):
    # 1. 計算 True Range
    self.df['TR'] = np.maximum(
        self.df['High'] - self.df['Low'],
        np.maximum(
            abs(self.df['High'] - self.df['Close'].shift(1)),
            abs(self.df['Low'] - self.df['Close'].shift(1))
        )
    )
    
    # 2. ATR 使用 Wilder's 平滑
    self.df['ATR'] = self.df['TR'].ewm(alpha=1/window, min_periods=window).mean()
    
    # 3. 波動率標準化：用 ATR/Close 百分位數決定倍數
    self.df['Volatility_Pct'] = self.df['ATR'] / self.df['Close']
    percentile = self.df['Volatility_Pct'].rolling(window).apply(
        lambda x: pd.Series(x).quantile(0.75), raw=False
    )
    
    # 4. 動態倍數：波動率高時放大閾值
    self.df['Multiplier'] = np.where(
        self.df['Volatility_Pct'] > percentile,
        2.5,  # 高波動區間
        np.where(self.df['Close'] <= 50, 1.5, 2.0)
    )
    
    self.df['Turn_Threshold'] = self.df['ATR'] * self.df['Multiplier']
```

---

### 2. 改進初始化邏輯 (find_swing_points)

**目標**：使用 N 根K線確認趨勢方向

```python
def find_swing_points(self, confirm_bars=3):
    # ... 前置準備 ...
    
    # 使用 confirm_bars 根K線確認方向
    for i in range(confirm_bars, len(self.df)):
        recent_highs = prices_high[i-confirm_bars:i+1]
        recent_lows = prices_low[i-confirm_bars:i+1]
        
        # 檢查是否形成明確趨勢
        if all(recent_highs[j] > recent_highs[j-1] for j in range(1, confirm_bars+1)):
            looking_for = 'high'
            extreme_price = prices_high[i]
            extreme_idx = i
            break
        elif all(recent_lows[j] < recent_lows[j-1] for j in range(1, confirm_bars+1)):
            looking_for = 'low'
            extreme_price = prices_low[i]
            extreme_idx = i
            break
```

---

### 3. 添加極點確認機制

**目標**：確保Swing Point經過驗證

```python
class SwingPoint:
    def __init__(self, idx, price, point_type, confirmed=False):
        self.idx = idx
        self.price = price
        self.point_type = point_type
        self.confirmed = confirmed
        self.bar_count = 0  # 形成前的K線數
    
    def confirm(self):
        self.confirmed = True
```

---

### 4. 添加信號輸出接口

**目標**：支援回測系統

```python
def get_signals(self):
    """輸出市場結構信號"""
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
                'strength': 'STRONG'
            })
        elif prev['Classification'] == 'LL' and curr['Classification'] == 'HH':
            signals.append({
                'date': curr['Date'],
                'type': 'BULLISH_REVERSAL', 
                'strength': 'STRONG'
            })
    
    return pd.DataFrame(signals)
```

---

### 5. 向量化視覺化

**目標**：移除 iterrows 提升效能

```python
def visualize_results(self):
    # 向量化散點圖
    colors = points_df['Classification'].map({
        'HH': 'red', 'HL': 'green', 'LH': 'orange', 'LL': 'blue'
    })
    markers = points_df['Classification'].map({
        'HH': '^', 'HL': '^', 'LH': 'v', 'LL': 'v'
    })
    
    ax.scatter(points_df['Date'], points_df['Price'], 
               c=colors, marker=markers, s=100)
```

---

## 實施順序

1. [ ] 重構 `calculate_daily_volatility` 使用 ATR
2. [ ] 添加 `SwingPoint` 類和確認機制
3. [ ] 改進初始化邏輯
4. [ ] 添加 `get_signals()` 回測接口
5. [ ] 向量化視覺化
6. [ ] 整合測試與驗證
