# High-Flat-High (HFH) 策略優化計劃

## 1. 概述

優化現有的 `TA/LW_CheckHFH.py` 中的 High-Flat-High 策略，使其更嚴謹，減少假信號。

## 2. 當前實現分析

**現有邏輯：**
- **條件 A (Uptrend)**: `Close > EMA10 > EMA22 > EMA50 > EMA100 > EMA250`
- **條件 B (Flat Zone)**: High-Low 範圍在 10% 內，至少 5 支蠟燭
- **條件 C (Breakout)**: 收盤價突破 Flat High + Uptrend 條件滿足

**問題：**
1. 沒有驗證第一個 High 之前的上升力道
2. Flat Zone 內的蠟燭身體大小沒有約束
3. 沒有假突破檢測機制

---

## 3. 優化後的策略架構

```
                    ┌─────────────────────────────────────────────┐
                    │           HIGH - FLAT - HIGH               │
                    │                                             │
                    │  HIGH-1   HIGH-2   HIGH-3 │ FLAT │ BREAKOUT │
                    │  (強上升)  (強上升)  (強上升) │(盤整) │ (突破)  │
                    │  必需連續   必需連續   必需連續│ 5+   │ 確認    │
                    │  3支強陽燭  Body>50%  收盤更高 │ Body │ 無假突 │
                    └─────────────────────────────────────────────┘
```

### 3.1 第一個 High 之前：強勢上升段（新增）

**目的：** 確保在盤整前有明確的上升趨勢

| 參數 | 建議值 | 說明 |
|------|--------|------|
| `min_strong_bullish` | 3 | 必需連續至少 3 支強陽燭 |
| `body_ratio` | 0.5 | 燭身 / 燭桿 比例 > 50% (排除十字星/陀螺) |
| `consecutive_higher` | True | 每支陽燭收盤價需高於前一支 |

**強陽燭定義：**
- `Close > Open` (陽燭)
- `Body = Close - Open`
- `Range = High - Low`
- `Body / Range > 0.5` (燭身至少佔整體的 50%)
- `Close > Previous Close` (收盤價遞增)

### 3.2 Flat Zone（優化）

**現有條件：**
- High-Low 範圍 ≤ 10%
- 至少 5 支蠟燭

**新增條件：**

| 參數 | 建議值 | 說明 |
|------|--------|------|
| `max_body_deviation` | 0.3 | 各燭身大小與平均燭身的偏差 ≤ 30% |
| `min_body_ratio` | 0.3 | 燭身 / 燭桿 最小比例（避免十字星） |

**燭身相似度計算：**
```python
# 計算盤整區間內各燭身與平均燭身的偏差
avg_body = mean(|Close - Open|)
for each candle in flat_zone:
    body_deviation = abs(candle_body - avg_body) / avg_body
    if body_deviation > max_body_deviation:  # 30%
        reject this flat_zone
```

### 3.3 突破日檢測（新增）

**假突破識別機制：**

| 檢測方法 | 條件 | 說明 |
|----------|------|------|
| **1. 收盤位置** | `(Close - Low) / (High - Low) > 0.6` | 收盤需在燭體上半部 |
| **2. 上影線** | `(High - Close) / (High - Low) < 0.2` | 上影線 < 20% 燭桿 |
| **3. 成交量** | `Volume > Vol_MA20 * 1.2` | 放量突破 |
| **4. 隔日確認** | `Next_Close > Breakout_Close * 0.97` | 隔日不跌超過 3% |

**阻斷假突破的條件：**
```python
# 如果突破日有以下情況，判定為假突破
false_breakout = (
    (close_strength < 0.5) or                           # 收盤太弱
    (upper_wick_ratio > 0.3) or                         # 上影線太長
    (volume < vol_ma20 * 0.8)                           # 成交量不足
)
```

---

## 4. 參數配置

```python
def calHFH_Enhanced(
    df,
    # === First High (強勢上升段) ===
    min_strong_bullish=3,        # 必需連續強陽燭數量
    body_ratio=0.5,              # 燭身 / 燭桿 最小比例
    require_consecutive_higher=True,  # 是否要求收盤價遞增
    
    # === Flat Zone (盤整區) ===
    min_flat_length=5,           # 盤整最少蠟燭數
    max_flat_pct=0.10,          # High-Low 最大範圍 (10%)
    max_body_deviation=0.30,    # 燭身大小偏差上限
    min_flat_body_ratio=0.30,   # 盤整區燭身最小比例
    
    # === Breakout (突破) ===
    min_close_strength=0.6,     # 收盤位置 (Low為0, High為1)
    max_upper_wick=0.2,         # 上影線最大比例
    min_volume_ratio=1.2,       # 成交量 / 均量 最小比例
    next_day_confirm=True,      # 是否需要隔日確認
    next_day_max_drop=0.03      # 隔日最大允許跌幅 (3%)
):
```

---

## 5. 數據結構

新增輸出欄位：

| 欄位名 | 類型 | 說明 |
|--------|------|------|
| `HFH` | bool | 最終信號 |
| `FlatCount` | int | 盤整區蠟燭數量 |
| `PreHighCount` | int | 強陽燭數量 |
| `BreakoutQuality` | float | 突破質量分數 (0-100) |
| `FalseBreakout` | bool | 是否為假突破 |

---

## 6. 實現架構

```mermaid
flowchart TD
    A[輸入 DataFrame] --> B[計算 EMA 排列]
    B --> C[識別強陽燭序列]
    C --> D{連續 ≥ 3 支強陽燭?}
    D -->|否| E[放棄]
    D -->|是| F[進入 Flat Zone 檢測]
    F --> G{Flat Zone 條件}
    G -->|High-Low ≤ 10%|
    G -->|燭身相似度 ≤ 30%|
    G -->|燭身 / 燭桿 ≥ 30%|
    G -->|至少 5 支燭?}
    G -->|否| E
    G -->|是| H[等待突破日]
    H --> I{突破日檢測}
    I -->|收盤位置 ≥ 60%|
    I -->|上影線 ≤ 20%|
    I -->|成交量 ≥ 1.2x|
    I -->|隔日確認|
    I -->|全部滿足| J[輸出 HFH 信號]
    I -->|任一失敗| K[標記假突破]
    J --> L[輸出 DataFrame]
    K --> L
```

---

## 7. 實施步驟

1. **修改函數簽名**：新增所有新參數
2. **實現強陽燭識別**：使用向量化的 `numpy` 運算
3. **實現燭身相似度檢測**：計算滾動窗口內的偏差
4. **實現突破日質量評估**：計算收盤位置、上影線比例
5. **實現假突破檢測**：隔日確認邏輯
6. **保持向後兼容**：新增欄位可有可無，不影響現有流程

---

## 8. 向後兼容性

為確保現有代碼不受影響：
- 新函數可命名為 `calHFH_Enhanced`
- 原 `calHFH` 保持不變
- 或使用可選參數 `enhanced=True` 觸發新邏輯

---

## 9. 測試策略

- 使用歷史數據進行回測
- 對比優化前後的信號數量與質量
- 驗證假突破篩選效果
