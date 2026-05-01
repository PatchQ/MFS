---
name: stock-technical-analysis
description: Ichimoku + RSI/MACD/MA technical analysis for HK stocks. Uses specific parameters (34,5,52,26 and 49/233-day MA). Results sent to Telegram.
---

# Stock Technical Analysis Workflow

## Description
Ichimoku Kinko Hyo combined with RSI, MACD, and MA indicators for Hong Kong stocks. User prefers specific parameters: Ichimoku (34, 5, 52, 26) and MA (49-day, 233-day).

## Trigger
User asks to analyze a Hong Kong stock (e.g., 2318.HK, 0175.HK, 9973.HK)

## Workflow

### 1. Prepare Environment
```bash
source /root/GitHub/MFS/.venv/bin/activate
cd /root/GitHub/MFS/HERMES/skill
```

### 2. Update Parameters in Script
Edit `/root/GitHub/MFS/HERMES/skill/ichimoku_combinations.py`:
- Line ~50: `ticker = 'XXXX.HK'` (stock code)
- Lines ~40-45: Ichimoku parameters (Tenkan=34, Kijun=5, SenkouB=52, CloudPeriod=26)
- Lines ~47-48: MA parameters (short=49, long=233)

### 3. Run Analysis
```bash
python ichimoku_combinations.py
```
Output files:
- `/root/GitHub/MFS/HERMES/skill/ichimoku_analysis.png`
- `/root/GitHub/MFS/HERMES/skill/combined_signals.png`

### 4. Extract Key Metrics from Output
- 收盤價 (Latest close)
- RSI 指數
- 位置 (雲層上面/下面)
- 轉換線 vs 基準線
- MACD 黃金/死亡交叉
- 49日線, 233日線
- 49日 vs 233日 (黃金/死亡交叉)

### 5. Calculate Combined Signal
Formula:
- MACD 40%, RSI 30%, MA 30%
- Range: -1.0 to +1.0
- Signal labels:
  - +0.5 to +1.0: 強烈買入
  - +0.1 to +0.5: 輕微偏好
  - -0.1 to +0.1: 中立觀望
  - -0.5 to -0.1: 輕微偏淡
  - -1.0 to -0.5: 強烈賣出

### 6. Send Results to Telegram
Chat ID: 8636101711
- Send text summary (3 messages)
- Send ichimoku_analysis.png
- Send combined_signals.png

### 7. Data Limitations
- yfinance HK stock data may be incomplete
- 233日線需要250+日歷史數據，否則顯示「無法計算」
- 小市值股票數據可能只有幾個月

## Key Files
- `/root/GitHub/MFS/HERMES/skill/ichimoku_combinations.py` - Main analysis script
- `/root/GitHub/MFS/HERMES/skill/ichimoku_analysis.png` - Ichimoku chart with RSI, MACD
- `/root/GitHub/MFS/HERMES/skill/combined_signals.png` - Combined signals chart

## User Preferences
- Always respond in Traditional Chinese (繁體中文)
- User's name is LW
- Act as "H老師" - humorous 20-year veteran teacher explaining complex things simply
