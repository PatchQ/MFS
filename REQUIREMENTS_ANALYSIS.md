# MFS 專案依賴分析報告
> 生成時間：2026-05-01

---

## 1. 專案概覽

**路徑：** `~/GitHub/MFS`  
**性質：** 量化交易研究系統（Quantitative Trading Research System）  
**語言：** Python  
**架構：** 技術分析（TA）+ AI 模型的回測系統  

---

## 2. Python 檔案統計

| 項目 | 數量 |
|------|------|
| 總 Python 檔案 | 68 個 |
| 主要執行腳本 | 18 個 |
| TA 指標模組 | 20 個 |
| AI 模型模組 | 7 個 |
| HEX 數據模組 | 13 個 |
| LLM 工具 | 2 個 |

---

## 3. 核心模組說明

### 3.1 TA 目錄（技術分析指標）
| 檔案 | 功能 |
|------|------|
| `LW_CheckBoss.py` | BOSS 指標檢測 |
| `LW_CheckVCP.py` | VCP（樞軸波動）檢測 |
| `LW_CheckIchimoku.py` | 一目均衡表 |
| `LW_CheckFisher.py` | Fisher 轉換指標 |
| `LW_CheckHFH.py` | 高位平坦/高位處理 |
| `LW_CheckBreakout200.py` | 突破 200 日均線 |
| `LW_CheckGBS22C.py` | GBS 22 日週期 |
| `LW_CheckWave.py` | 波浪理論 |
| `LW_CalHHLL.py` | 計算高低位 |
| `LW_Calindicator.py` | 計算指標 |

### 3.2 AI 目錄（AI 預測模型）
| 檔案 | 模型類型 |
|------|---------|
| `MLP.py` | 多層感知機 |
| `SVM.py` | 支援向量機 |
| `RandomForest.py` | 隨機森林 |
| `LogisticRegression.py` | 邏輯斯迴歸 |
| `DecisionTree.py` | 決策樹 |
| `XGBoost.py` | XGBoost |
| `LightGBM.py` | LightGBM |
| `ZPrediction.py` | Z 預測框架 |

### 3.3 HEX 目錄（數據獲取）
| 檔案 | 功能 |
|------|------|
| `TushareData.py` | Tushare 數據接口 |
| `AA_GetStockListData.py` | 股票清單數據 |
| `AA_GetIndustryList.py` | 行業分類 |
| `CCASS_GetAll.py` | CCASS 持股數據 |
| `IndexFuture.py` | 指數期貨 |
| `IndexOption.py` | 指數期權 |

### 3.4 主要執行腳本
| 腳本 | 功能 |
|------|------|
| `Run_Backtest.py` | 主回測引擎 |
| `Run_Backtest2.py` | 回測 v2 |
| `Run_Backtest_Combined.py` | 組合回測 |
| `Run_Daily.py` | 每日運行 |
| `BOSS_Optimizer.py` | BOSS 參數優化 |
| `AI_Backtest.py` | AI 模型回測 |
| `AI_VCP.py` | AI + VCP 分析 |
| `RUN_TUNE.py` | 參數調優 |

---

## 4. 依賴套件清單

### 4.1 已成功安裝（via uv）
```
pandas          - 數據處理
numpy           - 數值計算
scikit-learn    - 機器學習模型
yfinance        - Yahoo Finance 數據
backtesting     - 回測框架
matplotlib      - 繪圖
requests        - HTTP 請求
beautifulsoup4  - 網頁解析
joblib          - 模型序列化
openpyxl        - Excel 讀寫
holidays        - 假期數據（HK）
scipy           - 科學計算
tqdm            - 進度條
mplfinance      - K 線圖
prophet         - 時間序列預測
tushare         - A 股數據
pytrends        - Google Trends
openrouter      - LLM API 路由
curl_cffi      - 高速 HTTP
```

### 4.2 特殊說明
- **OpenRouter API Key**（`LLM/ApiKey.md`）：已記錄，需付費使用
- **News API**（`LLM/Skill.md`）：使用 Google News RSS，免費

---

## 5. 數據目錄結構

```
Data/
├── stocklist.csv        # 股票清單
├── slist.xlsx           # 股票清單 Excel
├── allvcp.xlsx          # VCP 股票
├── indlist.csv          # 行業清單
├── L_BOSSB.csv          # BOSS 回測結果（Long）
├── M_BOSSB.csv          # BOSS 回測結果（Medium）
├── L_VCP.csv            # VCP 回測結果（Long）
└── M_VCP.csv            # VCP 回測結果（Medium）
```

---

## 6. 環境設定

### 6.1 虛擬環境
- **位置：** `~/GitHub/MFS/.venv`
- **Python：** 3.11.15
- **激活方式：** `source ~/GitHub/MFS/.venv/bin/activate`

### 6.2 執行測試
```bash
cd ~/GitHub/MFS
source .venv/bin/activate
python Run_Backtest.py
```

---

## 7. Git 版本控制

| 項目 | 設定值 |
|------|--------|
| 遠端 | `https://github.com/PatchQ/MFS.git` |
| 分支 | `main` |
| 使用者 | PatchQ |
| PAT | ✅ 已設定 |

---

## 8. 執行流程建議

```
1. 每日數據更新：Run_Daily.py
2. 技術指標計算：ProcessTA.py
3. AI 模型預測：ProcessAI.py
4. 參數優化：BOSS_Optimizer.py
5. 回測：Run_Backtest.py / Run_Backtest_Combined.py
```
