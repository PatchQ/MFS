# AGENTS.md — MFS 工作守則

> 給 AI 編程助手（Kilo、Continue.dev、其他 agent）的自動載入指引  
> 最後更新：2026-06-02

---

## 1. 專案一句話

**MFS 是香港股票量化交易研究系統**：技術指標 + 機器學習 + 回測驗證 + Web 視覺化。  
遠端：`https://github.com/PatchQ/MFS.git`，主分支 `main`，作者 `PatchQ`。

---

## 2. 五大新架構模組（核心）

| 模組 | 路徑 | 職責 |
|------|------|------|
| **MFSDataHub** | `Core/MFSDataHub.py` | 單例數據中心 + 事件總線 |
| **IndicatorEngine** | `Core/IndicatorEngine.py` | 指標依賴解析 + 緩存 + 拓撲排序 |
| **DataSource** | `DataSource/` | 插件式數據源（Yahoo / HKEX / 騰訊） |
| **ScriptRunner** | `ScriptRunner/` | Subprocess + JSON-line 併發調度 |
| **SignalAgent** | `SignalAgent/` | AI 信號記憶/預測/推理/輸出（SQLite） |

## 3. 傳統子系統（沿用中）

| 路徑 | 內容 |
|------|------|
| `TA/` | 15 個技術指標模組 + 4 個 `*_bk.py` 備份（`LW_CheckBoss.py`, `LW_CheckIchimoku.py`, `LW_CheckHFH.py` 等） |
| `AI/` | 8 個 ML 模型（RF / SVM / MLP / LR / DT / XGBoost / LightGBM / ZPrediction） |
| `HEX/` | 港股數據抓取（AAStocks / CCASS / 期貨期權） |
| `HERMES/` | 應用層：voice_server / so_viewer / 掃描器 / Pine 腳本 |
| `UTIL/CommonConfig.py` | 全域配置（**所有 TA/AI 子模組在此被 `import *`**） |
| `LLM/`, `TIM/`, `ROO/` | LLM 工具、TIM 系統、debug 暫存腳本 |

## 4. 頂層執行入口

```bash
# 全量 pipeline（一次性）
python Run_Full.py

# 每日增量
python Run_Daily.py

# 技術指標
python ProcessTA.py                  # 批量
python ProcessTA.py --json-stdin     # 單筆（給 ScriptRunner 用）

# AI 預測
python ProcessAI.py
python ProcessAI.py --json-stdin

# 回測
python Run_Backtest2.py              # 主回測（backtesting.py）
python BOSS_Optimizer.py --mode grid --rounds 100

# HERMES 應用
python HERMES/voice_server.py --port 8765
python HERMES/so_viewer/app.py
python HERMES/merge_monthly.py       # HKEX 日檔→月度合併
python HERMES/hk_stock_gann_scanner.py
```

## 5. 禁區與避坑

- ⚠️ **`UTIL/CommonConfig.py` 的 `import *` 區塊**（第 20-38 行）— 全域注入 20+ 符號，不要嘗試重構 import 順序
- ⚠️ **不要刪除 `*_bk.py` 備份檔**（如 `TA/LW_CheckVCP_bk.py`），備份邏輯改由 git history 管理
- ⚠️ **不要 commit 大檔**：`backtest.log`（未 gitignore，已 ~13MB，建議加入 `.gitignore` + 設 rotating handler）、`Data/*.csv|xlsx|pkl|png` 已被 `.gitignore`
- ⚠️ **不要修改 `HERMES/so_viewer/app.py` 路徑硬編碼**（`/root/GitHub/SData/HKEX/...`），需先評估 Windows 端遷移策略
- ⚠️ **`Data/MODEL/`** 是訓練好的 `.pkl` 檔，不要覆蓋除非用戶明確要求
- ⚠️ **不要動 `LLM/ApiKey.md`**（gitignore，但若用戶貼上請勿 echo 回去）

## 6. 路徑常數

```python
PATH    = "../SData/YFData/"          # 原始 OHLCV
OUTPATH = "../SData/P_YFData/"        # 加指標後
FPATH   = "/root/GitHub/SData/FYFData/"   # Full 版
FOUTPATH= "/root/GitHub/SData/FP_YFData/" # Full 版輸出
DATADATE = "2024-01-01"
PROD     = True
HSI_TREND_FILTER = True  # EMA20 趨勢過濾
```

> Linux 路徑在 Windows 上需自行替換或 symlink。`IS_WINDOWS` 自動偵測，但 `so_viewer` / `voice_server` 仍未跨平台。

## 7. 提交風格（emoji 前綴）

```
🔧 改進/重構     🐛 Bug 修復     🌐 國際化/i18n
🌟 新功能       ⚡ 性能優化      📝 文檔         🧪 測試
```

範例：`<emoji> <scope>: <一句話描述>`

- `🔧 Tab3: 預設值改為本月第一日 + 本月 ~ 12月`
- `🐛 merge_monthly: 修復去重 bug（同一 strike 配不同 settle_price 係唔同合約，必須保留）`

## 8. 回測策略重要參數（`Run_Backtest2.py`）

```python
max_holdbars = 100   # 最大持倉 K 線
sl = -10.0           # 止損 %
tp = 20.0            # 止盈 %
dd = 0.0             # 追蹤止損（從最高點回撤 %）
cash = 200000        # 起始資金
commission = 0.002   # 手續費
```

**BOSSB 特殊邏輯**：使用進場時記錄的固定 `cl_price` / `tp2_price` 平倉，**非**百分比 sl/tp。

## 9. 信號欄位約定

`ProcessTA.py` 輸出欄位：
- TA 信號：`BOSSB`, `HHHL`, `VCP`, `HFH`, `ICHIMOKU`（布林）
- AI 信號（`ai="True"` 時附加）：`SVM`, `MLP`, `RF`
- 預測目標：訓練時 label = `F20D > 0.15`（20 日後漲幅 > 15%）

## 10. 並行 / 執行模型

- **macOS**：`ThreadPoolExecutor`（`IS_IOS`）
- **Windows/Linux**：`ProcessPoolExecutor`（`DEFAULT_MAX_WORKERS = os.cpu_count()`）
- **Subprocess 模式**：`ScriptRunner` 每個請求 spawn 子進程（記憶體隔離但成本高）

## 11. Kilo 設定參考

```jsonc
// ~/.config/kilo/kilo.jsonc
{
  "model": "minimax-cn-coding-plan/MiniMax-M3",
  "small_model": "kilo/kilo-auto/free",
  "permission": { "bash": "allow" }
}
```

- 預設模型：**M3**（minimax-cn-coding-plan/MiniMax-M3）
- 權限：bash 全開（操作時請謹慎刪除/覆蓋）
- 計畫存放：`.kilo/plans/*.md`

## 12. 觀察到的問題（先知道再動手）

1. `CommonConfig.py` 的 `import *` 啟動成本高、新人難以追蹤符號來源
2. `ProcessTA.py` 對 `IchimokuWrapper` 採 `try/except` fallback 雙路徑
3. `TA/`, `ROO/` 存在大量 `*_bk.py` 與 debug 暫存腳本
4. 沒有 pytest / CI 覆蓋
5. Linux 路徑硬編碼未跨平台
6. `backtest.log` 無 rotate

## 13. 常用診斷指令

```bash
# 查看最近 commit
git log --oneline -20

# 查看誰改了某檔
git log --oneline -- <file>

# 搜尋 TA 模組中某個函數
grep -rn "def calHHLL" TA/

# 查看回測 log 最後 50 行
Get-Content backtest.log -Tail 50

# 統計某指標的命中次數
python -c "import pandas as pd; df=pd.read_csv('Data/L_BOSSB.csv'); print(df.shape)"
```

## 14. 給 agent 的禮貌守則

- 修改前先 `read` 檔案，不要靠記憶
- 引用程式碼用 `file:line` 格式
- 修改後主動建議 `/local-review-uncommitted` 做 code review
- 不要一次大規模重構，先 small diff → 驗證 → 再推
- 遇到不明確需求時用 `question` 工具詢問，不要猜

---

> 本檔案由 Kilo 自動載入，作為 agent 行為指引。  
> 詳細架構請見 [`Architecture.md`](./Architecture.md)。
