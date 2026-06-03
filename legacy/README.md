# legacy/ — 凍結歷史代碼

> ⚠️ 此目錄下所有檔案**不應被 import 或執行**。保留實體供 diff 參考，  
> 真正的歷史紀錄在 git log 中（`git log --follow -- <file>`）。

## 內容索引

| 子目錄 | 來源 | 用途 |
|--------|------|------|
| `TA_bk/`    | `TA/LW_*_bk.py`            | 技術指標的早期版本（4 個） |
| `UTIL_bk/`  | `UTIL/LW_Collect_*.py`     | 資料抓取早期版本（2 個） |
| `backtests/`| `Run_Backtest_*.{bak,bk,BK}` | 早期回測腳本（3 個） |
| `runners/`  | `Run_Daily_old.py`, `ProcessTA2.py` | 早期 pipeline 入口（2 個） |
| `ROO/`      | `ROO/`（整個）             | HFH 優化過程的 debug/分析腳本（48 個，無任何 import 引用）|

## 為何不直接刪除？

- 部分檔案可能仍有研究價值（例如 `ROO/OptimizeHFH.py` 雖未引用，但實驗紀錄）
- 用戶決策（2026-06-03）：保留實體比依賴 git history 較保險
- 若確認要刪除，先在分支驗證後再 PR

## 還原方式

```bash
# 從 legacy/ 還原到原位
git mv legacy/TA_bk/LW_CheckBoss_bk.py TA/

# 從 git history 還原更早版本
git log --all --diff-filter=D --name-only | grep LW_CheckBoss_bk
git checkout <commit>^ -- TA/LW_CheckBoss_bk.py
```
