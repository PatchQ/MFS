---
name: cloudflare-tunnel
description: 快速建立 Cloudflare Tunnel 對外暴露本地服務（http://10.0.0.2:80）
trigger: cloudflare tunnel
---
# Cloudflare Tunnel — 本地服務對外訪問

## 使用時機
當用戶說「cloudflare」或要求建立 tunnel 時，執行以下流程。

## 操作步驟

### 1. 先檢查是否已有運行中的 cloudflared 進程
```bash
ps aux | grep cloudflared | grep -v grep
```
若有且 URL 相同 → 直接取現有 URL，跳過啟動。

### 2. 啟動 Cloudflare Tunnel（若未運行）
```bash
cloudflared tunnel --url http://10.0.0.2:80
```
- 以 background mode 啟動
- 等待 5-8 秒讓 cloudflared 完成初始化
- 解析 stdout/stderr 中的 `Your quick Tunnel has been created! Visit it at:` 行
- 擷取 `https://*.trycloudflare.com` URL

### 3. 若日誌中未找到 URL，檢查現有 cloudflare 日誌
```bash
cat /tmp/cloudflared.log 2>/dev/null | grep -i "trycloudflare.com" | tail -3
```

## 回應格式
直接回傳 URL，例：
```
✅ Cloudflare Tunnel 已啟動！

🌐 公開 URL: https://xxxxx.trycloudflare.com
```

## 注意事項
- 快速隧道（無帳戶）無 uptime 保證，適合臨時測試
- 若用戶已有運行的 tunnel，優先複用而非重啟
- URL 通常在 `cloudflared` 輸出中找到，格式為 `https://*.trycloudflare.com`
