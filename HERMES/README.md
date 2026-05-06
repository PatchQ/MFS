# 🎙️ Hermes Voice Chat

即時語音對話系統 — 三階段完整實現

## 功能架構

```
┌─────────────────────────────────────────────────────────────┐
│                    語音伺服器 (voice_server.py)              │
│                                                             │
│  WebSocket Server ←→ 瀏覽器 (WebRTC 音頻捕獲)                │
│         │                                                    │
│         ├── Groq Whisper (STT) ──── 語音 → 文字              │
│         ├── MiniMax LLM ─────────── 文字 → 回覆              │
│         └── MiniMax TTS ─────────── 回覆 → 語音              │
└─────────────────────────────────────────────────────────────┘
```

## 快速啟動

### 1. 設定 API Keys

```bash
export MINIMAX_CN_API_KEY="sk-cp-xxxxx"  # MiniMax LLM + TTS
export GROQ_API_KEY="sk_xxxx"             # Groq Whisper STT
```

### 2. 啟動伺服器

```bash
cd ~/GitHub/MFS/HERMES
python3 voice_server.py --port 8765
```

### 3. 打開瀏覽器

```
http://localhost:8765
```

## 三階段實現

### Stage 1: WebSocket + Groq STT
- ✅ WebSocket 伺服器 (`/ws/voice`)
- ✅ 瀏覽器音頻捕獲 (WebRTC)
- ✅ Groq Whisper 語音轉文字
- ✅ 簡單 VAD (語音活動檢測)

### Stage 2: MiniMax LLM + TTS
- ✅ MiniMax LLM 文字生成
- ✅ MiniMax TTS 語音合成
- ✅ 完整對話上下文管理
- ✅ 文字回覆 + 語音回覆

### Stage 3: Web 介面
- ✅ HTML5 語音頁面
- ✅ 即時音頻視覺化
- ✅ 按住說話 (hold-to-talk)
- ✅ 對話歷史顯示

## API 端點

| 端點 | 方法 | 說明 |
|------|------|------|
| `/` | GET | 語音聊天頁面 |
| `/ws/voice` | WS | 語音 WebSocket |
| `/health` | GET | 健康檢查 |
| `/test/stt` | GET | 測試 STT |
| `/test/llm` | POST | 測試 LLM |
| `/test/tts` | POST | 測試 TTS |

## WebSocket 訊息格式

### 客戶端 → 伺服器

```json
// 音頻數據
{"type": "audio_chunk", "data": "<base64>"}
// 開始錄音
{"type": "start_recording"}
// 停止錄音
{"type": "stop_recording"}
// 心跳
{"type": "ping"}
// 重置對話
{"type": "reset"}
```

### 伺服器 → 客戶端

```json
// 連接成功
{"type": "connected", "session_id": "xxx", "message": "..."}
// 用戶語音識別結果
{"type": "transcript", "text": "你好", "direction": "user"}
// AI 回覆
{"type": "transcript", "text": "你好！", "direction": "assistant"}
// 語音回覆
{"type": "audio", "data": "<base64>", "format": "mp3"}
// 狀態更新
{"type": "status", "message": "處理中..."}
// 錯誤
{"type": "error", "message": "..."}
```

## 對話流程

1. **連接** → WebSocket 握手成功
2. **錄音** → 按住麥克風說話
3. **傳輸** → 音頻 chunk 發送到伺服器
4. **辨識** → Groq Whisper 轉換為文字
5. **生成** → MiniMax LLM 生成回覆
6. **語音** → MiniMax TTS 合成音頻
7. **播放** → 瀏覽器播放回覆音頻
8. **重複** → 繼續對話

## 目錄結構

```
~/GitHub/MFS/HERMES/
├── voice_server.py      # 主伺服器 (所有階段)
├── static/
│   └── voice_chat.html  # Web 語音介面
├── run_voice_server.sh  # 啟動腳本
└── README.md           # 本文件
```

## 依賴

```
websockets>=10.0
aiohttp>=3.8
fastapi>=0.100
uvicorn>=0.20
```

## 常見問題

### Q: 出現 "Cannot access microphone"
需要在瀏覽器允許麥克風權限。HTTPS 環境下必須使用 HTTPS 或 localhost。

### Q: Groq STT 無法辨識
確認音頻格式為 16kHz mono。瀏覽器預設通常已正確設定。

### Q: MiniMax TTS 無聲音
確認 API Key 有 TTS 配額，檢查 `/test/tts` 端點。

### Q: 延遲太高
說話停頓後需等待語音活動檢測 (VAD) 確認說話結束，約 1-2 秒。
