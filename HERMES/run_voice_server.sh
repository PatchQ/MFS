#!/bin/bash
# Voice Chat Server Launcher
# Place this in ~/GitHub/MFS/HERMES/

cd "$(dirname "$0")"

echo "🎙️ 啟動 Hermes 語音助手..."
echo ""

# Check Python environment
PYTHON_CMD="${PYTHON:-python3}"

# Check if virtual env exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    PYTHON_CMD=".venv/bin/python"
elif [ -d "/usr/local/lib/hermes-agent/venv/bin/python" ]; then
    PYTHON_CMD="/usr/local/lib/hermes-agent/venv/bin/python"
fi

# Check API keys
if [ -z "$MINIMAX_CN_API_KEY" ] && [ -z "$MINIMAX_API_KEY" ]; then
    echo "⚠️  警告：MINIMAX_API_KEY 未設定"
fi

if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  警告：GROQ_API_KEY 未設定"
fi

echo "🚀 啟動服務器..."
$PYTHON_CMD voice_server.py --port 8765 --host 0.0.0.0
