#!/usr/bin/env python3
"""
Voice Chat Server - Real-time Voice Pipeline
============================================
Stages: WebSocket Server with ElevenLabs STT + MiniMax LLM + Edge TTS

STT: ElevenLabs Scribe v2 (Cantonese yue)
LLM: MiniMax-M2.7
TTS: Edge TTS (primary) + MiniMax TTS (fallback)

WebSocket Endpoint: ws://localhost:8765/voice
HTTP Endpoint: http://localhost:8765 (for the voice chat web page)

Usage:
    python voice_server.py [--port 8765]

The server provides a web page at http://localhost:8765 for voice interaction.
"""

import asyncio
import argparse
import base64
import json
import logging
import os
import ssl
import struct
import threading
import uuid
import wave
from pathlib import Path
from typing import Optional

import websockets
from websockets.server import WebSocketServerProtocol
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ============================================================================
# Configuration
# ============================================================================

MINIMAX_API_KEY = os.environ.get("MINIMAX_CN_API_KEY", os.environ.get("MINIMAX_API_KEY", ""))
MINIMAX_BASE_URL = "https://api.minimaxi.com"
MINIMAX_GROUP_ID = os.environ.get("MINIMAX_GROUP_ID", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")

# Sentence-ending punctuation for Chinese/English streaming TTS
SENTENCE_ENDINGS = frozenset("。！？.!?；;")
SENTENCE_PAUSES = frozenset("，、,, ")  # Commas pause but don't end sentence

LOGGER = logging.getLogger("voice_server")
LOGGER.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
LOGGER.addHandler(handler)

# ============================================================================
# Audio Utilities
# ============================================================================

def encode_audio_base64(audio_data: bytes) -> str:
    """Encode audio bytes to base64 string."""
    return base64.b64encode(audio_data).decode("utf-8")

def decode_audio_base64(data: str) -> bytes:
    """Decode base64 string to audio bytes."""
    return base64.b64decode(data)

async def stream_audio_chunks(websocket, audio_queue: asyncio.Queue, sample_rate: int = 16000):
    """Stream audio chunks from queue to WebSocket client as base64."""
    while True:
        chunk = await audio_queue.get()
        if chunk is None:
            break
        await websocket.send_text(json.dumps({
            "type": "audio_chunk",
            "data": encode_audio_base64(chunk),
            "sample_rate": sample_rate
        }))

# ============================================================================
# Groq STT Streaming
# ============================================================================

async def groq_stt_streaming(audio_chunk: bytes, sample_rate: int = 16000) -> Optional[str]:
    """
    Send audio chunk to Groq Whisper API for transcription.
    Groq supports streaming with chunked audio input.
    
    Returns partial or final transcription text.
    """
    if not GROQ_API_KEY:
        return None
    
    import aiohttp
    
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
    
    # For streaming STT, we send complete audio chunks
    # Groq's Whisper API processes audio and returns transcription
    form = aiohttp.FormData()
    form.add_field("model", "whisper-large-v3-turbo")
    form.add_field("language", "zh")  # Chinese
    form.add_field("response_format", "verbose_json")
    
    # Add audio as file
    form.add_field("file", audio_chunk, 
                   filename="audio.webm", 
                   content_type="audio/webm")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=form, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("text", "")
                else:
                    error_text = await resp.text()
                    LOGGER.warning(f"Groq STT error {resp.status}: {error_text[:100]}")
                    return None
    except Exception as e:
        LOGGER.error(f"Groq STT exception: {e}")
        return None

async def groq_stt_full(audio_bytes: bytes, duration_ms: int, sample_rate: int = 16000) -> Optional[str]:
    """
    Transcribe complete audio using Groq Whisper.
    Used when we collect a full utterance before sending.
    """
    if not GROQ_API_KEY:
        return None
    
    import aiohttp
    
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
    
    # Write audio to temporary WAV file
    import tempfile
    import wave
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(1)  # mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        
        with open(tmp_path, 'rb') as f:
            audio_data = f.read()
        
        form = aiohttp.FormData()
        form.add_field("model", "whisper-large-v3-turbo")
        form.add_field("language", "zh")
        form.add_field("response_format", "text")
        form.add_field("file", audio_data, filename="audio.wav", content_type="audio/wav")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=form, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    LOGGER.info(f"STT result: {text[:100]}")
                    return text.strip()
                else:
                    error_text = await resp.text()
                    LOGGER.warning(f"Groq STT error {resp.status}: {error_text[:100]}")
                    return None
    except Exception as e:
        LOGGER.error(f"Groq STT exception: {e}")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


async def elevenlabs_stt(audio_bytes: bytes, sample_rate: int = 16000) -> Optional[str]:
    """
    Transcribe audio using ElevenLabs Scribe v2 API.
    Supports Cantonese (yue) and other languages.
    """
    if not ELEVENLABS_API_KEY:
        LOGGER.warning("ElevenLabs API key not configured")
        return None

    import aiohttp

    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
    }

    try:
        # ElevenLabs expects multipart form data with audio file
        form_data = aiohttp.FormData()
        form_data.add_field(
            "file",
            audio_bytes,
            filename="audio.wav",
            content_type="audio/wav"
        )
        form_data.add_field("model_id", "scribe_v2")
        form_data.add_field("language", "yue")  # Cantonese

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=form_data,
                                    timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    try:
                        transcript = result.get("text", "").strip()
                        LOGGER.info(f"ElevenLabs STT: {transcript[:100]}")
                        return transcript if transcript else None
                    except (KeyError, IndexError) as e:
                        LOGGER.warning(f"ElevenLabs response parse error: {e}")
                        return None
                else:
                    error_text = await resp.text()
                    LOGGER.warning(f"ElevenLabs STT error {resp.status}: {error_text[:200]}")
                    return None
    except Exception as e:
        LOGGER.error(f"ElevenLabs STT exception: {e}")
        return None


# Alias for backward compatibility
deepgram_stt = elevenlabs_stt


# ============================================================================
# MiniMax LLM Streaming
# ============================================================================

async def minimax_llm_streaming(messages: list, audio_queue: Optional[asyncio.Queue] = None):
    """
    Send conversation to MiniMax LLM and stream the response via yield.
    
    Yields partial text chunks. Returns None when done.
    """
    if not MINIMAX_API_KEY:
        LOGGER.error("MINIMAX_API_KEY not set")
        yield "Sorry, LLM is not configured."
        return
    
    import aiohttp
    url = f"{MINIMAX_BASE_URL}/v1/text/chatcompletion_v2"
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "MiniMax-M2.7",
        "messages": messages,
        "stream": True,
        "max_tokens": 1024,
        "temperature": 0.7,
    }
    
    full_response = ""
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    LOGGER.error(f"MiniMax LLM error {resp.status}: {error_text[:200]}")
                    yield f"LLM error: {resp.status}"
                    return
                
                # Process streaming response
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line or line == "data: [DONE]":
                        continue
                    
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta_obj = choices[0].get("delta") or {}
                                delta = delta_obj.get("content", "") if isinstance(delta_obj, dict) else str(delta_obj)
                                if delta:
                                    full_response += str(delta)
                                    yield str(delta)
                        except json.JSONDecodeError:
                            continue
                
                LOGGER.info(f"LLM full response: {full_response[:200]}")
                
    except asyncio.TimeoutError:
        LOGGER.error("MiniMax LLM timeout")
        yield "Sorry, the request timed out."
    except Exception as e:
        LOGGER.error(f"MiniMax LLM exception: {e}")
        yield f"Sorry, an error occurred: {str(e)}"

async def minimax_llm(messages: list) -> str:
    """Non-streaming MiniMax LLM call using curl to handle non-ASCII in API key."""
    if not MINIMAX_API_KEY:
        return "Sorry, LLM is not configured."
    
    import json as json_module
    import subprocess
    
    # Use OpenAI-compatible endpoint (MiniMax native format)
    url = f"{MINIMAX_BASE_URL}/v1/text/chatcompletion_v2"
    
    # Convert messages format for MiniMax OpenAI-compatible API
    minimax_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            # System messages become user message with [System] prefix
            minimax_messages.append({"role": "user", "content": f"[System] {msg['content']}"})
        else:
            minimax_messages.append({"role": msg["role"], "content": msg["content"]})
    
    payload = {
        "model": "MiniMax-M2.7",
        "messages": minimax_messages,
        "max_tokens": 1024,
    }
    
    import os
    env = os.environ.copy()
    env['MINIMAX_API_KEY'] = MINIMAX_API_KEY
    
    # Use curl with subprocess
    cmd = [
        'curl', '-s', '-X', 'POST', url,
        '-H', 'Content-Type: application/json',
        '-H', f'Authorization: Bearer {MINIMAX_API_KEY}',
        '-d', json_module.dumps(payload),
        '--max-time', '60'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=65, env=env)
        response_text = result.stdout
        
        if not response_text:
            LOGGER.error(f"MiniMax LLM empty response, stderr: {result.stderr[:200]}")
            return "Sorry, LLM returned an empty response."
        
        data = json_module.loads(response_text)
        
        if data.get("error"):
            LOGGER.error(f"MiniMax LLM error: {data['error']}")
            return f"Sorry, an error occurred: {data['error']}"
        
        # MiniMax OpenAI-compatible format: choices[0].message.content
        choices = data.get("choices", [])
        if choices and len(choices) > 0:
            msg = choices[0].get("message", {})
            content = msg.get("content", "")
            if content:
                return content
        return "No response"
                
    except subprocess.TimeoutExpired:
        LOGGER.error("MiniMax LLM timeout")
        return "Sorry, the request timed out."
    except Exception as e:
        LOGGER.error(f"MiniMax LLM exception: {e}")
        return f"Sorry, an error occurred: {str(e)}"

# ============================================================================
# MiniMax TTS Streaming
# ============================================================================

# ============================================================================
# TTS Providers
# ============================================================================

async def minimax_tts_stream(text: str, voice_id: str = "male-qn-qingse") -> Optional[bytes]:
    """
    Convert text to speech. Tries MiniMax TTS first, falls back to Edge TTS.
    """
    # Try Edge TTS first (reliable, known-good MP3)
    audio_data = await _edge_tts_stream(text)
    if audio_data:
        LOGGER.info(f"Edge TTS generated {len(audio_data)} bytes")
        return audio_data
    
    # Fallback to MiniMax TTS
    LOGGER.warning("Edge TTS failed, trying MiniMax TTS")
    audio_data = await _minimax_tts(text, voice_id)
    if audio_data:
        return audio_data
    
    return None


async def _minimax_tts(text: str, voice_id: str = "male-qn-qingse") -> Optional[bytes]:
    """MiniMax TTS API call - uses MiniMax TTS v2 API."""
    if not MINIMAX_API_KEY:
        LOGGER.warning("MINIMAX_API_KEY not set")
        return None
    
    import aiohttp
    import base64
    
    # MiniMax TTS API endpoint - use api.minimaxi.com for Token Plan keys
    url = f"https://api.minimaxi.com/v1/t2a_v2"
    
    # Truncate text to stay within limits (max 500 chars for non-streaming)
    truncated = text[:500] if len(text) > 500 else text
    
    payload = {
        "model": "speech-2.8-hd",
        "text": truncated,
        "stream": False,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": 1.0,
            "pitch": 0,
            "volume": 0,
        },
        "audio_setting": {
            "sample_rate": 16000,
            "bitrate": 128000,
            "format": "mp3",
        },
    }
    
    try:
        # Use curl subprocess to handle API key correctly
        import subprocess
        import json as json_module
        
        cmd = [
            'curl', '-s', '-X', 'POST', url,
            '-H', 'Content-Type: application/json',
            '-H', f'Authorization: Bearer {MINIMAX_API_KEY}',
            '-d', json_module.dumps(payload),
            '--max-time', '30'
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=35)
        response_text = result.stdout
        
        if not response_text:
            LOGGER.warning(f"MiniMax TTS empty response")
            return None
        
        data = json_module.loads(response_text)
        
        # Extract audio from {"data":{"audio":"base64..."}}
        audio_b64 = data.get("data", {}).get("audio", "")
        if not audio_b64:
            LOGGER.warning(f"MiniMax TTS no audio in response: {response_text[:200]}")
            return None
        
        audio_data = base64.b64decode(audio_b64)
        LOGGER.info(f"MiniMax TTS generated {len(audio_data)} bytes (voice: {voice_id})")
        return audio_data
        
    except Exception as e:
        LOGGER.warning(f"MiniMax TTS exception: {e}")
        return None


async def _edge_tts_stream(text: str, voice: str = "zh-HK-HiuGaaiNeural") -> Optional[bytes]:
    """
    Edge TTS - Free, no API key needed.
    Uses edge-tts Python package or calls edge-tts CLI.
    """
    import tempfile
    import subprocess
    
    # Method 1: Use CLI (most reliable)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        result = subprocess.run(
            ["edge-tts", "--voice", voice, "--text", text, "--write-media", tmp_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and os.path.exists(tmp_path):
            with open(tmp_path, 'rb') as f:
                audio_data = f.read()
            os.unlink(tmp_path)
            if len(audio_data) > 100:  # Reasonable audio is > 100 bytes
                LOGGER.info(f"Edge TTS CLI generated {len(audio_data)} bytes")
                return audio_data
            else:
                LOGGER.warning(f"Edge TTS CLI returned suspiciously small file: {len(audio_data)} bytes")
                return None
        else:
            LOGGER.warning(f"Edge TTS CLI failed: {result.stderr[:100]}")
            return None
    except Exception as e:
        LOGGER.warning(f"Edge TTS CLI exception: {e}")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


async def _edge_tts_fast(text: str, voice: str = "zh-HK-HiuGaaiNeural") -> Optional[bytes]:
    """
    Fast async Edge TTS — runs CLI in thread pool to avoid blocking event loop.
    Returns MP3 bytes or None on failure.
    """
    import tempfile
    import subprocess

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,  # use default thread pool
            lambda: subprocess.run(
                ["edge-tts", "--voice", voice, "--text", text, "--write-media", tmp_path],
                capture_output=True, text=True, timeout=20
            )
        )
        if result.returncode == 0 and os.path.exists(tmp_path):
            audio_data = await loop.run_in_executor(None, lambda: open(tmp_path, 'rb').read())
            try:
                os.unlink(tmp_path)
            except:
                pass
            if len(audio_data) > 100:
                LOGGER.info(f"[TTS] Edge fast: {len(audio_data)} bytes for '{text[:30]}'")
                return audio_data
        LOGGER.warning(f"[TTS] Edge fast failed: {result.stderr[:80] if result.stderr else 'no output'}")
        return None
    except Exception as e:
        LOGGER.error(f"[TTS] Edge fast exception: {e}")
        try:
            os.unlink(tmp_path)
        except:
            pass
        return None

async def _edge_tts_cli(text: str, voice: str = "zh-HK-HiuGaaiNeural") -> Optional[bytes]:
    """Edge TTS via CLI."""
    import tempfile
    import subprocess
    
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        result = subprocess.run(
            ["edge-tts", "--voice", voice, "--text", text, "--write-media", tmp_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and os.path.exists(tmp_path):
            with open(tmp_path, 'rb') as f:
                audio_data = f.read()
            os.unlink(tmp_path)
            LOGGER.info(f"Edge TTS CLI generated {len(audio_data)} bytes")
            return audio_data
        else:
            LOGGER.warning(f"Edge TTS CLI failed: {result.stderr[:100]}")
            return None
    except Exception as e:
        LOGGER.warning(f"Edge TTS CLI exception: {e}")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

async def minimax_tts_streaming(text: str, audio_queue: asyncio.Queue, voice_id: str = "male-qn-qingse"):
    """
    Convert text to speech and stream chunks to audio_queue.
    MiniMax TTS doesn't natively support chunked streaming,
    so we generate full audio and chunk it for streaming playback.
    """
    audio_data = await minimax_tts_stream(text, voice_id)
    if audio_data:
        # Chunk the audio for progressive playback
        chunk_size = 4096
        for i in range(0, len(audio_data), chunk_size):
            await audio_queue.put(audio_data[i:i+chunk_size])
            await asyncio.sleep(0.01)  # Small delay for streaming effect

# ============================================================================
# Conversation Context
# ============================================================================

class ConversationContext:
    """Manages conversation history for a session."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: list[dict] = [
            {"role": "system", "content": "你係一個有用、友善嘅AI語音助手。請用簡潔、輕鬆嘅語氣回答，適合語音交流。"}
        ]
        self.audio_buffer: list[bytes] = []
        self.silence_frames = 0
        self.is_recording = False
    
    def add_user_message(self, text: str):
        self.history.append({"role": "user", "content": text})
    
    def add_assistant_message(self, text: str):
        self.history.append({"role": "assistant", "content": text})
    
    def get_messages(self) -> list[dict]:
        return self.history

# ============================================================================

# ============================================================================
# Voice WebSocket Handler
# ============================================================================

async def handle_voice_client(websocket: WebSocket):
    """Handle a voice chat client connection."""
    session_id = str(uuid.uuid4())[:8]
    context = ConversationContext(session_id)
    audio_buffer = bytearray()
    sample_rate = 16000
    is_recording = False
    silence_threshold = 30
    silence_count = 0

    LOGGER.info(f"Voice client connected: {session_id}")

    # Starlette requires explicit accept before WebSocket communication
    await websocket.accept()

    try:
        # First message to client - send via send_text convenience method
        await websocket.send_text(json.dumps({
            "type": "connected",
            "session_id": session_id,
            "message": "連接成功！請按下麥克風按鈕開始說話"
        }))

        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            # Starlette 1.0.0: message can be str (raw text) or dict
            payload = None
            try:
                if isinstance(message, str):
                    # Raw string - could be "websocket.connect" or other non-JSON text
                    try:
                        payload = json.loads(message)
                    except json.JSONDecodeError:
                        # Not JSON - check if it's a WebSocket control message
                        LOGGER.warning(f"Received non-JSON text: {repr(message)[:80]}")
                        if message == "websocket.connect":
                            LOGGER.info("Browser WebSocket handshake received")
                            continue
                        continue
                elif isinstance(message, dict):
                    if message.get("type") == "websocket.connect":
                        # Browser WebSocket connected
                        LOGGER.info("WebSocket connect message received")
                        continue
                    elif message.get("type") == "websocket.disconnect":
                        LOGGER.info("WebSocket disconnected by client")
                        break
                    elif "text" in message and message["text"]:
                        try:
                            payload = json.loads(message["text"])
                        except (json.JSONDecodeError, TypeError) as e:
                            LOGGER.warning(f"JSON parse error: {e}")
                            continue
                    elif "bytes" in message and message["bytes"]:
                        audio_buffer.extend(message["bytes"])
                        continue
                    else:
                        LOGGER.info(f"Unknown message type: {message.get('type')}")
                        continue
                else:
                    continue
            except Exception as e:
                LOGGER.warning(f"Message parsing error: {e}, message type: {type(message)}")
                continue

            # Guard: payload must be a dict
            if not isinstance(payload, dict):
                LOGGER.warning(f"Unexpected payload type: {type(payload)}, value: {repr(payload)[:100]}")
                continue

            msg_type = payload.get("type", "")

            if msg_type == "websocket.connect":
                # TestClient sends this on connect - just acknowledge silently
                continue

            elif msg_type == "text":
                # Handle text messages (for testing without microphone)
                text_content = payload.get("content", "")
                if text_content:
                    LOGGER.info(f"Text message: {text_content}")
                    context.add_user_message(text_content)

                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "message": "思考中..."
                    }))

                    full_response = await minimax_llm(context.get_messages())

                    if full_response:
                        context.add_assistant_message(full_response)

                        await websocket.send_text(json.dumps({
                            "type": "transcript",
                            "text": full_response,
                            "direction": "assistant"
                        }))

                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "message": "生成語音回覆中..."
                        }))

                        audio_data = await minimax_tts_stream(full_response)
                        if audio_data:
                            await websocket.send_text(json.dumps({
                                "type": "audio",
                                "data": encode_audio_base64(audio_data),
                                "format": "mp3"
                            }))

                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "message": ""
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "LLM 回應失敗"
                        }))
                continue

            elif msg_type == "flush":
                # User released mic button — stream LLM response, TTS each sentence ASAP
                import time
                t0 = time.time()
                if audio_buffer:
                    LOGGER.info(f"[TIMING] Flushing audio buffer: {len(audio_buffer)} bytes")
                    t1 = time.time()
                    text = await deepgram_stt(bytes(audio_buffer), sample_rate)
                    t2 = time.time()
                    LOGGER.info(f"[TIMING] STT done in {t2-t1:.2f}s | result: {text}")
                    audio_buffer.clear()
                    is_recording = False
                    silence_count = 0

                    if text:
                        LOGGER.info(f"User said: {text}")
                        await websocket.send_text(json.dumps({
                            "type": "transcript",
                            "text": text,
                            "direction": "user"
                        }))

                        context.add_user_message(text)

                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "message": "思考中..."
                        }))

                        # ---- STREAMING: LLM + TTS sentence-by-sentence ----
                        full_response = ""
                        sentence_buffer = ""
                        t3 = time.time()
                        LOGGER.info(f"[STREAM] LLM streaming start")
                        llm_stream = minimax_llm_streaming(context.get_messages())
                        sentence_count = 0
                        async for token in llm_stream:
                            full_response += token
                            sentence_buffer += token
                            # Send streaming text update
                            await websocket.send_text(json.dumps({
                                "type": "transcript",
                                "text": full_response,
                                "direction": "assistant",
                                "delta": token,
                                "streaming": True
                            }))
                            # If sentence is complete, TTS it immediately
                            if sentence_buffer and sentence_buffer[-1] in SENTENCE_ENDINGS:
                                sentence_count += 1
                                LOGGER.info(f"[STREAM] Sentence {sentence_count} complete: '{sentence_buffer[:50]}...'")
                                audio_data = await _edge_tts_fast(sentence_buffer.strip())
                                if audio_data:
                                    await websocket.send_text(json.dumps({
                                        "type": "audio_chunk",
                                        "data": encode_audio_base64(audio_data),
                                        "format": "mp3",
                                        "sentence_idx": sentence_count
                                    }))
                                sentence_buffer = ""

                        t4 = time.time()
                        LOGGER.info(f"[TIMING] LLM stream done in {t4-t3:.2f}s | {sentence_count} sentences")

                        # TTS any remaining text in buffer
                        if sentence_buffer.strip():
                            LOGGER.info(f"[STREAM] Final buffer: '{sentence_buffer}'")
                            audio_data = await _edge_tts_fast(sentence_buffer.strip())
                            if audio_data:
                                await websocket.send_text(json.dumps({
                                    "type": "audio_chunk",
                                    "data": encode_audio_base64(audio_data),
                                    "format": "mp3",
                                    "sentence_idx": sentence_count + 1,
                                    "final": True
                                }))

                        context.add_assistant_message(full_response)
                        LOGGER.info(f"[TIMING] Total streaming round-trip: {t4-t0:.2f}s")

                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "message": ""
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "無法辨識語音，請再說一次"
                        }))

            elif msg_type == "audio_chunk":
                # Receive base64-encoded audio from browser
                audio_data_b64 = payload.get("data", "")
                audio_format = payload.get("format", "wav")
                LOGGER.info(f"Received audio_chunk: format={audio_format}, b64_len={len(audio_data_b64)}")
                if audio_data_b64:
                    try:
                        audio_bytes = base64.b64decode(audio_data_b64)
                        audio_buffer.extend(audio_bytes)
                        LOGGER.info(f"Audio buffer accumulated: {len(audio_buffer)} bytes")
                    except Exception as e:
                        LOGGER.error(f"Base64 decode error: {e}")

            elif msg_type == "start_recording":
                is_recording = True
                silence_count = 0
                audio_buffer.clear()
                LOGGER.info(f"Session {session_id}: Recording started")
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "message": "錄音中..."
                }))

            elif msg_type == "stop_recording":
                is_recording = False
                LOGGER.info(f"Session {session_id}: Recording stopped")
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "message": "處理中..."
                }))

                if len(audio_buffer) > sample_rate:
                    text = await groq_stt_full(bytes(audio_buffer), len(audio_buffer), sample_rate)
                    if text:
                        context.add_user_message(text)
                        full_response = await minimax_llm(context.get_messages())
                        if full_response:
                            context.add_assistant_message(full_response)
                            audio_data = await minimax_tts_stream(full_response)
                            if audio_data:
                                await websocket.send_text(json.dumps({
                                    "type": "audio",
                                    "data": encode_audio_base64(audio_data),
                                    "format": "mp3"
                                }))
                audio_buffer.clear()

            elif msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

            elif msg_type == "reset":
                context.history = [
                    {"role": "system", "content": "你係一個有用、友善嘅AI語音助手。請用簡潔、輕鬆嘅語氣回答，適合語音交流。"}
                ]
                audio_buffer.clear()
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "message": "對話已重置"
                }))

    except WebSocketDisconnect:
        LOGGER.info(f"Session {session_id}: Client disconnected")
    except Exception as e:
        import traceback
        LOGGER.error(f"Session {session_id}: Error: {e}\n{traceback.format_exc()}")



def check_audio_silence(audio_data: bytes, threshold: int = 500) -> bool:
    """
    Check if audio data is mostly silence.
    Audio is 16-bit signed integers (little-endian).
    """
    if len(audio_data) < 2:
        return True
    
    samples = []
    for i in range(0, len(audio_data) - 1, 2):
        try:
            sample = struct.unpack('<h', audio_data[i:i+2])[0]
            samples.append(sample)
        except:
            pass
    
    if not samples:
        return True
    
    rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
    return rms < threshold


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="Voice Chat Server", version="1.0.0")

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def root():
    """Serve the voice chat page."""
    html_path = Path(__file__).parent / "static" / "voice_chat.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return {"message": "Voice Chat Server is running", "websocket": "/ws/voice"}

@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """WebSocket endpoint for voice chat."""
    await handle_voice_client(websocket)

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "elevenlabs_configured": bool(ELEVENLABS_API_KEY),
        "minimax_configured": bool(MINIMAX_API_KEY),
    }

@app.get("/test/stt")
async def test_stt():
    """Test STT with a dummy audio."""
    test_audio = bytes(16000 * 2)  # 1 second of silence
    result = await groq_stt_full(test_audio, len(test_audio), 16000)
    return {"result": result}

@app.post("/test/llm")
async def test_llm():
    """Test LLM."""
    messages = [{"role": "user", "content": "你好，講個笑話"}]
    result = await minimax_llm(messages)
    return {"result": result}

@app.post("/test/tts")
async def test_tts(text: str = "你好，這是語音測試"):
    """Test TTS."""
    audio = await minimax_tts_stream(text)
    if audio:
        return {"audio_size": len(audio), "status": "ok"}
    return {"status": "error"}

@app.post("/test/stt")
async def test_stt(audio_b64: str = ""):
    """Test STT with base64-encoded audio."""
    if not audio_b64:
        return {"status": "error", "message": "no audio"}
    import base64
    audio_data = base64.b64decode(audio_b64)
    result = await deepgram_stt(audio_data)
    if result:
        return {"transcript": result, "status": "ok"}
    return {"status": "error", "message": "STT failed"}

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Voice Chat Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    args = parser.parse_args()
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║              🎙️  Voice Chat Server v1.0                      ║
╠═══════════════════════════════════════════════════════════════╣
║  HTTP:       http://{args.host}:{args.port}                         ║
║  WebSocket:  ws://{args.host}:{args.port}/ws/voice              ║
╠═══════════════════════════════════════════════════════════════╣
║  ElevenLabs STT: {'✅ Configured' if ELEVENLABS_API_KEY else '❌ Not configured':<28}         ║
║  MiniMax LLM: {'✅ Configured' if MINIMAX_API_KEY else '❌ Not configured':<30}         ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
