"""
ScriptRunner - 第四階段：腳本隔離執行器

提供 JSON-line 協議的 subprocess 調度能力
"""

from .protocol import (
    TARequest,
    TAResponse,
    AIRequest,
    AIResponse,
    parse_json_line,
    emit_json_line
)
from .runner import (
    ScriptRunner,
    ScriptRunnerSimple
)

__all__ = [
    # Protocol
    'TARequest',
    'TAResponse',
    'AIRequest',
    'AIResponse',
    'parse_json_line',
    'emit_json_line',
    # Runner
    'ScriptRunner',
    'ScriptRunnerSimple',
]

__version__ = '1.0.0'
