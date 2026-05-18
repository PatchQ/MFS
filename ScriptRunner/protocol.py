"""
ScriptRunner Protocol - JSON-line 協議定義

定義 TA/AI 請求和響應的數據模型
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import json


@dataclass
class TARequest:
    """技術分析請求"""
    sno: str           # 股票代碼，如 "0001.HK"
    stype: str         # "L" 或 "M"
    tdate: str         # 日期，如 "2024-01-01"
    ai: str = "False"  # 是否執行 AI Signal

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TARequest':
        return cls(
            sno=d['sno'],
            stype=d['stype'],
            tdate=d['tdate'],
            ai=d.get('ai', 'False')
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TAResponse:
    """技術分析響應"""
    sno: str
    stype: str
    success: bool
    signal: Optional[bool] = None
    indicators: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TAResponse':
        return cls(**d)


@dataclass
class AIRequest:
    """AI 分析請求"""
    sno: str           # 股票代碼
    stype: str         # "L" 或 "M"
    tdate: str         # 日期
    model: str = "RF"  # 模型名稱: RF, SVM, MLP

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AIRequest':
        return cls(
            sno=d['sno'],
            stype=d['stype'],
            tdate=d['tdate'],
            model=d.get('model', 'RF')
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AIResponse:
    """AI 分析響應"""
    sno: str
    stype: str
    model: str
    success: bool
    prediction: Optional[bool] = None
    probability: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AIResponse':
        return cls(**d)


def parse_json_line(line: str) -> Optional[Dict[str, Any]]:
    """解析一行 JSON"""
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def emit_json_line(obj: Any) -> str:
    """發送一行 JSON"""
    return json.dumps(obj, ensure_ascii=False)
