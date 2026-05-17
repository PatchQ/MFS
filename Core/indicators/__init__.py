"""
MFS Indicators Package
技術指標抽象基類和插件系統
"""

from .base import BaseIndicator, IndicatorResult, create_indicator

# 導入包裝器
from .ichimoku_wrapper import IchimokuWrapper
from .hfh_wrapper import HFHWrapper

# 自動註冊包裝器到 BaseIndicator 注册表
IchimokuWrapper.register(IchimokuWrapper())
HFHWrapper.register(HFHWrapper())

__all__ = [
    'BaseIndicator',
    'IndicatorResult',
    'create_indicator',
    'IchimokuWrapper',
    'HFHWrapper',
]