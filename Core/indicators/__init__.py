"""
MFS Indicators Package
技術指標抽象基類和插件系統
"""

from .base import BaseIndicator, IndicatorResult, create_indicator

# 導入包裝器
from .ichimoku_wrapper import IchimokuWrapper
from .hfh_wrapper import HFHWrapper
from .boss_wrapper import BossWrapper
from .vcp_wrapper import VCPWrapper
from .fisher_wrapper import FisherWrapper
from .gbs22c_wrapper import GBS22CWrapper
from .breakout200_wrapper import Breakout200Wrapper

# 自動註冊包裝器到 BaseIndicator 注册表
IchimokuWrapper.register(IchimokuWrapper())
HFHWrapper.register(HFHWrapper())
BossWrapper.register(BossWrapper())
VCPWrapper.register(VCPWrapper())
FisherWrapper.register(FisherWrapper())
GBS22CWrapper.register(GBS22CWrapper())
Breakout200Wrapper.register(Breakout200Wrapper())

__all__ = [
    'BaseIndicator',
    'IndicatorResult',
    'create_indicator',
    'IchimokuWrapper',
    'HFHWrapper',
    'BossWrapper',
    'VCPWrapper',
    'FisherWrapper',
    'GBS22CWrapper',
    'Breakout200Wrapper',
]