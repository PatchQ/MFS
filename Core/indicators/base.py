"""
BaseIndicator - 所有技術指標的抽象基類

提供指標插件的標準接口，確保所有指標實現：
- name: 指標名稱
- dependencies: 依賴的數據字段
- validate(): 參數驗證
- compute(): 指標計算
- merge(): 多周期合併
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class IndicatorResult:
    """指標計算結果容器"""
    name: str
    data: pd.DataFrame
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def merge(self, other: 'IndicatorResult') -> 'IndicatorResult':
        """合併兩個指標結果（用於多周期合併）"""
        if not isinstance(other, IndicatorResult):
            raise TypeError(f"Cannot merge IndicatorResult with {type(other)}")
        merged = pd.concat([self.data, other.data], axis=0)
        merged = merged[~merged.index.duplicated(keep='last')]
        merged = merged.sort_index()
        return IndicatorResult(
            name=self.name,
            data=merged,
            params={**self.params, **other.params},
            metadata={**self.metadata, **other.metadata}
        )


class BaseIndicator(ABC):
    """
    技術指標抽象基類

    所有自定義指標必須繼承此類並實現：
    - name: 指標唯一名稱
    - dependencies: 所需的數據依賴字段
    - validate(): 驗證參數是否有效
    - compute(): 執行指標計算
    - merge(): 合併多周期數據（可選覆寫）
    """

    # 類級別的指標註冊表（向後兼容鉤子）
    _registry: Dict[str, 'BaseIndicator'] = {}

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        初始化指標

        Args:
            params: 指標參數字典
        """
        self.params = params or {}
        self._result: Optional[IndicatorResult] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """指標名稱（子類必須實現）"""
        pass

    @property
    def dependencies(self) -> List[str]:
        """
        指標依賴的數據字段列表

        返回範例:
            ['open', 'high', 'low', 'close', 'volume']  # 完整OHLCV
            ['close']  # 僅收盤價
            ['high', 'low', 'close']  # 僅部分字段

        子類可覆寫此屬性
        """
        return ['close']

    def validate(self, data: pd.DataFrame) -> bool:
        """
        驗證數據是否滿足指標計算要求

        Args:
            data: 輸入的K線數據

        Returns:
            True if data is valid, raises ValueError otherwise
        """
        missing = set(self.dependencies) - set(data.columns)
        if missing:
            raise ValueError(
                f"Indicator '{self.name}' missing required columns: {missing}. "
                f"Available columns: {list(data.columns)}"
            )
        if data.empty:
            raise ValueError(f"Indicator '{self.name}' requires non-empty data")
        return True

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        執行指標計算

        Args:
            data: 輸入的K線數據（pd.DataFrame）

        Returns:
            計算後的指標數據（pd.DataFrame），通常添加新列到原數據或返回新DataFrame
        """
        pass

    def merge(self, other: 'IndicatorResult') -> IndicatorResult:
        """
        合併多周期指標結果（用於多周期策略）

        默認實現：簡單concat去重
        子類可根據指標特性覆寫此方法

        Args:
            other: 另一個周期的指標結果

        Returns:
            合併後的 IndicatorResult
        """
        if self._result is None:
            raise RuntimeError(f"Indicator '{self.name}' has no result to merge")
        return self._result.merge(other)

    def __init_subclass__(cls, **kwargs):
        """自動註冊子類到指標注册表（向後兼容）"""
        super().__init_subclass__(**kwargs)
        # 延遲註冊，避免在類定義時就觸發（允許之後再實現name屬性）
        # 子類應調用 cls.register() 來顯式註冊

    @classmethod
    def register(cls, indicator: 'BaseIndicator'):
        """
        將指標實例註冊到全局注册表（向後兼容鉤子）

        Args:
            indicator: 指標實例
        """
        cls._registry[indicator.name] = indicator

    @classmethod
    def get_registered(cls, name: str) -> Optional['BaseIndicator']:
        """獲取已註冊的指標"""
        return cls._registry.get(name)

    @classmethod
    def list_registered(cls) -> List[str]:
        """列出所有已註冊的指標名稱"""
        return list(cls._registry.keys())

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', params={self.params})>"


# 向後兼容：保留舊的指標函數調用方式
# 允許現有的 LW_CheckIchimoku.py 等模組继续使用
IndicatorPlugin = BaseIndicator  # 別名兼容


def create_indicator(name: str, params: Optional[Dict] = None) -> Optional[BaseIndicator]:
    """
    工廠函數：根據名稱創建指標實例（向後兼容）

    Args:
        name: 指標名稱
        params: 指標參數

    Returns:
        指標實例，或 None（若未找到）
    """
    indicator_cls = BaseIndicator.get_registered(name)
    if indicator_cls is None:
        return None
    # 假設注冊的是類而非實例
    if isinstance(indicator_cls, type):
        return indicator_cls(params=params)
    return indicator_cls