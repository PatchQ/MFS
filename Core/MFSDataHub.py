"""
MFSDataHub - 統一數據管理和事件總線中心

提供功能：
- 統一的數據管理（set_data/get_data）
- 事件總線（publish/subscribe）
- 指標準冊（register_indicator）
- 向後兼容現有TA模組
"""

from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
import threading
from enum import Enum, auto


class EventType(Enum):
    """事件類型枚舉"""
    DATA_UPDATED = auto()
    INDICATOR_REGISTERED = auto()
    INDICATOR_COMPUTED = auto()
    MARKET_ALERT = auto()
    CUSTOM = auto()


@dataclass
class Event:
    """事件對象"""
    type: EventType
    source: str
    data: Any = None
    timestamp: float = field(default=0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        import time
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class Subscriber:
    """訂閱者信息"""
    callback: Callable[[Event], None]
    event_types: Set[EventType] = field(default_factory=set)
    source_filter: Optional[str] = None
    once: bool = False
    _dispatched: bool = field(default=False, repr=False)


class MFSDataHub:
    """
    統一數據管理和事件總線中心

    功能：
    - 數據存儲：set_data(key, df) / get_data(key)
    - 事件訂閱：subscribe(event_types, callback, source_filter)
    - 事件發布：publish(event)
    - 指標準冊：register_indicator(name, indicator_class)

    向後兼容：
    - 兼容現有TA模組的數據訪問模式
    - 允許指標直接訪問ohlcv數據
    """

    _instance: Optional['MFSDataHub'] = None
    _lock = threading.Lock()

    def __new__(cls):
        """單例模式，確保全局只有一個DataHub"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # 數據存儲
        self._data: Dict[str, pd.DataFrame] = {}

        # 事件訂閱者
        self._subscribers: List[Subscriber] = []

        # 指標注册表
        self._indicators: Dict[str, Any] = {}

        # 鎖（線程安全）
        self._data_lock = threading.RLock()
        self._event_lock = threading.RLock()

    # ==================== 數據管理 ====================

    def set_data(self, key: str, data: pd.DataFrame) -> None:
        """
        設置數據

        Args:
            key: 數據鍵名（如 'BTC/USDT_1h', 'SPY_D'）
            data: K線數據DataFrame
        """
        with self._data_lock:
            self._data[key] = data.copy()

            # 自動發布數據更新事件
            self.publish(Event(
                type=EventType.DATA_UPDATED,
                source=key,
                data=data,
                metadata={'key': key, 'rows': len(data)}
            ))

    def get_data(self, key: str) -> Optional[pd.DataFrame]:
        """
        獲取數據

        Args:
            key: 數據鍵名

        Returns:
            DataFrame或None（若不存在）
        """
        with self._data_lock:
            return self._data.get(key)

    def list_keys(self) -> List[str]:
        """列出所有已存儲的數據鍵"""
        with self._data_lock:
            return list(self._data.keys())

    def has_data(self, key: str) -> bool:
        """檢查數據是否存在"""
        with self._data_lock:
            return key in self._data

    def remove_data(self, key: str) -> bool:
        """刪除指定數據"""
        with self._data_lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    # ==================== 事件總線 ====================

    def subscribe(
        self,
        event_types: List[EventType],
        callback: Callable[[Event], None],
        source_filter: Optional[str] = None,
        once: bool = False
    ) -> Subscriber:
        """
        訂閱事件

        Args:
            event_types: 感興趣的事件類型列表
            callback: 回調函數
            source_filter: 只接收特定source的事件（可選）
            once: 是否只觸發一次

        Returns:
            Subscriber實例
        """
        with self._event_lock:
            subscriber = Subscriber(
                callback=callback,
                event_types=set(event_types),
                source_filter=source_filter,
                once=once
            )
            self._subscribers.append(subscriber)
            return subscriber

    def unsubscribe(self, subscriber: Subscriber) -> None:
        """取消訂閱"""
        with self._event_lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def publish(self, event: Event) -> None:
        """
        發布事件

        Args:
            event: Event對象
        """
        with self._event_lock:
            subscribers_copy = self._subscribers.copy()

        for subscriber in subscribers_copy:
            # 檢查事件類型匹配
            if event.type not in subscriber.event_types:
                continue

            # 檢查source過濾器
            if subscriber.source_filter and event.source != subscriber.source_filter:
                continue

            # 檢查是否已觸發過（對於once=True的情況）
            if subscriber._dispatched:
                continue

            try:
                subscriber.callback(event)

                if subscriber.once:
                    subscriber._dispatched = True
                    with self._event_lock:
                        if subscriber in self._subscribers:
                            self._subscribers.remove(subscriber)

            except Exception as e:
                # 避免單個訂閱者錯誤影響其他訂閱者
                import logging
                logging.error(f"Event subscriber error: {e}")

    # ==================== 指標準冊 ====================

    def register_indicator(
        self,
        name: str,
        indicator_class: type,
        params: Optional[Dict] = None
    ) -> None:
        """
        註冊指標

        Args:
            name: 指標名稱
            indicator_class: 指標類（必須是BaseIndicator子類）
            params: 預設參數
        """
        with self._data_lock:
            self._indicators[name] = {
                'class': indicator_class,
                'params': params or {}
            }

            # 發布指標註冊事件
            self.publish(Event(
                type=EventType.INDICATOR_REGISTERED,
                source='MFSDataHub',
                data={'name': name, 'class': indicator_class.__name__},
                metadata={'name': name}
            ))

    def get_indicator_class(self, name: str) -> Optional[type]:
        """獲取指標類"""
        with self._data_lock:
            info = self._indicators.get(name)
            if info:
                return info['class']
            return None

    def create_indicator(self, name: str, params: Optional[Dict] = None) -> Optional[Any]:
        """
        創建指標實例

        Args:
            name: 指標名稱
            params: 參數（可覆蓋預設）

        Returns:
            指標實例或None
        """
        with self._data_lock:
            info = self._indicators.get(name)
            if not info:
                return None

            merged_params = {**info['params'], **(params or {})}
            return info['class'](params=merged_params)

    def list_indicators(self) -> List[str]:
        """列出所有已註冊的指標"""
        with self._data_lock:
            return list(self._indicators.keys())

    # ==================== 向後兼容接口 ====================

    def set_ohlcv(self, symbol: str, timeframe: str, data: pd.DataFrame) -> None:
        """
        設置OHLCV數據（向後兼容）

        兼容舊接口：LW_CheckIchimoku.py 等

        Args:
            symbol: 交易對（如 'BTCUSDT'）
            timeframe: 時間周期（如 '1h', '1d'）
            data: OHLCV數據
        """
        key = f"{symbol}_{timeframe}"
        self.set_data(key, data)

    def get_ohlcv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        獲取OHLCV數據（向後兼容）

        Args:
            symbol: 交易對
            timeframe: 時間周期

        Returns:
            OHLCV DataFrame
        """
        key = f"{symbol}_{timeframe}"
        return self.get_data(key)

    def update_latest(self, symbol: str, timeframe: str, candle: Dict) -> None:
        """
        更新最新K線（向後兼容）

        Args:
            symbol: 交易對
            timeframe: 時間周期
            candle: K線數據字典
        """
        key = f"{symbol}_{timeframe}"
        df = self.get_data(key)
        if df is None:
            return

        # 這裡的實現視具體需求而定
        # 通常是最後一行更新或追加新行

    # ==================== 類級別便捷方法 ====================

    @classmethod
    def get_instance(cls) -> 'MFSDataHub':
        """獲取單例實例"""
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """重置單例（主要用於測試）"""
        with cls._lock:
            cls._instance = None


# 全局便捷函數
_hub: Optional[MFSDataHub] = None

def get_hub() -> MFSDataHub:
    """獲取全局DataHub實例"""
    global _hub
    if _hub is None:
        _hub = MFSDataHub()
    return _hub


def set_data(key: str, data: pd.DataFrame) -> None:
    """全局set_data"""
    get_hub().set_data(key, data)


def get_data(key: str) -> Optional[pd.DataFrame]:
    """全局get_data"""
    return get_hub().get_data(key)


def subscribe(
    event_types: List[EventType],
    callback: Callable[[Event], None],
    source_filter: Optional[str] = None
) -> Subscriber:
    """全局subscribe"""
    return get_hub().subscribe(event_types, callback, source_filter)


def publish(event: Event) -> None:
    """全局publish"""
    get_hub().publish(event)


def register_indicator(name: str, indicator_class: type, params: Optional[Dict] = None) -> None:
    """全局register_indicator"""
    get_hub().register_indicator(name, indicator_class, params)