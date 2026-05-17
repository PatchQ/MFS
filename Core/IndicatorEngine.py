"""
IndicatorEngine - 指標準執行引擎

提供功能：
- 管理指標執行順序（根據依賴關係自動解析）
- 支持指標緩存和增量計算
- 插件式指標擴展
- 向後兼容現有TA模組
"""

from typing import Any, Dict, List, Optional, Set, Type, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
import threading
import time

from .indicators.base import BaseIndicator, IndicatorResult


class CycleNode:
    """依賴圖循環檢測節點"""
    indicator: BaseIndicator
    deps: List[str]


@dataclass
class ComputeTask:
    """計算任務"""
    indicator_name: str
    params: Dict[str, Any]
    data_key: str


@dataclass
class ExecutionResult:
    """執行結果"""
    success: bool
    indicator_name: str
    result: Optional[IndicatorResult] = None
    error: Optional[str] = None
    compute_time: float = 0.0


class IndicatorEngine:
    """
    指準執行引擎

    功能：
    - 自動解析指標依賴並確定執行順序
    - 支持指標結果緩存
    - 支援多 symbol/timeframe 組合
    - 插件式指標擴展

    向後兼容：
    - 兼容現有TA模組的指標調用方式
    - 支持舊的指標函數接口
    """

    def __init__(self, datahub=None):
        """
        初始化引擎

        Args:
            datahub: MFSDataHub實例（可選，默认创建新实例）
        """
        from .MFSDataHub import MFSDataHub
        self._datahub = datahub or MFSDataHub()

        # 緩存：key = (data_key, indicator_name, params_hash) -> IndicatorResult
        self._cache: Dict[tuple, IndicatorResult] = {}

        # 執行鎖
        self._lock = threading.RLock()

        # 緩存配置
        self._cache_enabled = True
        self._cache_ttl = 300  # 秒

    @property
    def datahub(self):
        """獲取數據中心"""
        return self._datahub

    # ==================== 指標注册 ====================

    def register_indicator(
        self,
        name: str,
        indicator_class: Type[BaseIndicator],
        params: Optional[Dict] = None
    ) -> None:
        """
        註冊指標到引擎

        Args:
            name: 指標名稱
            indicator_class: 指標類
            params: 預設參數
        """
        self._datahub.register_indicator(name, indicator_class, params)

    def register_indicator_instance(self, indicator: BaseIndicator) -> None:
        """
        註冊指標實例

        Args:
            indicator: 指標實例
        """
        BaseIndicator.register(indicator)

    # ==================== 依賴解析 ====================

    def _build_dependency_graph(
        self,
        indicator_names: List[str]
    ) -> Dict[str, Set[str]]:
        """
        構建依賴圖

        Args:
            indicator_names: 指標名稱列表

        Returns:
            {indicator_name: set of dependencies}
        """
        graph = {}

        for name in indicator_names:
            indicator_cls = self._datahub.get_indicator_class(name)
            if indicator_cls is None:
                # 嘗試從BaseIndicator注册表獲取
                indicator_cls = BaseIndicator.get_registered(name)

            if indicator_cls is None:
                raise ValueError(f"Indicator '{name}' not found")

            # 創建臨時實例以獲取依賴
            instance = indicator_cls()
            graph[name] = set(instance.dependencies)

        return graph

    def _topological_sort(self, graph: Dict[str, Set[str]]) -> List[str]:
        """
        拓撲排序（Kahn算法）

        Args:
            graph: 依賴圖

        Returns:
            排序後的指標名稱列表
        """
        # 計算入度
        in_degree = defaultdict(int)
        all_nodes = set(graph.keys())

        for node, deps in graph.items():
            for dep in deps:
                # deps是數據依賴（如 'close', 'volume'），不是指標依賴
                # 只有在graph中的才計入
                if dep in all_nodes:
                    in_degree[node] += 1

        # 初始化隊列
        queue = [node for node in all_nodes if in_degree[node] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # 減少相關節點的入度
            for other_node, deps in graph.items():
                if node in deps:
                    in_degree[other_node] -= 1
                    if in_degree[other_node] == 0:
                        queue.append(other_node)

        if len(result) != len(all_nodes):
            raise ValueError("Circular dependency detected in indicators")

        return result

    def _detect_cycles(self, indicators: List[BaseIndicator]) -> Optional[List[str]]:
        """
        檢測循環依賴

        Args:
            indicators: 指鏢列表

        Returns:
            循環路徑或None
        """
        visited = set()
        rec_stack = set()
        path = []

        def dfs(indicator: BaseIndicator) -> Optional[List[str]]:
            visited.add(indicator.name)
            rec_stack.add(indicator.name)
            path.append(indicator.name)

            for dep_name in indicator.dependencies:
                # 檢查依賴是否為其他指標
                dep_cls = self._datahub.get_indicator_class(dep_name)
                if dep_cls:
                    if dep_name not in visited:
                        result = dfs(dep_cls())
                        if result:
                            return result
                    elif dep_name in rec_stack:
                        path.append(dep_name)
                        return path

            path.pop()
            rec_stack.remove(indicator.name)
            return None

        for indicator in indicators:
            if indicator.name not in visited:
                cycle = dfs(indicator)
                if cycle:
                    return cycle

        return None

    # ==================== 緩存管理 ====================

    def _get_cache_key(
        self,
        data_key: str,
        indicator_name: str,
        params: Dict
    ) -> tuple:
        """生成緩存鍵"""
        import hashlib
        import json
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        return (data_key, indicator_name, params_hash)

    def get_cached(
        self,
        data_key: str,
        indicator_name: str,
        params: Dict
    ) -> Optional[IndicatorResult]:
        """獲取緩存結果"""
        if not self._cache_enabled:
            return None

        cache_key = self._get_cache_key(data_key, indicator_name, params)
        result = self._cache.get(cache_key)

        if result:
            # 檢查TTL
            cache_time = result.metadata.get('cache_time', 0)
            if time.time() - cache_time > self._cache_ttl:
                del self._cache[cache_key]
                return None

        return result

    def set_cache(
        self,
        data_key: str,
        indicator_name: str,
        params: Dict,
        result: IndicatorResult
    ) -> None:
        """設置緩存"""
        if not self._cache_enabled:
            return

        cache_key = self._get_cache_key(data_key, indicator_name, params)
        result.metadata['cache_time'] = time.time()
        self._cache[cache_key] = result

    def clear_cache(self) -> None:
        """清空緩存"""
        with self._lock:
            self._cache.clear()

    # ==================== 指標執行 ====================

    def compute(
        self,
        data_key: str,
        indicator_name: str,
        params: Optional[Dict] = None,
        force: bool = False
    ) -> ExecutionResult:
        """
        計算單個指標

        Args:
            data_key: 數據鍵
            indicator_name: 指鏢名稱
            params: 參數
            force: 是否強制重新計算（忽略緩存）

        Returns:
            ExecutionResult
        """
        params = params or {}
        start_time = time.time()

        # 檢查緩存
        if not force:
            cached = self.get_cached(data_key, indicator_name, params)
            if cached:
                return ExecutionResult(
                    success=True,
                    indicator_name=indicator_name,
                    result=cached,
                    compute_time=time.time() - start_time
                )

        # 獲取數據
        data = self._datahub.get_data(data_key)
        if data is None:
            return ExecutionResult(
                success=False,
                indicator_name=indicator_name,
                error=f"Data '{data_key}' not found"
            )

        # 創建指標實例
        indicator = self._datahub.create_indicator(indicator_name, params)
        if indicator is None:
            return ExecutionResult(
                success=False,
                indicator_name=indicator_name,
                error=f"Indicator '{indicator_name}' not found"
            )

        try:
            # 驗證數據
            indicator.validate(data)

            # 計算
            result_data = indicator.compute(data)

            result = IndicatorResult(
                name=indicator_name,
                data=result_data,
                params=params
            )

            # 緩存
            self.set_cache(data_key, indicator_name, params, result)

            return ExecutionResult(
                success=True,
                indicator_name=indicator_name,
                result=result,
                compute_time=time.time() - start_time
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                indicator_name=indicator_name,
                error=str(e),
                compute_time=time.time() - start_time
            )

    def compute_batch(
        self,
        data_key: str,
        indicator_names: List[str],
        params_map: Optional[Dict[str, Dict]] = None,
        force: bool = False
    ) -> Dict[str, ExecutionResult]:
        """
        批量計算多個指標

        自動解析依賴順序

        Args:
            data_key: 數據鍵
            indicator_names: 指鏢名稱列表
            params_map: 各指標的參數映射
            force: 是否強制重新計算

        Returns:
            {indicator_name: ExecutionResult}
        """
        params_map = params_map or {}
        results = {}

        # 解析依賴順序
        try:
            dep_graph = {}
            for name in indicator_names:
                indicator_cls = self._datahub.get_indicator_class(name) or \
                              BaseIndicator.get_registered(name)
                if indicator_cls:
                    instance = indicator_cls()
                    dep_graph[name] = set(instance.dependencies)

            # 拓撲排序
            sorted_names = self._topological_sort(dep_graph)

            # 只保留请求的指標
            sorted_names = [n for n in sorted_names if n in indicator_names]

        except Exception as e:
            # 如果依賴解析失敗，按原順序計算
            sorted_names = indicator_names

        # 依次計算
        for name in sorted_names:
            params = params_map.get(name, {})
            result = self.compute(data_key, name, params, force)
            results[name] = result

            # 如果失敗，不阻塞後續指標（但記錄錯誤）
            # 可根據需求改為立即返回

        return results

    # ==================== 多周期合併 ====================

    def compute_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str],
        indicator_name: str,
        params: Optional[Dict] = None
    ) -> Dict[str, ExecutionResult]:
        """
        多周期指標計算

        Args:
            symbol: 交易對
            timeframes: 時間周期列表
            indicator_name: 指鏢名稱
            params: 參數

        Returns:
            {timeframe: ExecutionResult}
        """
        results = {}

        for tf in timeframes:
            data_key = f"{symbol}_{tf}"
            result = self.compute(data_key, indicator_name, params)
            results[tf] = result

        return results

    def merge_results(
        self,
        results: List[IndicatorResult]
    ) -> IndicatorResult:
        """
        合併多個 IndicatorResult

        Args:
            results: IndicatorResult列表

        Returns:
            合併後的結果
        """
        if not results:
            raise ValueError("No results to merge")

        if len(results) == 1:
            return results[0]

        merged = results[0]
        for result in results[1:]:
            merged = merged.merge(result)

        return merged

    # ==================== 向後兼容接口 ====================

    def run_indicator(
        self,
        data: pd.DataFrame,
        indicator_name: str,
        params: Optional[Dict] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        運行指標（向後兼容介面）

        兼容舊的調用方式：engine.run_indicator(df, 'Ichimoku', period=9)

        Args:
            data: K線數據
            indicator_name: 指鏢名稱
            params: 參數
            **kwargs: 其他參數（兼容舊接口）

        Returns:
            包含指標結果的DataFrame
        """
        params = params or {}
        params.update(kwargs)

        # 嘗試創建指標
        indicator = self._datahub.create_indicator(indicator_name, params)
        if indicator is None:
            # 嘗試從BaseIndicator注册表
            cls = BaseIndicator.get_registered(indicator_name)
            if cls:
                indicator = cls(params=params)

        if indicator is None:
            raise ValueError(f"Indicator '{indicator_name}' not found")

        # 驗證並計算
        indicator.validate(data)
        return indicator.compute(data)

    def add_indicator(
        self,
        data: pd.DataFrame,
        indicator_name: str,
        params: Optional[Dict] = None,
        output_col: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        添加指標到數據（向後兼容介面）

        將指標結果追加到原DataFrame

        Args:
            data: 原數據
            indicator_name: 指鏢名稱
            params: 參數
            output_col: 輸出列名（可選，默認用指標名）
            **kwargs: 其他參數

        Returns:
            添加指標列後的DataFrame
        """
        result_df = self.run_indicator(data, indicator_name, params, **kwargs)

        # 合併結果
        if isinstance(result_df, pd.DataFrame):
            output = pd.concat([data, result_df], axis=1)
        else:
            # 結果是Series
            col_name = output_col or indicator_name
            output = data.copy()
            output[col_name] = result_df

        return output

    # ==================== 狀態查詢 ====================

    def list_indicators(self) -> List[str]:
        """列出所有可用指標"""
        return self._datahub.list_indicators()

    def get_cache_stats(self) -> Dict[str, int]:
        """獲取緩存統計"""
        return {
            'size': len(self._cache),
            'enabled': self._cache_enabled,
            'ttl': self._cache_ttl
        }

    def __repr__(self) -> str:
        return f"<IndicatorEngine(cached={len(self._cache)}, indicators={len(self.list_indicators())})>"