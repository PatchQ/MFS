"""
HFH (High-Flat-High) 形態指標包裝器
將 LW_CheckHFH.calHFH() 適配到 BaseIndicator 接口
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from .base import BaseIndicator


class HFHWrapper(BaseIndicator):
    """
    HFH (High-Flat-High) 形態指標包裝器

    調用底層 LW_CheckHFH.calHFH() 進行計算
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        初始化 HFH 包裝器

        Args:
            params: 指標參數（可選）
        """
        super().__init__(params=params)

    @property
    def name(self) -> str:
        """指標名稱"""
        return "hf_h"

    @property
    def dependencies(self) -> List[str]:
        """
        依賴的數據字段

        HFH 指標需要標準 OHLCV 數據以及均線 EMA22/50/100/250
        """
        return ["Open", "High", "Low", "Close", "Volume", "EMA22", "EMA50", "EMA100", "EMA250"]

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        執行 HFH 指標計算

        Args:
            data: 輸入的 K 線數據（需包含 Open, High, Low, Close, Volume 及均線）

        Returns:
            添加了 HFH 指標列的 DataFrame
        """
        # 驗證數據
        self.validate(data)

        # 調用底層 TA 模組函數
        from TA.LW_CheckHFH import calHFH

        result = calHFH(
            df=data.copy(),
            # 從 params 取出自定義參數，使用預設值
            min_strong_bullish=self.params.get("min_strong_bullish", 2),
            body_ratio=self.params.get("body_ratio", 0.5),
            require_consecutive_higher=self.params.get("require_consecutive_higher", False),
            min_flat_length=self.params.get("min_flat_length", 4),
            max_flat_pct=self.params.get("max_flat_pct", 0.12),
            max_body_deviation=self.params.get("max_body_deviation", 5.0),
            min_flat_body_ratio=self.params.get("min_flat_body_ratio", 0.0),
            min_close_strength=self.params.get("min_close_strength", 0.5),
            max_upper_wick=self.params.get("max_upper_wick", 0.35),
            min_volume_ratio=self.params.get("min_volume_ratio", 1.2),
            next_day_confirm=self.params.get("next_day_confirm", False),
            next_day_max_drop=self.params.get("next_day_max_drop", 0.03),
            use_dynamic_flat_pct=self.params.get("use_dynamic_flat_pct", False),
            atr_period=self.params.get("atr_period", 14),
            atr_flat_multiplier=self.params.get("atr_flat_multiplier", 1.5)
        )

        return result