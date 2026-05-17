"""
Fisher Transform 指標包裝器
將 LW_CheckFisher.checkFisher() 適配到 BaseIndicator 接口
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from .base import BaseIndicator


class FisherWrapper(BaseIndicator):
    """
    Fisher Transform 指標包裝器

    調用底層 LW_CheckFisher.checkFisher() 進行計算
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        初始化 Fisher 包裝器

        Args:
            params: 指標參數（可選）
        """
        super().__init__(params=params)

    @property
    def name(self) -> str:
        """指標名稱"""
        return "fisher"

    @property
    def dependencies(self) -> List[str]:
        """依賴的數據字段"""
        return ["Open", "High", "Low", "Close", "Volume"]

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        執行 Fisher Transform 指標計算

        Args:
            data: 輸入的 K 線數據（需包含 Open, High, Low, Close, Volume）

        Returns:
            添加了 Fisher 指標列的 DataFrame
        """
        # 驗證數據
        self.validate(data)

        # 調用底層 TA 模組函數
        from TA.LW_CheckFisher import checkFisher

        result = checkFisher(
            df=data.copy(),
            sno=self.params.get("sno", "UNKNOWN"),
            stype=self.params.get("stype", "common"),
            params=self.params.get("fisher_params", None)
        )

        return result