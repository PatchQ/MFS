"""
VCP (Volatility Contraction Pattern) 指標包裝器
將 LW_CheckVCP.checkVCP() 適配到 BaseIndicator 接口
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from .base import BaseIndicator


class VCPWrapper(BaseIndicator):
    """
    VCP 指標包裝器

    調用底層 LW_CheckVCP.checkVCP() 進行計算
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        初始化 VCP 包裝器

        Args:
            params: 指標參數（可選）
        """
        super().__init__(params=params)

    @property
    def name(self) -> str:
        """指標名稱"""
        return "vcp"

    @property
    def dependencies(self) -> List[str]:
        """依賴的數據字段"""
        return ["Open", "High", "Low", "Close", "Volume"]

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        執行 VCP 指標計算

        Args:
            data: 輸入的 K 線數據（需包含 Open, High, Low, Close, Volume）

        Returns:
            添加了 VCP 指標列的 DataFrame
        """
        # 驗證數據
        self.validate(data)

        # 調用底層 TA 模組函數
        from TA.LW_CheckVCP import checkVCP

        result = checkVCP(df=data.copy())

        return result