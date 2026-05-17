"""
Breakout200 指標包裝器
將 LW_CheckBreakout200.checkBreakout200() 適配到 BaseIndicator 接口
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from .base import BaseIndicator


class Breakout200Wrapper(BaseIndicator):
    """
    Breakout200 指標包裝器

    調用底層 LW_CheckBreakout200.checkBreakout200() 進行計算
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None, sno: Optional[str] = None, stype: Optional[str] = None):
        """
        初始化 Breakout200 包裝器

        Args:
            params: 指標參數（可選）
            sno: 股票代碼（可選）
            stype: 股票類型（可選）
        """
        super().__init__(params=params)
        self.sno = sno
        self.stype = stype

    @property
    def name(self) -> str:
        """指標名稱"""
        return "breakout200"

    @property
    def dependencies(self) -> List[str]:
        """依賴的數據字段"""
        return ["Open", "High", "Low", "Close", "Volume"]

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        執行 Breakout200 指標計算

        Args:
            data: 輸入的 K 線數據（需包含 Open, High, Low, Close, Volume）

        Returns:
            添加了 Breakout200 指標列的 DataFrame
        """
        # 驗證數據
        self.validate(data)

        # 調用底層 TA 模組函數
        from TA.LW_CheckBreakout200 import checkBreakout200

        result = checkBreakout200(df=data, sno=self.sno, stype=self.stype, params=self.params)

        return result