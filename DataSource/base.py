"""
DataSource Plugin System - Base Classes

Provides abstract base class for all data source plugins.
"""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from datetime import datetime


class DataSourceError(Exception):
    """Custom exception for data source errors."""
    pass


class BaseDataSource(ABC):
    """Abstract base class for all data source plugins.
    
    All data source implementations must inherit from this class
    and implement the required abstract methods.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Data source name (e.g., 'yahoo_finance', 'hkex', 'tencent')."""
        pass
    
    @abstractmethod
    def get_ohlcv(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        """Get OHLCV data for a given ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., '9988.HK', '700.HK')
            start: Start date in 'YYYY-MM-DD' format
            end: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
            
        Raises:
            DataSourceError: When data retrieval fails
        """
        pass
    
    @abstractmethod
    def is_available(self, ticker: str) -> bool:
        """Check if this data source supports the given ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if the ticker is supported, False otherwise
        """
        pass