"""
DataSource Plugin System

Unified interface for financial data sources with pluggable architecture.
"""

import pandas as pd

from .base import BaseDataSource, DataSourceError
from .registry import DataSourceRegistry
from .yahoo_finance import YahooFinanceSource
from .hkex import HKEXSource
from .tencent import TencentSource

__all__ = [
    'BaseDataSource',
    'DataSourceError',
    'DataSourceRegistry',
    'YahooFinanceSource',
    'HKEXSource',
    'TencentSource',
    'get_ohlcv',
    'list_sources',
]

# Create and populate global registry instance
_registry = DataSourceRegistry()
_registry.register(YahooFinanceSource())
_registry.register(HKEXSource())
_registry.register(TencentSource())

# Convenience functions that delegate to the global registry
def get_ohlcv(
    ticker: str,
    source: str | None = None,
    start: str | None = None,
    end: str | None = None
) -> pd.DataFrame:
    """Get OHLCV data for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        source: Specific data source name (optional)
        start: Start date 'YYYY-MM-DD' (optional)
        end: End date 'YYYY-MM-DD' (optional)
        
    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    return _registry.get_ohlcv(ticker, source, start, end)


def list_sources() -> list:
    """List all registered data sources.
    
    Returns:
        List of registered data source names
    """
    return _registry.list_sources()