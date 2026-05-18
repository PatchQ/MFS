"""
DataSource Registry - Plugin Registration and Management

Provides a singleton registry for managing pluggable data source plugins.
"""

from typing import Optional, Dict, List
import pandas as pd

from .base import BaseDataSource, DataSourceError


class DataSourceRegistry:
    """Registry for data source plugins (Singleton pattern).
    
    Manages registration, unregistration, and retrieval of data source plugins.
    Allows for easy swapping of data sources without changing downstream code.
    """
    
    _instance: Optional['DataSourceRegistry'] = None
    _sources: Dict[str, BaseDataSource]
    
    def __new__(cls) -> 'DataSourceRegistry':
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._sources = {}
        return cls._instance
    
    def register(self, source: BaseDataSource) -> None:
        """Register a data source plugin.
        
        Args:
            source: An instance of a BaseDataSource subclass
            
        Raises:
            ValueError: If a source with the same name is already registered
        """
        name = source.name
        if name in self._sources:
            raise ValueError(f"Data source '{name}' is already registered")
        self._sources[name] = source
    
    def unregister(self, name: str) -> None:
        """Unregister a data source plugin.
        
        Args:
            name: Name of the data source to unregister
            
        Raises:
            KeyError: If the specified data source is not found
        """
        if name not in self._sources:
            raise KeyError(f"Data source '{name}' is not registered")
        del self._sources[name]
    
    def get(self, name: str) -> BaseDataSource:
        """Get a registered data source by name.
        
        Args:
            name: Name of the data source
            
        Returns:
            The registered BaseDataSource instance
            
        Raises:
            KeyError: If the specified data source is not found
        """
        if name not in self._sources:
            raise KeyError(f"Data source '{name}' is not registered")
        return self._sources[name]
    
    def list_sources(self) -> List[str]:
        """List all registered data source names.
        
        Returns:
            List of registered data source names
        """
        return list(self._sources.keys())
    
    def get_ohlcv(
        self,
        ticker: str,
        source: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        """Get OHLCV data using a specified or auto-selected data source.
        
        Args:
            ticker: Stock ticker symbol
            source: Specific data source name (None = auto-select based on ticker)
            start: Start date in 'YYYY-MM-DD' format
            end: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
            
        Raises:
            DataSourceError: When data retrieval fails or no suitable source found
        """
        if source is not None:
            # Use specified data source
            ds = self.get(source)
            return ds.get_ohlcv(ticker, start, end)
        
        # Auto-select: try each registered source that supports this ticker
        for name, ds in self._sources.items():
            if ds.is_available(ticker):
                return ds.get_ohlcv(ticker, start, end)
        
        raise DataSourceError(
            f"No available data source found for ticker '{ticker}'"
        )