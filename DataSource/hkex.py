"""
HKEX Data Source Plugin

Fetches OHLCV data from HKEX (Hong Kong Stock Exchange).
Supports stocks and options (SO, IO).
"""

import requests
from typing import Optional
import pandas as pd
from datetime import datetime
import time
import re

from .base import BaseDataSource, DataSourceError


class HKEXSource(BaseDataSource):
    """HKEX data source implementation."""
    
    # HKEX API endpoints
    STOCK_URL = "https://www.hkex.com.hk/eng/stat/smstat/day quotation/d_{date}e.htm"
    OPTIONS_URL = "https://www.hkex.com.hk/eng/stat/smstat/day quotation/so/{date}e.htm"
    
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    @property
    def name(self) -> str:
        return "hkex"
    
    def is_available(self, ticker: str) -> bool:
        """Check if HKEX supports this ticker.
        
        HKEX supports:
        - HK stocks: 5-digit codes like '9988', '0700'
        - Stock options: SO codes like 'SO 9988', 'SO 0700'
        - Index options: IO codes like 'IO HSI', 'IO HHI'
        """
        if not ticker:
            return False
        
        # Stock options pattern: SO <code>
        if ticker.upper().startswith('SO '):
            code = ticker[3:].strip()
            return code.isdigit() and 4 <= len(code) <= 5
        
        # Index options pattern: IO <name>
        if ticker.upper().startswith('IO '):
            return True
        
        # Regular HK stock: 4-5 digits
        return ticker.isdigit() and 4 <= len(ticker) <= 5
    
    def _convert_ticker(self, ticker: str) -> str:
        """Convert ticker format for HKEX API."""
        # Remove .HK suffix if present
        ticker = re.sub(r'\.HK$', '', ticker, flags=re.IGNORECASE)
        return ticker.zfill(5)  # Pad to 5 digits
    
    def get_ohlcv(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """Fetch OHLCV data from HKEX.
        
        Args:
            ticker: Stock/option ticker
            start: Start date 'YYYY-MM-DD' (for future use)
            end: End date 'YYYY-MM-DD' (for future use)
            max_retries: Number of retry attempts on failure
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        # Determine if it's stock option, index option, or regular stock
        ticker_upper = ticker.upper()
        
        if ticker_upper.startswith('SO '):
            return self._get_stock_option_ohlcv(ticker, max_retries)
        elif ticker_upper.startswith('IO '):
            return self._get_index_option_ohlcv(ticker, max_retries)
        else:
            return self._get_stock_ohlcv(ticker, max_retries)
    
    def _get_stock_ohlcv(self, ticker: str, max_retries: int) -> pd.DataFrame:
        """Fetch regular stock OHLCV from HKEX."""
        stock_code = self._convert_ticker(ticker)
        date_str = datetime.now().strftime('%Y%m%d')
        
        for attempt in range(max_retries):
            try:
                # HKEX provides daily quotes in HTML format
                url = self.STOCK_URL.format(date=date_str)
                response = requests.get(url, headers=self.HEADERS, timeout=10)
                response.raise_for_status()
                
                # Parse HTML to find the stock data
                # This is a simplified implementation
                # In production, you would use BeautifulSoup for proper parsing
                html_content = response.text
                
                # Find the stock entry in the HTML table
                pattern = rf'{stock_code}</td><td>([^<]+)</td><td>([^<]+)</td><td>([^<]+)</td><td>([^<]+)</td><td>([^<]+)</td><td>([^<]+)</td>'
                match = re.search(pattern, html_content)
                
                if match:
                    close, open_price, high, low, volume, turnover = match.groups()
                    
                    df = pd.DataFrame({
                        'Date': [datetime.now().strftime('%Y-%m-%d')],
                        'Open': [float(open_price)],
                        'High': [float(high)],
                        'Low': [float(low)],
                        'Close': [float(close)],
                        'Volume': [int(volume.replace(',', ''))]
                    })
                    return df
                
                raise DataSourceError(f"Stock {ticker} not found in HKEX data")
                
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise DataSourceError(
                        f"Failed to fetch stock data from HKEX for '{ticker}': {e}"
                    )
                time.sleep(1)
        
        raise DataSourceError(f"Max retries exceeded for ticker '{ticker}'")
    
    def _get_stock_option_ohlcv(self, ticker: str, max_retries: int) -> pd.DataFrame:
        """Fetch stock option OHLCV from HKEX."""
        # Extract stock code from SO <code> format
        stock_code = ticker[3:].strip()
        date_str = datetime.now().strftime('%Y%m%d')
        
        for attempt in range(max_retries):
            try:
                url = self.OPTIONS_URL.format(date=date_str)
                response = requests.get(url, headers=self.HEADERS, timeout=10)
                response.raise_for_status()
                
                # Parse options data
                html_content = response.text
                
                # Pattern for stock options: SO <code>
                pattern = rf'SO {stock_code}</td><td>([^<]+)</td><td>([^<]+)</td><td>([^<]+)</td><td>([^<]+)</td><td>([^<]+)</td>'
                match = re.search(pattern, html_content)
                
                if match:
                    close, open_price, high, low, volume = match.groups()
                    
                    df = pd.DataFrame({
                        'Date': [datetime.now().strftime('%Y-%m-%d')],
                        'Open': [float(open_price)],
                        'High': [float(high)],
                        'Low': [float(low)],
                        'Close': [float(close)],
                        'Volume': [int(volume.replace(',', ''))]
                    })
                    return df
                
                raise DataSourceError(f"Stock option {ticker} not found in HKEX data")
                
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise DataSourceError(
                        f"Failed to fetch SO data from HKEX for '{ticker}': {e}"
                    )
                time.sleep(1)
        
        raise DataSourceError(f"Max retries exceeded for ticker '{ticker}'")
    
    def _get_index_option_ohlcv(self, ticker: str, max_retries: int) -> pd.DataFrame:
        """Fetch index option OHLCV from HKEX."""
        # Extract index name from IO <name> format
        index_name = ticker[3:].strip().upper()
        date_str = datetime.now().strftime('%Y%m%d')
        
        for attempt in range(max_retries):
            try:
                url = self.OPTIONS_URL.format(date=date_str)
                response = requests.get(url, headers=self.HEADERS, timeout=10)
                response.raise_for_status()
                
                html_content = response.text
                
                # Pattern for index options: IO <name>
                pattern = rf'IO {index_name}</td><td>([^<]+)</td><td>([^<]+)</td><td>([^<]+)</td><td>([^<]+)</td><td>([^<]+)</td>'
                match = re.search(pattern, html_content)
                
                if match:
                    close, open_price, high, low, volume = match.groups()
                    
                    df = pd.DataFrame({
                        'Date': [datetime.now().strftime('%Y-%m-%d')],
                        'Open': [float(open_price)],
                        'High': [float(high)],
                        'Low': [float(low)],
                        'Close': [float(close)],
                        'Volume': [int(volume.replace(',', ''))]
                    })
                    return df
                
                raise DataSourceError(f"Index option {ticker} not found in HKEX data")
                
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise DataSourceError(
                        f"Failed to fetch IO data from HKEX for '{ticker}': {e}"
                    )
                time.sleep(1)
        
        raise DataSourceError(f"Max retries exceeded for ticker '{ticker}'")