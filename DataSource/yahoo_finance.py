"""
Yahoo Finance Data Source Plugin

Fetches OHLCV data from Yahoo Finance API.
"""

import requests
from typing import Optional
import pandas as pd
from datetime import datetime
import time

from .base import BaseDataSource, DataSourceError


class YahooFinanceSource(BaseDataSource):
    """Yahoo Finance data source implementation."""
    
    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    @property
    def name(self) -> str:
        return "yahoo_finance"
    
    def is_available(self, ticker: str) -> bool:
        """Check if Yahoo Finance supports this ticker.
        
        Yahoo Finance typically supports:
        - US stocks: 'AAPL', 'MSFT', etc.
        - HK stocks: '9988.HK', '700.HK', etc.
        - Chinese stocks: '9988.HK', '600519.SS', etc.
        """
        # Accept any ticker format that looks reasonable
        if not ticker or len(ticker) < 1:
            return False
        # Basic validation: should not contain spaces and reasonable length
        return ' ' not in ticker and len(ticker) <= 20
    
    def get_ohlcv(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker (e.g., '9988.HK', 'AAPL')
            start: Start date 'YYYY-MM-DD'
            end: End date 'YYYY-MM-DD'
            max_retries: Number of retry attempts on failure
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        # Convert dates to timestamps
        period1 = None
        period2 = None
        
        if start:
            period1 = int(datetime.strptime(start, '%Y-%m-%d').timestamp())
        if end:
            period2 = int(datetime.strptime(end, '%Y-%m-%d').timestamp())
        else:
            # 如果沒有提供 end，默認使用今天
            period2 = int(datetime.now().timestamp())
        
        url = self.BASE_URL.format(ticker=ticker)
        params = {'interval': '1d'}  # Daily candles
        if period1:
            params['period1'] = period1
        if period2:
            params['period2'] = period2
        
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=self.HEADERS,
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                
                # Parse the response
                result = data.get('chart', {}).get('result', [])
                if not result:
                    raise DataSourceError(f"No data returned for ticker '{ticker}'")
                
                quote = result[0].get('indicators', {}).get('quote', [{}])[0]
                timestamps = result[0].get('timestamp', [])
                
                if not timestamps or not quote:
                    raise DataSourceError(f"No price data available for '{ticker}'")
                
                # Build DataFrame
                df = pd.DataFrame({
                    'Date': pd.to_datetime(timestamps, unit='s').strftime('%Y-%m-%d'),
                    'Open': quote.get('open', []),
                    'High': quote.get('high', []),
                    'Low': quote.get('low', []),
                    'Close': quote.get('close', []),
                    'Volume': quote.get('volume', [])
                })
                
                # Filter out rows with NaN values
                df.dropna(subset=['Close', 'Volume'], inplace=True)
                df.reset_index(drop=True, inplace=True)
                
                return df
                
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise DataSourceError(
                        f"Failed to fetch data from Yahoo Finance for '{ticker}': {e}"
                    )
                time.sleep(1)  # Wait before retry
                
            except (KeyError, ValueError, IndexError) as e:
                raise DataSourceError(
                    f"Failed to parse Yahoo Finance response for '{ticker}': {e}"
                )
        
        raise DataSourceError(f"Max retries exceeded for ticker '{ticker}'")