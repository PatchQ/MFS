"""
Tencent Data Source Plugin

Fetches OHLCV data from Tencent Finance API.
Compatible with existing HEX/StockPrice.py implementation.
"""

import requests
from typing import Optional
import pandas as pd
from datetime import datetime
import time

from .base import BaseDataSource, DataSourceError


class TencentSource(BaseDataSource):
    """Tencent Finance data source implementation."""
    
    # Tencent Finance API endpoints
    BASE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    # Market code mapping for Tencent
    MARKET_MAP = {
        'hk': 'hk',      # Hong Kong
        'sh': 'sh',      # Shanghai
        'sz': 'sz',      # Shenzhen
    }
    
    @property
    def name(self) -> str:
        return "tencent"
    
    def is_available(self, ticker: str) -> bool:
        """Check if Tencent Finance supports this ticker.
        
        Tencent supports:
        - HK stocks: 'hk001988', 'hk00700', etc.
        - SH stocks: 'sh600519', 'sh601318', etc.
        - SZ stocks: 'sz000858', 'sz002594', etc.
        """
        if not ticker:
            return False
        
        ticker_lower = ticker.lower()
        
        # Check format: <market><code>
        if ticker_lower.startswith('hk'):
            code = ticker_lower[2:]
            return code.isdigit() and len(code) >= 4
        elif ticker_lower.startswith('sh') or ticker_lower.startswith('sz'):
            code = ticker_lower[2:]
            return code.isdigit() and len(code) == 6
        
        return False
    
    def _convert_ticker(self, ticker: str) -> tuple:
        """Convert ticker to Tencent format.
        
        Returns:
            tuple: (market_code, stock_code)
        """
        ticker_lower = ticker.lower()
        
        if ticker_lower.startswith('hk'):
            return 'hk', ticker_lower[2:]
        elif ticker_lower.startswith('sh'):
            return 'sh', ticker_lower[2:]
        elif ticker_lower.startswith('sz'):
            return 'sz', ticker_lower[2:]
        else:
            # Assume HK stock if just digits
            return 'hk', ticker_lower
    
    def get_ohlcv(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        max_retries: int = 3
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Tencent Finance.
        
        Args:
            ticker: Stock ticker (e.g., 'hk001988', 'sh600519', 'sz000858')
            start: Start date 'YYYY-MM-DD'
            end: End date 'YYYY-MM-DD'
            max_retries: Number of retry attempts on failure
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        market, code = self._convert_ticker(ticker)
        
        # Determine timeframe based on date range
        fq = 'day'
        if start and end:
            start_date = datetime.strptime(start, '%Y-%m-%d')
            end_date = datetime.strptime(end, '%Y-%m-%d')
            days = (end_date - start_date).days
            
            if days <= 7:
                fq = 'week'  # Use weekly for very short ranges
            elif days <= 30:
                fq = 'month'
        
        params = {
            '_var': 'kline_dayqfq',
            'param': f'{market}{code},day,{start or "2020-01-01"},{end or datetime.now().strftime("%Y-%m-%d")},200,qafq'
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    self.BASE_URL,
                    params=params,
                    headers=self.HEADERS,
                    timeout=10
                )
                response.raise_for_status()
                
                # Parse response - Tencent returns JavaScript-like format
                text = response.text
                
                # Extract JSON from JavaScript wrapper
                # Format: var kline_dayqfq={...}
                import json
                import re
                
                json_match = re.search(r'=\s*(\{.*\})', text)
                if not json_match:
                    raise DataSourceError(f"Invalid response format from Tencent for '{ticker}'")
                
                data = json.loads(json_match.group(1))
                
                # Extract data
                qfqday = data.get('data', {}).get(f'{market}{code}', {}).get('qfqday', [])
                
                if not qfqday:
                    # Try non-qfq data
                    day_data = data.get('data', {}).get(f'{market}{code}', {}).get('day', [])
                    if not day_data:
                        raise DataSourceError(f"No data available for '{ticker}' from Tencent")
                    qfqday = day_data
                
                # Convert to DataFrame
                records = []
                for item in qfqday:
                    if len(item) >= 6:
                        records.append({
                            'Date': item[0],
                            'Open': float(item[1]),
                            'High': float(item[2]),
                            'Low': float(item[3]),
                            'Close': float(item[4]),
                            'Volume': int(float(item[5]))
                        })
                
                if not records:
                    raise DataSourceError(f"No records found for '{ticker}'")
                
                df = pd.DataFrame(records)
                
                # Filter by date range if specified
                if start:
                    df = df[df['Date'] >= start]
                if end:
                    df = df[df['Date'] <= end]
                
                df.reset_index(drop=True, inplace=True)
                return df
                
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise DataSourceError(
                        f"Failed to fetch data from Tencent for '{ticker}': {e}"
                    )
                time.sleep(1)
                
            except (KeyError, ValueError, IndexError) as e:
                raise DataSourceError(
                    f"Failed to parse Tencent response for '{ticker}': {e}"
                )
        
        raise DataSourceError(f"Max retries exceeded for ticker '{ticker}'")
    
    @staticmethod
    def get_price(ticker: str, format='df') -> any:
        """Legacy compatibility method matching HEX/StockPrice.py get_price().
        
        Args:
            ticker: Stock ticker
            format: 'df' for DataFrame, 'dict' for dictionary
            
        Returns:
            DataFrame or dict with price data
        """
        instance = TencentSource()
        df = instance.get_ohlcv(ticker)
        
        if format == 'df':
            return df
        elif format == 'dict':
            return df.to_dict('records')
        else:
            return df