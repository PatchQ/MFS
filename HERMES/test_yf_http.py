"""測試：yfinance 在 StockPrice 環境中的 HTTP 請求"""
import sys; sys.path.insert(0, '/root/GitHub/MFS')

import UTIL.CommonConfig as cc
from datetime import datetime

import curl_cffi.requests as cr
orig_request = cr.Session.request

def patched_request(self, method, url, **kwargs):
    print(f'[HTTP] {method} {url}', flush=True)
    if 'chart' in url:
        print(f'  params={kwargs.get("params")}', flush=True)
    return orig_request(self, method, url, **kwargs)

cr.Session.request = patched_request

print('=== 測試 0700.HK ===', flush=True)
ticker = cc.yf.Ticker('0700.HK')
data = ticker.history(start='2026-04-30', auto_adjust=False)
print(f'Result: empty={data.empty}, len={len(data)}', flush=True)
print(data, flush=True)
