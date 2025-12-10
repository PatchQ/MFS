import pandas as pd
import requests
from io import StringIO
import time
import yfinance as yf
from datetime import datetime
import os


def get_all_us_stocks():
    all_tickers = []
    
    # 1. NASDAQ
    print("Get NASDAQ List...")
    nasdaq_tickers = get_nasdaq_tickers()
    all_tickers.extend(nasdaq_tickers)
    print(f"NASDAQ: {len(nasdaq_tickers)}")
    
    # 2. NYSE
    print("Get NYSE List...")
    nyse_tickers = get_nyse_tickers()
    all_tickers.extend(nyse_tickers)
    print(f"NYSE: {len(nyse_tickers)}")
    
    # 3. AMEX
    print("Get AMEX List...")
    amex_tickers = get_amex_tickers()
    all_tickers.extend(amex_tickers)
    print(f"AMEX: {len(amex_tickers)}")
    
    # unique
    unique_tickers = list(set(all_tickers))
    print(f"\nAll {len(unique_tickers)}")
    
    return unique_tickers

def get_nasdaq_tickers():
    """get NASDAQ from nasdaqtrader website"""
    try:
        url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers)
        
        # 解析文本数据
        data = response.text
        lines = data.split('\n')[1:-1]  # 跳过标题行和最后空行
        
        tickers = []
        for line in lines:
            if '|' in line:
                ticker = line.split('|')[0].strip()
                market = line.split('|')[2].strip()
                etf = line.split('|')[6].strip()
                # 过滤掉基金和权证
                #if not any(x in ETF for x in ['FUND', 'ETF', 'TRUST', 'PORTFOLIO']):
                if etf=="N":
                    tickers.append(ticker+"|"+market)
        
        return tickers
    except Exception as e:
        print(f"Get NASDAQ List fail: {e}")
        return []

def get_nyse_tickers():    
    try:
        url = "https://www.nyse.com/api/quotes/filter"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Content-Type': 'application/json'
        }
        
        # NYSE 需要 POST 请求
        payload = {
            "instrumentType": "EQUITY",
            "pageNumber": 1,
            "sortColumn": "NORMALIZED_TICKER",
            "sortOrder": "ASC",
            "maxResultsPerPage": 10000,
            "filterToken": ""
        }
        
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()        

        df = pd.DataFrame(data)        

        df.to_csv("Data/nyse_stock_list.csv", index=False)
        
        mask = df['instrumentName'].str.contains('FUND', case=False, na=False)
        #tickers = [item['normalizedTicker'] for item in data if 'normalizedTicker' in item]
        fdf = df[~mask]
        
        return fdf['normalizedTicker']
        
    except Exception as e:
        print(f"get NYSE List Fail: {e}")        
        return []


def get_amex_tickers():
    """从 AMEX 获取股票列表"""
    try:
        url = "http://www.nasdaqtrader.com/dynamic/SymDir/amex-listed.txt"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers)
        
        data = response.text
        lines = data.split('\n')[1:-1]
        
        tickers = []
        for line in lines:
            if '|' in line:
                ticker = line.split('|')[0].strip()
                if not any(x in ticker for x in ['^', '$']):
                    tickers.append(ticker)
        
        return tickers
    except Exception as e:
        print(f"get AMEX List fail: {e}")
        return []
    
def get_stocks_from_third_party():

    all_tickers = []
    
    #get from GitHub
    #print("get from GitHub...")
    #github_tickers = get_from_github()
    #all_tickers.extend(github_tickers)
        
    #get from other website
    print("get from other website...")
    other_tickers = get_from_other_sources()
    all_tickers.extend(other_tickers)
    
    #unique number
    unique_tickers = list(set([t.upper() for t in all_tickers if t and len(t) <= 5]))
    print(f"get {len(unique_tickers)} numbers")
    
    return unique_tickers

def get_from_github():
    try:
        #public list from github
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        response = requests.get(url)
        tickers = response.text.strip().split('\n')
        return tickers
    except:
        return []


def get_from_other_sources():    
    tickers = []
    
    #website : ETF.com
    try:
        url = "https://etfdb.com/compare/volume/"
        tables = pd.read_html(url)
        if tables:
            etf_table = tables[0]
            tickers.extend(etf_table['Symbol'].tolist())
    except:
        pass
    
    #website : slickcharts.com
    try:
        url = "https://www.slickcharts.com/sp500"
        tables = pd.read_html(url)
        if tables:
            sp500_table = tables[0]
            tickers.extend(sp500_table['Symbol'].tolist())
    except:
        pass

    try:
        url = "https://www.slickcharts.com/nasdaq100"
        tables = pd.read_html(url)
        if tables:
            nasdaq100_table = tables[0]
            tickers.extend(nasdaq100_table['Symbol'].tolist())
    except:
        pass

    try:
        url = "https://www.slickcharts.com/dowjones"
        tables = pd.read_html(url)
        if tables:
            dowjones_table = tables[0]
            tickers.extend(dowjones_table['Symbol'].tolist())
    except:
        pass

    
    return tickers



class USStockListFetcher:
    def __init__(self, cache_days=1):
        self.cache_days = cache_days
        self.cache_file = "Data/us_stock_list_cache.csv"
    
    def get_complete_stock_list(self, use_cache=True):

        # 检查缓存
        if use_cache and self._is_cache_valid():
            print("use cache...")
            return self._load_from_cache()
        
        print("get new us list...")
        
                
        exchange_tickers = get_all_us_stocks()
        third_party_tickers = get_stocks_from_third_party()
        
        all_tickers = list(set(exchange_tickers + third_party_tickers))
        print(f"\nAll: {len(all_tickers)} numbers")
        
        # 基本清理
        #cleaned_tickers = self._clean_tickers(all_tickers)
        #print(f"clean: {len(cleaned_tickers)} numebrs")
        
        # 验证部分股票代码（可选）
        print("\nValidation...")
        #validated_tickers = self._validate_tickers(cleaned_tickers, sample_size=10)
        validated_tickers = all_tickers
        
        # 保存到缓存
        self._save_to_cache(validated_tickers)    
        
        return validated_tickers
    
    def _clean_tickers(self, tickers):
        
        cleaned = []
        
        for ticker in tickers:
            if not ticker:
                continue
            
            # 转换为字符串并去空格
            ticker = str(ticker).strip().upper()
            
            # 过滤条件
            if (len(ticker) <= 5 and  # 股票代码通常1-5个字符
                not any(char in ticker for char in ['/', '\\', ' ', '.', '^', '$']) and
                ticker not in ['NA', 'N/A', 'NULL']):
                cleaned.append(ticker)
        
        return list(set(cleaned))
    
    def _validate_tickers(self, tickers, sample_size=10):
        
        validated = []
        print(f"random check {min(sample_size, len(tickers))} numbers...")
        
        # 随机抽样验证
        import random
        sample_tickers = random.sample(tickers, min(sample_size, len(tickers)))
        
        for i, ticker in enumerate(sample_tickers):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # 检查是否有基本数据
                if info and 'symbol' in info:
                    validated.append(ticker)
                    print(f"  ✓ {ticker} 有效 ({i+1}/{len(sample_tickers)})")
                else:
                    print(f"  ✗ {ticker} 无效 ({i+1}/{len(sample_tickers)})")
                
                # 避免请求过于频繁
                if i % 10 == 0:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"  ✗ {ticker} 验证失败: {e} ({i+1}/{len(sample_tickers)})")
        
        valid_ratio = len(validated) / len(sample_tickers)
        print(f"pass ratio: {valid_ratio:.1%}")
        
        # 如果验证通过率足够高，返回所有股票代码
        if valid_ratio >= 0.8:
            return tickers
        else:
            # 如果通过率低，只返回验证通过的
            return validated
    
    def _is_cache_valid(self):
        """检查缓存是否有效"""
        if not os.path.exists(self.cache_file):
            return False
        
        file_time = os.path.getmtime(self.cache_file)
        cache_age = (time.time() - file_time) / (60 * 60 * 24)  # 转换为天数
        
        return cache_age <= self.cache_days
    
    def _load_from_cache(self):        
        try:
            df = pd.read_csv(self.cache_file)
            return df['Ticker'].tolist()
        except:
            return []
    
    def _save_to_cache(self, tickers):
        
        try:
            df = pd.DataFrame({'Ticker': tickers})
            df.to_csv(self.cache_file, index=False)
            print(f"save list to : {self.cache_file}")
        except Exception as e:
            print(f"save fail: {e}")
    
    def export_stock_list(self, filename=None):
        
        if filename is None:            
            filename = "Data/us_stock_list.csv"

        tickers = self.get_complete_stock_list()
        
        df = pd.DataFrame({
            'Ticker': tickers,
            'Exchange': '',  
            'Company': '',   
            'Last_Updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        df.to_csv(filename, index=False)
        print(f"Export to : {filename}")
        
        return filename


def main():
    
    fetcher = USStockListFetcher(cache_days=0)
    
    print("Start to get the list...")
    #us_stocks = fetcher.get_complete_stock_list()
    
    #print(f"\got {len(us_stocks)} numbers")
            
    export_file = fetcher.export_stock_list()
    
    

if __name__ == "__main__":
    stocks = main()