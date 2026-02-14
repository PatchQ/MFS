import pandas as pd
import time as t
from datetime import datetime, timedelta
import yfinance as yf
import concurrent.futures as cf
from tqdm import tqdm
import os
from tqdm import tqdm
import logging
from typing import List
from curl_cffi import requests


PATH = "../SData/USData/"
#SDATE = "2024-01-01"
SDATE = "1980-01-01"
EDATE = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def getData(sno):        
    ticker = yf.Ticker(sno)
    outputlist = ticker.history(period="max",auto_adjust=True)
    outputlist.index = pd.to_datetime(pd.to_datetime(outputlist.index).strftime('%Y%m%d'))
    outputlist = outputlist[outputlist['Volume'] > 0]
    outputlist.insert(0,"sno", sno)    
    outputlist.to_csv(PATH+"/"+sno+".csv")

def safe_getData(ticker):
    try:
        return getData(ticker)
    except Exception as e:
        logger.error(f"Error processing {ticker}: {str(e)}")
        return None
    

def USgetAll():
    STOCKLIST = pd.read_csv("Data/us_stock_list.csv",dtype=str)
    #INDEXLIST = pd.Series(["^HSI","^DJI","^IXIC","^GSPC","^N225","^FTSE","^GDAXI","^FCHI","000001.SS","399001.SZ"])
    SLIST = STOCKLIST[["Ticker"]]    
    SLIST = SLIST[:]

    with cf.ThreadPoolExecutor(max_workers=5) as executor:
        list(tqdm(executor.map(getData,SLIST["Ticker"],chunksize=1),total=len(SLIST)))


def getData(sno, max_retries=3, base_delay=2):
    """
    使用重试机制获取单个ticker的数据:cite[3]
    
    参数:
        ticker: 股票代码
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒），采用指数退避:cite[9]
    
    返回:
        成功则返回数据，否则返回None
    """
    # 为每个Ticker创建一个独立的session，有助于提高线程安全性和连接复用:cite[1]
    session = requests.Session(impersonate="chrome")
    # 可以考虑在此处随机更换User-Agent:cite[9]
    
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(sno, session=session)
            outputlist = ticker.history(period="max",auto_adjust=True)
            outputlist.index = pd.to_datetime(pd.to_datetime(outputlist.index).strftime('%Y%m%d'))
            outputlist = outputlist[outputlist['Volume'] > 0]
            outputlist.insert(0,"sno", sno)    
            outputlist.to_csv(PATH+"/"+sno+".csv")
            
            if outputlist.empty:
                logger.warning(f"获取到空数据: {ticker}")
                return None
                
            return outputlist

        except Exception as e:
            error_msg = str(e).lower()
            wait_time = base_delay * (2 ** attempt)  # 指数退避:cite[9]
            
            # 检查是否为速率限制错误:cite[3]
            if 'too many requests' in error_msg or 'rate limit' in error_msg or '429' in error_msg:
                logger.warning(f"速率限制触发 [{ticker}], 尝试 {attempt + 1}/{max_retries}, 等待 {wait_time}秒")
                t.sleep(wait_time)
            # 检查连接相关错误
            elif any(err in error_msg for err in ['connection', 'timeout', 'request']):
                logger.warning(f"网络错误 [{ticker}], 尝试 {attempt + 1}/{max_retries}, 等待 {wait_time}秒")
                t.sleep(wait_time)
            else:
                logger.error(f"未知错误 [{ticker}]: {e}")
                # 对于未知错误，可以选择立即失败或也进行重试
                # 这里我们选择也进行重试
                t.sleep(wait_time)
                
    logger.error(f"所有重试均失败: {ticker}")
    return None

def process_batch(tickers_batch, batch_num, total_batches, batch_delay=5, request_delay=1):
    """
    处理单个批次的数据
    
    参数:
        tickers_batch: 当前批次的ticker列表
        batch_num: 当前批次序号
        total_batches: 总批次数
        batch_delay: 批次间的延迟（秒）
        request_delay: 单个批次内请求间的延迟（秒）:cite[2]
    """
    batch_results = []
    
    logger.info(f"开始处理批次 {batch_num}/{total_batches}, 包含 {len(tickers_batch)} 个ticker")
    
    for i, ticker in enumerate(tickers_batch):
        # 获取单个ticker数据
        result = getData(ticker)
        if result is not None:
            batch_results.append((ticker, result))
        
        # 在同一批次内，如果不是最后一个请求，则添加请求间延迟:cite[2]
        if i < len(tickers_batch) - 1 and request_delay > 0:
            t.sleep(request_delay)
    
    logger.info(f"批次 {batch_num} 完成, 成功获取 {len(batch_results)}/{len(tickers_batch)} 个ticker数据")
    return batch_results

def USgetAll_optimized(csv_path="Data/us_stock_list.csv", 
                         batch_size=30, 
                         max_workers=5, 
                         batch_delay=10,
                         request_delay=1):
    """
    优化版本的批量获取股票数据函数，避免触发yfinance限制
    
    参数:
        csv_path: 股票列表CSV文件路径
        batch_size: 每批次处理的股票数量:cite[1]
        max_workers: 线程池最大工作线程数:cite[1]
        batch_delay: 批次间的延迟时间（秒）
        request_delay: 单个批次内请求间的延迟（秒）:cite[2]
    """
    try:
        # 读取股票列表
        STOCKLIST = pd.read_csv(csv_path, dtype=str)
        SLIST = STOCKLIST[["Ticker"]].dropna()
        tickers = SLIST["Ticker"].tolist()
        
        logger.info(f"开始处理 {len(tickers)} 只股票，批次大小: {batch_size}, 工作线程: {max_workers}")
        
        # 配置yfinance缓存:cite[6]
        # 可以将缓存设置到SSD或特定目录以提高性能
        # yf.set_tz_cache_location("/path/to/your/cache")
        
        # 分批处理
        batches = []
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(tickers))
            batches.append(tickers[start_idx:end_idx])
        
        all_results = []
        
        # 使用线程池处理批次
        with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有批次任务
            future_to_batch = {
                executor.submit(process_batch, batch, i+1, total_batches, batch_delay, request_delay): i 
                for i, batch in enumerate(batches)
            }
            
            # 使用tqdm显示总体进度
            for future in tqdm(cf.as_completed(future_to_batch), total=len(batches), desc="处理批次"):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result(timeout=300)  # 5分钟超时
                    all_results.extend(batch_results)
                    
                    # 批次间延迟（在批次完成后）:cite[2]
                    if batch_idx < len(batches) - 1 and batch_delay > 0:
                        logger.info(f"批次间延迟 {batch_delay} 秒...")
                        t.sleep(batch_delay)
                        
                except cf.TimeoutError:
                    logger.error(f"批次 {batch_idx + 1} 处理超时")
                except Exception as e:
                    logger.error(f"批次 {batch_idx + 1} 处理失败: {str(e)}")
        
        # 整理最终结果
        successful_tickers = [result[0] for result in all_results]
        logger.info(f"全部处理完成！成功获取 {len(successful_tickers)}/{len(tickers)} 只股票数据")
        
        # 返回成功获取的ticker列表和所有数据
        return {
            'successful_tickers': successful_tickers,
            'all_data': all_results,
            'success_rate': len(successful_tickers) / len(tickers) * 100
        }
        
    except FileNotFoundError:
        logger.error(f"股票列表文件未找到: {csv_path}")
        return None
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        return None

# 使用示例
if __name__ == "__main__":
    start = t.perf_counter()
    # 调用优化后的函数
    result = USgetAll_optimized(
        csv_path="Data/us_stock_list.csv",
        batch_size=50,      # 较小的批次大小有助于避免限制:cite[1]
        max_workers=5,      # 保守的并发数
        batch_delay=5,     # 批次间延迟10秒
        request_delay=1     # 请求间延迟1秒:cite[2]
    )
    
    if result:
        print(f"成功获取 {len(result['successful_tickers'])} 只股票数据")
        print(f"成功率: {result['success_rate']:.2f}%")

    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')


    





    
   
