import time as t
import tushare as ts
import pandas as pd

ts.set_token('4228ed3bd2b53edd9c1ced494c8190da84cd05b2869822905b4a1bbd')

if __name__ == '__main__':

    start = t.perf_counter()

    pro = ts.pro_api()

    df = pro.query('daily', ts_code='000001.SZ', start_date='20180701', end_date='20180718')

    print(df)
   
    finish = t.perf_counter()
    print(f'It took {round(finish-start,2)} second(s) to finish.')
