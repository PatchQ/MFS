import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  

class VectorizedVCPScanner:
    def __init__(self, df):
        """
        初始化向量化 VCP 掃描器
        """
        self.df = df.copy()
        if not self.df.empty:
            self.df['Date'] = cc.pd.to_datetime(self.df.index)
            self.df = self.df.sort_values('Date').reset_index(drop=True)
            
    def calculate_trend_template(self):
        """計算 Mark Minervini 趨勢模板 (全向量化)"""
        # 計算移動平均線
        self.df['MA_20'] = self.df['Close'].rolling(window=20, min_periods=1).mean()
        self.df['MA_50'] = self.df['Close'].rolling(window=50, min_periods=1).mean()
        self.df['MA_150'] = self.df['Close'].rolling(window=150, min_periods=1).mean()
        self.df['MA_200'] = self.df['Close'].rolling(window=200, min_periods=1).mean()
        
        # 52週高低點 (約 250 個交易日)
        self.df['High_52W'] = self.df['High'].rolling(window=250, min_periods=1).max()
        self.df['Low_52W'] = self.df['Low'].rolling(window=250, min_periods=1).min()
        
        # 條件判斷 (勝率 55% 的適中設定：距離低點 > 20%，距離高點 25% 以內)
        self.df['Trend_Template'] = (
            (self.df['Close'] > self.df['MA_150']) & 
            (self.df['Close'] > self.df['MA_200']) & 
            (self.df['MA_150'] > self.df['MA_200']) & 
            (self.df['MA_50'] > self.df['MA_150']) & 
            (self.df['Close'] > self.df['MA_50']) &
            (self.df['Close'] >= self.df['Low_52W'] * 1.20) & 
            (self.df['Close'] >= self.df['High_52W'] * 0.75)  
        )
        return self.df

    def calculate_vcp_structure(self):
        """透過多重時間框架滾動視窗，向量化計算 VCP 波動收縮結構"""
        # 定義輔助函式：計算特定天數內的最大回撤 (Highest High - Lowest Low) / Highest High
        def get_rolling_retracement(window):
            roll_max = self.df['High'].rolling(window=window, min_periods=1).max()
            roll_min = self.df['Low'].rolling(window=window, min_periods=1).min()
            # 避免除以零
            return cc.np.where(roll_max > 0, (roll_max - roll_min) / roll_max, 0)

        # 1. 測量不同週期的震幅
        self.df['Depth_60D'] = get_rolling_retracement(60) # 大基底左側
        self.df['Depth_20D'] = get_rolling_retracement(20) # 內部收縮
        self.df['Depth_5D']  = get_rolling_retracement(5)  # 右側樞紐點緊湊度 (Tightness)
        
        # 2. VCP 幾何學條件：震幅必須階梯式遞減，且右側必須夠緊
        self.df['Contraction_Structure'] = (
            (self.df['Depth_60D'] > self.df['Depth_20D']) & 
            (self.df['Depth_20D'] > self.df['Depth_5D']) &
            (self.df['Depth_60D'] < 0.40) &  # 大基底跌幅不超過 40% (排除暴跌爛股)
            (self.df['Depth_5D'] < 0.08)     # 右側 5 天的震幅必須極小 (< 8%)，代表籌碼穩定
        )
        
        # 3. 成交量行為
        self.df['Vol_MA50'] = self.df['Volume'].rolling(window=50, min_periods=1).mean()
        self.df['Vol_MA10'] = self.df['Volume'].rolling(window=10, min_periods=1).mean()
        
        # 量縮：近期 10 日均量明顯低於 50 日均量
        self.df['Volume_Contraction'] = self.df['Vol_MA10'] < (self.df['Vol_MA50'] * 0.85)
        
        return self.df

    def calculate_breakout(self):
        """計算突破與量能爆發"""
        # 阻力位：過去 20 天的最高點 (不含當日)
        self.df['Resistance'] = self.df['High'].rolling(window=20, min_periods=1).max().shift(1)
        
        # 突破日條件：收盤大於阻力，且成交量大於 50日均量的 1.3 倍 (適中條件)
        # 同時要求收盤價收在當天振幅的上半部 (強勢收盤)
        daily_range = self.df['High'] - self.df['Low']
        close_strength =cc.np.where(daily_range > 0, (self.df['Close'] - self.df['Low']) / daily_range, 0)
        
        self.df['Breakout_Detected'] = (
            (self.df['Close'] >= self.df['Resistance']) & 
            (self.df['Volume'] >= self.df['Vol_MA50'] * 1.3) &
            (close_strength >= 0.5)
        )
        return self.df

    def run_scan(self):
        """執行所有向量化計算"""
        self.calculate_trend_template()
        self.calculate_vcp_structure()
        self.calculate_breakout()
        return self.df

def checkVCP(df):
    conditions = []        
    # df = pd.read_csv(PATH+"/"+stype+"/"+sno+".csv")         
    # df = convertData(df)    
        
    scanner = VectorizedVCPScanner(df)        
    df = scanner.run_scan()

    # 綜合條件 (Shift 1 確保我們是在「收縮完成」的隔天抓到「突破」)
    # 我們要求：昨天(含)之前符合趨勢與收縮，今天發生突破！
    
    # 這裡的邏輯是關鍵：VCP 型態建立在突破的前一天
    trend_yesterday = df['Trend_Template'].shift(1).fillna(False)
    vcp_yesterday = df['Contraction_Structure'].shift(1).fillna(False)
    vol_dry_yesterday = df['Volume_Contraction'].shift(1).fillna(False)
    breakout_today = df['Breakout_Detected']
    
    # 最終決策
    df['VCP'] = trend_yesterday & vcp_yesterday & vol_dry_yesterday & breakout_today

    if len(df) != 0:
        if 'Date' in df.columns:
            df = df.set_index('Date')
        df.index.name = 'index'    
        
    # 如果你有需要存檔，可以在這解除註解
    # df.to_csv(f"{cc.OUTPATH}/{stype}/P_{sno}.csv")

    return df

def ProcessVCP(stype):
    # 取得檔案列表
    try:
        snolist = list(map(lambda s: s.replace(".csv", ""), os.listdir(cc.PATH+"/"+stype)))
    except FileNotFoundError:
        print(f"Directory {cc.PATH}/{stype} not found.")
        return
        
    SLIST = cc.pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype)

    # 平行運算
    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        # 使用 list 將結果載入，觸發 tqdm 進度條
        list(cc.tqdm(executor.map(checkVCP, SLIST["sno"], SLIST["stype"], chunksize=1), total=len(SLIST)))

if __name__ == '__main__':
    start = cc.t.perf_counter()

    # 假設這是你的資料夾結構分類
    ProcessVCP("L")    
    #ProcessVCP("M")

    finish = cc.t.perf_counter()
    print(f'It took {round(finish-start, 2)} second(s) to finish.')