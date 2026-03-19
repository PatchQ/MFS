import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== VCP 模式識別 ====================

def calculate_volatility(df, period=20):
    """計算歷史波動率"""
    df['Volatility'] = df['Close'].pct_change().rolling(window=period).std() * np.sqrt(252)
    return df

def find_vcp_patterns(df, min_contraction=0.3, max_contraction=0.5, lookback=50):
    """
    識別VCP模式

    參數:
    - min_contraction: 最小波動率收縮比例 (30%)
    - max_contraction: 最大波動率收縮比例 (50%)
    - lookback: 向前看的週期數

    VCP特徵:
    1. 股價在上漲趨勢中
    2. 波動率逐漸收縮
    3. 價格波動範圍變窄
    4. 成交量逐漸減少
    """
    df = df.copy()
    df = calculate_volatility(df)

    # 計算價格波動範圍
    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Low']
    df['Avg_Range'] = df['High_Low_Range'].rolling(window=20).mean()

    # 計算移動平均線 (50日和200日)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # 計算成交量變化
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

    # 識別VCP模式
    vcp_signals = []

    for i in range(lookback, len(df)):
        if pd.isna(df['SMA_50'].iloc[i]) or pd.isna(df['SMA_200'].iloc[i]):
            continue

        # 檢查是否在上漲趨勢中 (股價在50日和200日均線之上)
        if df['Close'].iloc[i] < df['SMA_50'].iloc[i] or df['Close'].iloc[i] < df['SMA_200'].iloc[i]:
            continue

        # 檢查50日均線在200日均線之上 (長期趨勢向上)
        if df['SMA_50'].iloc[i] < df['SMA_200'].iloc[i]:
            continue

        # 計算過去N天的波動率收縮
        start_idx = max(0, i - lookback)
        early_volatility = df['Volatility'].iloc[start_idx:i-20].mean() if i > 20 else df['Volatility'].iloc[start_idx:i].mean()
        recent_volatility = df['Volatility'].iloc[i-20:i].mean()

        if early_volatility == 0 or pd.isna(early_volatility):
            continue

        contraction_ratio = 1 - (recent_volatility / early_volatility)

        # 檢查波動率收縮是否在設定範圍內
        if min_contraction <= contraction_ratio <= max_contraction:
            # 檢查價格波動範圍是否收縮
            early_range = df['High_Low_Range'].iloc[start_idx:i-20].mean()
            recent_range = df['High_Low_Range'].iloc[i-20:i].mean()

            if early_range > 0 and recent_range < early_range:
                # 檢查成交量是否減少 (或者至少沒有大幅增加)
                early_volume = df['Volume_Ratio'].iloc[start_idx:i-20].mean()
                recent_volume = df['Volume_Ratio'].iloc[i-20:i].mean()

                vcp_signals.append({
                    'Date': df.index[i],
                    'Close': df['Close'].iloc[i],
                    'Contraction_Ratio': contraction_ratio,
                    'Early_Volatility': early_volatility,
                    'Recent_Volatility': recent_volatility,
                    'Volume_Trend': 'Decreasing' if recent_volume < early_volume else 'Stable'
                })

    return pd.DataFrame(vcp_signals)

def detect_breakout(df, window=5):
    """
    檢測價格突破
    突破定義: 股價在盤整後放量突破最高點
    """
    breakout_signals = []

    for i in range(window, len(df)):
        # 獲取過去window天的最高價
        highest_high = df['High'].iloc[i-window:i].max()
        current_close = df['Close'].iloc[i]

        # 檢查是否突破
        if current_close > highest_high:
            # 檢查成交量是否放大 (大於20日平均成交量的1.5倍)
            avg_volume = df['Volume_MA'].iloc[i]
            current_volume = df['Volume'].iloc[i]

            if current_volume > avg_volume * 1.5:
                breakout_signals.append({
                    'Date': df.index[i],
                    'Close': current_close,
                    'Breakout_Level': highest_high,
                    'Volume': current_volume,
                    'Volume_Ratio': current_volume / avg_volume if avg_volume > 0 else 0
                })

    return pd.DataFrame(breakout_signals)

# ==================== 交易信號 ====================

def generate_trading_signals(df, min_contraction=0.3, max_contraction=0.5):
    """
    生成交易信號
    買入信號: VCP模式形成 + 放量突破
    賣出信號: 價格跌破止損位 or 達到目標收益
    """
    df = df.copy()

    # 識別VCP模式
    vcp_patterns = find_vcp_patterns(df, min_contraction, max_contraction)

    # 識別突破
    breakouts = detect_breakout(df)

    # 合併信號
    signals = []

    for _, vcp in vcp_patterns.iterrows():
        vcp_date = vcp['Date']

        # 尋找在VCP形成後的突破
        for _, breakout in breakouts.iterrows():
            if breakout['Date'] > vcp_date:
                # 計算從VCP低點到突破的時間
                days_after = (breakout['Date'] - vcp_date).days

                if days_after <= 30:  # 30天內的突破視為有效
                    signals.append({
                        'Entry_Date': breakout['Date'],
                        'Entry_Price': breakout['Close'],
                        'Breakout_Level': breakout['Breakout_Level'],
                        'VCP_Contraction': vcp['Contraction_Ratio'],
                        'Volume_Ratio': breakout['Volume_Ratio'],
                        'Signal_Type': 'VCP_Breakout'
                    })
                    break

    return pd.DataFrame(signals)

# ==================== 回測引擎 ====================

class VCPBacktester:
    def __init__(self, initial_capital=100000, stop_loss=0.08, take_profit=0.20):
        """
        初始化回測器

        參數:
        - initial_capital: 初始資金
        - stop_loss: 止損比例 (8%)
        - take_profit: 止盈比例 (20%)
        """
        self.initial_capital = initial_capital
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trades = []
        self.equity_curve = []

    def run_backtest(self, df, min_contraction=0.3, max_contraction=0.5):
        """
        運行回測
        """
        df = df.copy()

        # 確保有足夠的歷史數據
        if len(df) < 250:
            print("數據不足，需要至少250天的數據")
            return None

        # 生成交易信號
        signals = generate_trading_signals(df, min_contraction, max_contraction)

        if signals.empty:
            print("沒有找到交易信號")
            return None

        # 初始化資金
        capital = self.initial_capital
        position = None
        trades = []

        # 模擬交易
        for idx, signal in signals.iterrows():
            entry_date = signal['Entry_Date']
            entry_price = signal['Entry_Price']

            # 計算止損和止盈價位
            stop_loss_price = entry_price * (1 - self.stop_loss)
            take_profit_price = entry_price * (1 + self.take_profit)

            # 從進場日開始模擬持倉
            df_after_entry = df[df.index > entry_date]

            if df_after_entry.empty:
                continue

            exit_price = None
            exit_date = None
            exit_reason = None

            for i, (date, row) in enumerate(df_after_entry.iterrows()):
                current_price = row['Close']
                high_price = row['High']
                low_price = row['Low']

                # 檢查止損 (賣出)
                if low_price <= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_date = date
                    exit_reason = 'Stop_Loss'
                    break

                # 檢查止盈 (賣出)
                if high_price >= take_profit_price:
                    exit_price = take_profit_price
                    exit_date = date
                    exit_reason = 'Take_Profit'
                    break

                # 如果持有超過60天，強制平倉
                if i >= 60:
                    exit_price = current_price
                    exit_date = date
                    exit_reason = 'Time_Exit'
                    break

            # 記錄交易
            if exit_price is not None:
                profit = (exit_price - entry_price) / entry_price
                trades.append({
                    'Entry_Date': entry_date,
                    'Entry_Price': entry_price,
                    'Exit_Date': exit_date,
                    'Exit_Price': exit_price,
                    'Profit_Pct': profit * 100,
                    'Exit_Reason': exit_reason,
                    'VCP_Contraction': signal['VCP_Contraction']
                })

        self.trades = pd.DataFrame(trades)
        return self.trades

    def calculate_metrics(self):
        """
        計算回測指標
        """
        if self.trades.empty:
            return None

        trades = self.trades

        # 基本指標
        total_trades = len(trades)
        winning_trades = len(trades[trades['Profit_Pct'] > 0])
        losing_trades = len(trades[trades['Profit_Pct'] <= 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # 收益指標
        total_return = trades['Profit_Pct'].sum()
        avg_return = trades['Profit_Pct'].mean()
        avg_win = trades[trades['Profit_Pct'] > 0]['Profit_Pct'].mean() if winning_trades > 0 else 0
        avg_loss = trades[trades['Profit_Pct'] <= 0]['Profit_Pct'].mean() if losing_trades > 0 else 0

        # 風險指標
        profit_factor = abs(trades[trades['Profit_Pct'] > 0]['Profit_Pct'].sum() /
                          trades[trades['Profit_Pct'] <= 0]['Profit_Pct'].sum()) if losing_trades > 0 else float('inf')

        # 最大回撤
        cumulative = (1 + trades['Profit_Pct']/100).cumprod()
        max_drawdown = ((cumulative.cummax() - cumulative) / cumulative.cummax()).max()

        # 夏普比率 (假設無風險利率為0)
        if trades['Profit_Pct'].std() > 0:
            sharpe_ratio = (trades['Profit_Pct'].mean() / trades['Profit_Pct'].std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        metrics = {
            'Total_Trades': total_trades,
            'Winning_Trades': winning_trades,
            'Losing_Trades': losing_trades,
            'Win_Rate': f"{win_rate*100:.2f}%",
            'Total_Return': f"{total_return:.2f}%",
            'Average_Return': f"{avg_return:.2f}%",
            'Average_Win': f"{avg_win:.2f}%",
            'Average_Loss': f"{avg_loss:.2f}%",
            'Profit_Factor': f"{profit_factor:.2f}",
            'Max_Drawdown': f"{max_drawdown*100:.2f}%",
            'Sharpe_Ratio': f"{sharpe_ratio:.2f}"
        }

        return metrics

    def print_results(self):
        """打印回測結果"""
        metrics = self.calculate_metrics()

        if metrics is None:
            print("沒有交易記錄")
            return

        print("\n" + "="*50)
        print("VCP策略 回測結果")
        print("="*50)
        print(f"總交易次數: {metrics['Total_Trades']}")
        print(f"獲勝次數: {metrics['Winning_Trades']}")
        print(f"失敗次數: {metrics['Losing_Trades']}")
        print(f"勝率: {metrics['Win_Rate']}")
        print("-"*50)
        print(f"總收益: {metrics['Total_Return']}")
        print(f"平均收益: {metrics['Average_Return']}")
        print(f"平均獲勝: {metrics['Average_Win']}")
        print(f"平均虧損: {metrics['Average_Loss']}")
        print("-"*50)
        print(f"利潤因子: {metrics['Profit_Factor']}")
        print(f"最大回撤: {metrics['Max_Drawdown']}")
        print(f"夏普比率: {metrics['Sharpe_Ratio']}")
        print("="*50)

    def plot_results(self):
        """繪製交易結果圖表"""
        if self.trades.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 收益分佈
        axes[0, 0].hist(self.trades['Profit_Pct'], bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('收益 (%)')
        axes[0, 0].set_ylabel('次數')
        axes[0, 0].set_title('收益分佈')

        # 2. 累積收益
        cumulative = self.trades['Profit_Pct'].cumsum()
        axes[0, 1].plot(cumulative)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[0, 1].set_xlabel('交易次數')
        axes[0, 1].set_ylabel('累積收益 (%)')
        axes[0, 1].set_title('累積收益曲線')

        # 3. 收益原因分析
        exit_reasons = self.trades['Exit_Reason'].value_counts()
        axes[1, 0].pie(exit_reasons, labels=exit_reasons.index, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('平倉原因分析')

        # 4. 每筆交易收益
        axes[1, 1].bar(range(len(self.trades)), self.trades['Profit_Pct'],
                      color=['green' if x > 0 else 'red' for x in self.trades['Profit_Pct']])
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].set_xlabel('交易次數')
        axes[1, 1].set_ylabel('收益 (%)')
        axes[1, 1].set_title('每筆交易收益')

        plt.tight_layout()
        plt.savefig('vcp_backtest_results.png', dpi=150)
        plt.show()

# ==================== 使用範例 ====================

def main():
    """
    主函數 - 演示如何使用VCP策略回測
    """
    # 設置參數
    ticker = "AAPL"  # 可以改為其他股票代碼
    start_date = "2020-01-01"
    end_date = "2024-12-31"

    # 獲取數據
    print(f"正在獲取 {ticker} 的歷史數據...")
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        print("無法獲取數據")
        return

    print(f"獲取到 {len(df)} 天的數據")

    # 運行回測
    print("\n開始運行VCP策略回測...")
    backtester = VCPBacktester(
        initial_capital=100000,  # 初始資金10萬
        stop_loss=0.08,          # 8%止損
        take_profit=0.20         # 20%止盈
    )

    trades = backtester.run_backtest(df, min_contraction=0.3, max_contraction=0.5)

    # 打印結果
    backtester.print_results()

    # 繪製圖表
    backtester.plot_results()

    # 顯示交易明細
    if trades is not None and not trades.empty:
        print("\n交易明細:")
        print(trades.to_string())

if __name__ == "__main__":
    main()