import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================
# VCP (Volatility Contraction Pattern) 策略
# ============================================

class VCPBacktest:
    def __init__(self, symbol, start_date, end_date, initial_capital=100000):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0  # 持股數量
        self.position_price = 0  # 持股成本
        self.trades = []  # 交易記錄
        self.equity_curve = []  # 資金曲線

    def download_data(self):
        """下載股票數據"""
        print(f"正在下載 {self.symbol} 的數據...")
        df = yf.download(self.symbol, start=self.start_date, end=self.end_date, progress=False)

        if df.empty:
            raise ValueError("無法獲取數據，請檢查股票代碼")

        # 確保數據格式正確
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']

        df = df.reset_index()
        print(f"成功下載 {len(df)} 條數據")
        return df

    def calculate_vcp_indicators(self, df, lookback=20, contraction_threshold=0.15):
        """計算VCP相關指標"""
        df = df.copy()

        # 計算區間最高價和最低價
        df['High_20'] = df['High'].rolling(window=lookback).max()
        df['Low_20'] = df['Low'].rolling(window=lookback).min()

        # 計算區間波動率
        df['Range_20'] = (df['High_20'] - df['Low_20']) / df['Low_20']

        # 計算收縮率與歷史波動率比較
        df['Avg_Range'] = df['Range_20'].rolling(window=20).mean()
        df['Contraction'] = df['Range_20'] / df['Avg_Range']

        # 計算均線
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()

        # 計算價格動量
        df['Momentum'] = df['Close'].pct_change(periods=10)

        return df

    def detect_vcp_pattern(self, df, i, min_contraction=0.5, max_range=0.2):
        """檢測VCP模式"""
        if i < 50:
            return False

        # 檢查當前波動是否收縮
        if pd.isna(df.loc[i, 'Contraction']).any():
            return False

        # 波動收縮條件
        is_contraction = df.loc[i, 'Contraction'] < min_contraction
        is_low_range = df.loc[i, 'Range_20'] < max_range

        # 價格在均線之上
        above_ma20 = df.loc[i, 'Close'] > df.loc[i, 'MA20']
        above_ma50 = df.loc[i, 'Close'] > df.loc[i, 'MA50']

        # 均線多頭排列
        ma_bullish = df.loc[i, 'MA20'] > df.loc[i, 'MA50']

        # 價格在區間內部
        in_range = df.loc[i, 'Low_20'] < df.loc[i, 'Close'] < df.loc[i, 'High_20']

        return (is_contraction and is_low_range and above_ma20 and
                above_ma50 and ma_bullish and in_range)

    def detect_breakout(self, df, i):
        """檢測突破信號"""
        if i < 1:
            return False

        # 價格突破20日高點
        prev_close = df.loc[i-1, 'Close']
        curr_close = df.loc[i, 'Close']
        high_20 = df.loc[i, 'High_20']

        # 前一天在區間內，今天突破高點
        prev_in_range = df.loc[i-1, 'Low_20'] < prev_close < df.loc[i-1, 'High_20']
        breakout = curr_close > high_20 and prev_in_range

        return breakout

    def detect_sell_signal(self, df, i, stop_loss=0.08, trailing_stop=0.05):
        """檢測賣出信號"""
        if self.position == 0:
            return False, "No position"

        current_price = df.loc[i, 'Close']

        # 止損條件
        if current_price < self.position_price * (1 - stop_loss):
            return True, "Stop Loss"

        # 移動止損
        if current_price > self.position_price * 1.10:  # 獲利10%後啟動移動止損
            trailing_stop_price = self.position_price * (1 + trailing_stop)
            if current_price < trailing_stop_price:
                return True, "Trailing Stop"

        # 跌破 MA20
        if current_price < df.loc[i, 'MA20']:
            return True, "Below MA20"

        # 跌破 MA50
        if current_price < df.loc[i, 'MA50']:
            return True, "Below MA50"

        return False, "Hold"

    def run_backtest(self,
                     min_contraction=0.5,
                     max_range=0.2,
                     stop_loss=0.08,
                     trailing_stop=0.05,
                     position_size=0.95):
        """運行回測"""
        # 獲取數據
        df = self.download_data()

        # 計算指標
        df = self.calculate_vcp_indicators(df)

        print("\n開始回測...")
        print("=" * 50)

        # 重置狀態
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []

        vcp_confirmed = False
        vcp_low = 0

        for i in range(50, len(df)):
            date = df.loc[i, 'Date'] if 'Date' in df.columns else df.index[i]
            current_price = df.loc[i, 'Close']

            # 記錄當前資產淨值
            current_equity = self.capital + self.position * current_price
            self.equity_curve.append({
                'Date': date,
                'Equity': current_equity
            })

            # 檢測賣出信號
            if self.position > 0:
                should_sell, sell_reason = self.detect_sell_signal(df, i, stop_loss, trailing_stop)

                if should_sell:
                    # 執行賣出
                    proceeds = self.position * current_price
                    self.capital += proceeds
                    profit = proceeds - (self.position * self.position_price)
                    profit_pct = (profit / (self.position * self.position_price)) * 100

                    self.trades.append({
                        'Date': date,
                        'Action': 'SELL',
                        'Price': current_price,
                        'Shares': self.position,
                        'Reason': sell_reason,
                        'Profit': profit,
                        'Profit%': profit_pct
                    })

                    print(f"賣出 | 日期: {date} | 價格: {current_price:.2f} | "
                          f"原因: {sell_reason} | 獲利: {profit_pct:.2f}%")

                    self.position = 0
                    self.position_price = 0
                    vcp_confirmed = False

            # 檢測買入信號
            if self.position == 0:
                # 檢測VCP模式
                if self.detect_vcp_pattern(df, i, min_contraction, max_range):
                    vcp_confirmed = True
                    vcp_low = df.loc[i, 'Low_20']

                # 檢測突破買入
                if vcp_confirmed and self.detect_breakout(df, i):
                    # 買入
                    shares_to_buy = int((self.capital * position_size) / current_price)

                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        self.capital -= cost
                        self.position = shares_to_buy
                        self.position_price = current_price

                        self.trades.append({
                            'Date': date,
                            'Action': 'BUY',
                            'Price': current_price,
                            'Shares': shares_to_buy,
                            'Reason': 'VCP Breakout',
                            'Profit': 0,
                            'Profit%': 0
                        })

                        print(f"買入 | 日期: {date} | 價格: {current_price:.2f} | "
                              f"數量: {shares_to_buy} | VCP突破")

                        vcp_confirmed = False

        # 如果最後還有持股，根據最後一天收盤價平倉
        if self.position > 0:
            last_price = df.iloc[-1]['Close']
            proceeds = self.position * last_price
            self.capital += proceeds
            profit = proceeds - (self.position * self.position_price)
            profit_pct = (profit / (self.position * self.position_price)) * 100

            self.trades.append({
                'Date': df.index[-1],
                'Action': 'SELL',
                'Price': last_price,
                'Shares': self.position,
                'Reason': 'End of Backtest',
                'Profit': profit,
                'Profit%': profit_pct
            })

            self.position = 0

        return self.generate_report()

    def generate_report(self):
        """生成回測報告"""
        if not self.trades:
            return "無交易記錄"

        trades_df = pd.DataFrame(self.trades)

        # 計算統計數據
        total_trades = len(trades_df)
        buy_trades = trades_df[trades_df['Action'] == 'BUY']
        sell_trades = trades_df[trades_df['Action'] == 'SELL']

        winning_trades = sell_trades[sell_trades['Profit'] > 0]
        losing_trades = sell_trades[sell_trades['Profit'] <= 0]

        win_rate = len(winning_trades) / len(sell_trades) * 100 if len(sell_trades) > 0 else 0

        total_profit = sell_trades['Profit'].sum()
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100

        # 最大回撤
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['Peak'] = equity_df['Equity'].cummax()
        equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak']
        max_drawdown = equity_df['Drawdown'].min() * 100

        # 年化收益率
        days = (pd.to_datetime(self.end_date) - pd.to_datetime(self.start_date)).days
        annual_return = total_return * 365 / days if days > 0 else 0

        print("\n" + "=" * 50)
        print("           VCP 策略回測報告")
        print("=" * 50)
        print(f"股票代碼: {self.symbol}")
        print(f"回測期間: {self.start_date} ~ {self.end_date}")
        print(f"初始資金: ${self.initial_capital:,.2f}")
        print(f"最終資金: ${self.capital:,.2f}")
        print("-" * 50)
        print(f"總交易次數: {len(sell_trades)}")
        print(f"獲勝次數: {len(winning_trades)}")
        print(f"失敗次數: {len(losing_trades)}")
        print(f"勝率: {win_rate:.2f}%")
        print("-" * 50)
        print(f"總收益: ${total_profit:,.2f}")
        print(f"總收益率: {total_return:.2f}%")
        print(f"年化收益率: {annual_return:.2f}%")
        print(f"最大回撤: {max_drawdown:.2f}%")
        print("=" * 50)

        return {
            'symbol': self.symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(sell_trades),
            'win_rate': win_rate,
            'trades': trades_df
        }


def main():
    """主函數 - 執行VCP策略回測"""

    # 回測參數設置
    SYMBOL = "AAPL"              # 股票代碼
    START_DATE = "2020-01-01"    # 開始日期
    END_DATE = "2024-12-31"      # 結束日期
    INITIAL_CAPITAL = 100000     # 初始資金

    # VCP策略參數
    MIN_CONTRACTION = 0.5        # 最小收縮比率 (相對於歷史平均)
    MAX_RANGE = 0.2              # 最大波動範圍 (20%)
    STOP_LOSS = 0.08             # 止損比例 (8%)
    TRAILING_STOP = 0.05         # 移動止損比例 (5%)
    POSITION_SIZE = 0.95         # 倉位大小 (95%)

    print("=" * 60)
    print("      VCP (Volatility Contraction Pattern) 策略回測")
    print("=" * 60)
    print(f"股票: {SYMBOL}")
    print(f"期間: {START_DATE} ~ {END_DATE}")
    print(f"初始資金: ${INITIAL_CAPITAL:,}")
    print("-" * 60)

    # 創建回測對象並運行
    backtest = VCPBacktest(
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL
    )

    # 運行回測
    results = backtest.run_backtest(
        min_contraction=MIN_CONTRACTION,
        max_range=MAX_RANGE,
        stop_loss=STOP_LOSS,
        trailing_stop=TRAILING_STOP,
        position_size=POSITION_SIZE
    )

    # 顯示交易記錄
    print("\n" + "=" * 60)
    print("              交易記錄")
    print("=" * 60)
    if results != "無交易記錄" and 'trades' in results:
        trades_df = results['trades']
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        print(trades_df.to_string(index=False))

    return results


if __name__ == "__main__":
    results = main()