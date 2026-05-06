#!/usr/bin/env python3
"""
🌤️ 一目均衡表 + 技術指標黃金組合
Ichimoku Kinko Hyo + RSI / MACD / MA Combination

適合忙人嘅懶人版本！作者：H老師
"""

import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from datetime import datetime, timedelta

# 設定中文字體（使用文泉驛正黑）
fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
zh_font = fm.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============ 第一部分：一目均衡表計算 ============

def calculate_ichimoku(df, tenkan_period=34, kijun_period=5, senkou_b_period=52, cloud_shift=26):
    """
    計算一目均衡表嘅所有線
    
    參數說明：
    - tenkan_period (34): 轉換線週期
    - kijun_period (5): 基準線週期
    - senkou_b_period (52): 先行線週期
    - cloud_shift (26): 雲層推移週期（固定26）
    """
    
    # 複製數據，避免修改原始數據
    data = df.copy()
    
    # -------- 轉換線 (Tenkan-sen) --------
    # 即係最高價同最低價嘅平均值
    high_9 = data['High'].rolling(window=tenkan_period).max()
    low_9 = data['Low'].rolling(window=tenkan_period).min()
    data['Tenkan_Sen'] = (high_9 + low_9) / 2
    
    # -------- 基準線 (Kijun-sen) --------
    # 長期嘅最高價同最低價平均值
    high_26 = data['High'].rolling(window=kijun_period).max()
    low_26 = data['Low'].rolling(window=kijun_period).min()
    data['Kijun_Sen'] = (high_26 + low_26) / 2
    
    # -------- 先行上限 (Senkou Span A) --------
    # 轉換線同基準線嘅平均值，向前推移26日
    data['Senkou_Span_A'] = ((data['Tenkan_Sen'] + data['Kijun_Sen']) / 2).shift(cloud_shift)
    
    # -------- 先行下限 (Senkou Span B) --------
    # 長期最高價同最低價嘅平均值，向前推移26日
    high_52 = data['High'].rolling(window=senkou_b_period).max()
    low_52 = data['Low'].rolling(window=senkou_b_period).min()
    data['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(cloud_shift)
    
    # -------- 雲層 (Kumo) --------
    # 雲層係先行上限同下限之間嘅空間
    data['Cloud_Upper'] = data[['Senkou_Span_A', 'Senkou_Span_B']].max(axis=1)
    data['Cloud_Lower'] = data[['Senkou_Span_A', 'Senkou_Span_B']].min(axis=1)
    
    # -------- 遲行線 (Chikou Span) --------
    # 今日收盤價，向前睇26日前嘅位置
    data['Chikou_Span'] = data['Close'].shift(-cloud_shift)
    
    return data


# ============ 第二部分：RSI 計算 ============

def calculate_rsi(df, period=14):
    """
    計算相對強弱指數 (RSI)
    
    簡單理解：
    - RSI > 70 = 身體太熱（超買）
    - RSI < 30 = 身體太凍（超賣）
    """
    
    # 計算價格變化
    delta = df['Close'].diff()
    
    # 分開上升同下降
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    # 計算平均上升同下降
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # 計算 RS 同 RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


# ============ 第三部分：MACD 計算 ============

def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    計算移動平均收斂發散指標 (MACD)
    
    簡單理解：
    - MACD 線 > 信號線 = 綠燈（可能係買入時機）
    - MACD 線 < 信號線 = 紅燈（可能係賣出時機）
    """
    
    # 快速同慢速指數移動平均線
    ema_fast = df['Close'].ewm(span=fast).mean()
    ema_slow = df['Close'].ewm(span=slow).mean()
    
    # MACD 線 = 快速線 - 慢速線
    macd_line = ema_fast - ema_slow
    
    # 信號線 = MACD 線嘅移動平均
    signal_line = macd_line.ewm(span=signal).mean()
    
    # MACD 柱狀圖
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram


# ============ 第四部分：移動平均線 ============

def calculate_ma(df, periods=[49, 233]):
    """
    計算簡單移動平均線 (SMA)
    
    - 49日線 = 短期乘客滿意度
    - 233日線 = 長期服務質量
    """
    
    ma_dict = {}
    for period in periods:
        ma_dict[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
    
    return pd.DataFrame(ma_dict)


# ============ 第五部分：信號判斷 ============

def generate_signals(df):
    """
    根據三個黃金組合生成交易信號
    
    Returns:
        信號意義：
        - 1 = 強烈買入
        - 0.5 = 輕微買入
        - 0 = 中立/觀望
        - -0.5 = 輕微賣出
        - -1 = 強烈賣出
    """
    
    signals = pd.DataFrame(index=df.index)
    signals['Price'] = df['Close']
    signals['Tenkan'] = df['Tenkan_Sen']
    signals['Kijun'] = df['Kijun_Sen']
    signals['Cloud_Upper'] = df['Cloud_Upper']
    signals['Cloud_Lower'] = df['Cloud_Lower']
    signals['RSI'] = df['RSI']
    signals['MACD'] = df['MACD']
    signals['MACD_Signal'] = df['MACD_Signal']
    signals['SMA_49'] = df['SMA_49']
    signals['SMA_233'] = df['SMA_233']
    
    # -------- 初始化信號（用 float 避免TypeError）--------
    signals['Signal_1_Ichimoku_RSI'] = 0.0  # 組合一
    signals['Signal_2_Ichimoku_MACD'] = 0.0  # 組合二
    signals['Signal_3_Ichimoku_MA'] = 0.0   # 組合三
    signals['Combined_Signal'] = 0.0        # 綜合信號
    
    # -------- 組合一：Ichimoku + RSI --------
    # 買入條件：股價喺雲上面 + RSI > 70 (超買確認上升)
    # 賣出條件：股價喺雲下面 + RSI < 30 (超賣確認下跌)
    
    above_cloud = signals['Price'] > signals['Cloud_Upper']
    below_cloud = signals['Price'] < signals['Cloud_Lower']
    in_cloud = ~(above_cloud | below_cloud)
    
    # 強烈買入：雲上面 + RSI > 70
    signals.loc[above_cloud & (signals['RSI'] > 70), 'Signal_1_Ichimoku_RSI'] = 1
    # 輕微買入：雲上面 + RSI 正常但係上升緊
    signals.loc[above_cloud & (signals['RSI'] <= 70) & (signals['RSI'] > 50), 'Signal_1_Ichimoku_RSI'] = 0.5
    # 強烈賣出：雲下面 + RSI < 30
    signals.loc[below_cloud & (signals['RSI'] < 30), 'Signal_1_Ichimoku_RSI'] = -1
    # 輕微賣出：雲下面 + RSI 正常但係下降緊
    signals.loc[below_cloud & (signals['RSI'] >= 30) & (signals['RSI'] < 50), 'Signal_1_Ichimoku_RSI'] = -0.5
    
    # -------- 組合二：Ichimoku + MACD --------
    # 買入條件：轉換線 > 基準線 + MACD > 信號線 + 股價喺雲上面
    # 賣出條件：轉換線 < 基準線 + MACD < 信號線 + 股價喺雲下面
    
    golden_cross = signals['Tenkan'] > signals['Kijun']  # 黃金交叉
    death_cross = signals['Tenkan'] < signals['Kijun']   # 死亡交叉
    macd_bullish = signals['MACD'] > signals['MACD_Signal']  # MACD 做嘢
    macd_bearish = signals['MACD'] < signals['MACD_Signal']  # MACD 衰嘢
    
    # 強烈買入：三個條件全部符合
    buy_2 = golden_cross & macd_bullish & above_cloud
    signals.loc[buy_2, 'Signal_2_Ichimoku_MACD'] = 1
    
    # 輕微買入：兩個條件符合
    buy_2_partial = (golden_cross & macd_bullish) | (golden_cross & above_cloud) | (macd_bullish & above_cloud)
    signals.loc[buy_2_partial & ~buy_2, 'Signal_2_Ichimoku_MACD'] = 0.5
    
    # 強烈賣出：三個條件全部符合
    sell_2 = death_cross & macd_bearish & below_cloud
    signals.loc[sell_2, 'Signal_2_Ichimoku_MACD'] = -1
    
    # 輕微賣出：兩個條件符合
    sell_2_partial = (death_cross & macd_bearish) | (death_cross & below_cloud) | (macd_bearish & below_cloud)
    signals.loc[sell_2_partial & ~sell_2, 'Signal_2_Ichimoku_MACD'] = -0.5
    
    # -------- 組合三：Ichimoku + MA --------
    # 買入條件：股價 > 49日線 > 233日線 + 股價喺雲上面
    # 賣出條件：股價 < 49日線 < 233日線 + 股價喺雲下面
    
    ma_bullish = signals['SMA_49'] > signals['SMA_233']  # 黃金交叉（長遠向上）
    ma_bearish = signals['SMA_49'] < signals['SMA_233']  # 死亡交叉（長遠向下）
    price_above_ma49 = signals['Price'] > signals['SMA_49']
    price_below_ma49 = signals['Price'] < signals['SMA_49']
    
    # 強烈買入
    buy_3 = above_cloud & price_above_ma49 & ma_bullish
    signals.loc[buy_3, 'Signal_3_Ichimoku_MA'] = 1
    
    # 輕微買入
    buy_3_partial = above_cloud & price_above_ma49
    signals.loc[buy_3_partial & ~buy_3, 'Signal_3_Ichimoku_MA'] = 0.5
    
    # 強烈賣出
    sell_3 = below_cloud & price_below_ma49 & ma_bearish
    signals.loc[sell_3, 'Signal_3_Ichimoku_MA'] = -1
    
    # 輕微賣出
    sell_3_partial = below_cloud & price_below_ma49
    signals.loc[sell_3_partial & ~sell_3, 'Signal_3_Ichimoku_MA'] = -0.5
    
    # -------- 綜合信號（懶人版）--------
    # 計算三個組合嘅加權平均
    signals['Combined_Signal'] = (
        signals['Signal_1_Ichimoku_RSI'] * 0.3 +
        signals['Signal_2_Ichimoku_MACD'] * 0.4 +
        signals['Signal_3_Ichimoku_MA'] * 0.3
    )
    
    return signals


# ============ 第六部分：圖表繪製 ============

def plot_ichimoku_combinations(df, signals, ticker):
    """
    繪製一目均衡表 + 黃金組合信號圖表
    """
    
    # 創建副圖
    make_marketcolor = mpf.make_marketcolors(
        up='#00FF00', down='#FF0000',
        edge='inherit',
        wick='inherit',
        volume='in'
    )
    make_style = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=make_marketcolor)
    
    # 準備 K線圖數據
    plot_data = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # -------- 第一張圖：一目均衡表 --------
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(f'Ichimoku + Golden Combinations Analysis - {ticker}', fontsize=16, fontweight='bold')
    
    # K線圖
    ax1 = axes[0]
    ax1.plot(df.index, df['Tenkan_Sen'], label='Tenkan-sen (Conversion)', color='blue', linewidth=1)
    ax1.plot(df.index, df['Kijun_Sen'], label='Kijun-sen (Base)', color='red', linewidth=1)
    ax1.plot(df.index, df['Cloud_Upper'], label='Cloud Upper', color='green', linewidth=1, linestyle='--')
    ax1.plot(df.index, df['Cloud_Lower'], label='Cloud Lower', color='red', linewidth=1, linestyle='--')
    ax1.fill_between(df.index, df['Cloud_Upper'], df['Cloud_Lower'], 
                      where=df['Cloud_Upper'] >= df['Cloud_Lower'],
                      color='lightgreen', alpha=0.3, label='Cloud (Bullish)')
    ax1.fill_between(df.index, df['Cloud_Upper'], df['Cloud_Lower'], 
                      where=df['Cloud_Upper'] < df['Cloud_Lower'],
                      color='lightcoral', alpha=0.3, label='Cloud (Bearish)')
    ax1.set_title('Ichimoku Kinko Hyo', fontsize=12)
    ax1.legend(loc='upper left', prop=zh_font)
    ax1.grid(True, alpha=0.3)
    
    # -------- 第二張圖：RSI + MACD --------
    ax2 = axes[1]
    ax2.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=1)
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax2.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
    ax2.set_title('RSI (Relative Strength Index)', fontsize=12)
    ax2.legend(loc='upper left', prop=zh_font)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # -------- 第三張圖：MACD --------
    ax3 = axes[2]
    ax3.plot(df.index, df['MACD'], label='MACD Line', color='blue', linewidth=1)
    ax3.plot(df.index, df['MACD_Signal'], label='Signal Line', color='orange', linewidth=1)
    colors = ['green' if val >= 0 else 'red' for val in df['MACD_Histogram']]
    ax3.bar(df.index, df['MACD_Histogram'], color=colors, alpha=0.5, label='MACD Histogram')
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3.set_title('MACD (Moving Average Convergence Divergence)', fontsize=12)
    ax3.legend(loc='upper left', prop=zh_font)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('HERMES/Skill/ichimoku_analysis.png', dpi=150, bbox_inches='tight')
    print('✅ 圖表已保存到 HERMES/Skill/ichimoku_analysis.png')
    
    # -------- 第四張圖：綜合信號 --------
    fig2, ax = plt.subplots(figsize=(16, 8))
    
    # 繪製股價
    ax.plot(signals.index, signals['Price'], label='收盤價', color='black', linewidth=2)
    ax.fill_between(signals.index, signals['Price'].min(), signals['Price'].max(), 
                      where=signals['Combined_Signal'] > 0,
                      color='green', alpha=0.1, label='買入區域')
    ax.fill_between(signals.index, signals['Price'].min(), signals['Price'].max(), 
                      where=signals['Combined_Signal'] < 0,
                      color='red', alpha=0.1, label='賣出區域')
    
    # 繪製綜合信號
    ax2 = ax.twinx()
    ax2.plot(signals.index, signals['Combined_Signal'], 
             label='綜合信號', color='purple', linewidth=2, linestyle='--')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.axhline(y=0.5, color='green', linestyle=':', alpha=0.7, label='買入閾值')
    ax2.axhline(y=-0.5, color='red', linestyle=':', alpha=0.7, label='賣出閾值')
    ax2.set_ylabel('Signal Strength', fontsize=12)
    ax2.set_ylim(-1.5, 1.5)
    
    ax.set_title(f'Combined Trading Signals - {ticker}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    
    # 合併圖例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', prop=zh_font)
    
    plt.tight_layout()
    plt.savefig('HERMES/Skill/combined_signals.png', dpi=150, bbox_inches='tight')
    print('Combined signals chart saved to HERMES/Skill/combined_signals.png')
    
    return fig, fig2


def print_signal_summary(signals, ticker):
    """
    打印信號摘要
    """
    
    print(f'\n{"="*60}')
    print(f'📊 {ticker} 交易信號摘要')
    print(f'{"="*60}')
    
    # 取最新嘅信號
    latest = signals.iloc[-1]
    prev = signals.iloc[-2] if len(signals) > 1 else latest
    
    print(f'\n📅 最新日期：{signals.index[-1].strftime("%Y-%m-%d")}')
    print(f'💰 最新收盤價：{latest["Price"]:.2f}')
    
    print(f'\n🌤️ 組合一（Ichimoku + RSI）：')
    print(f'   RSI 指數：{latest["RSI"]:.2f}')
    if latest['Price'] > latest['Cloud_Upper']:
        position = '📈 雲層上面（多頭）'
    elif latest['Price'] < latest['Cloud_Lower']:
        position = '📉 雲層下面（空頭）'
    else:
        position = '⚖️ 雲層入面（中性）'
    print(f'   位置：{position}')
    print(f'   信號：{latest["Signal_1_Ichimoku_RSI"]:.1f}')
    
    print(f'\n📈 組合二（Ichimoku + MACD）：')
    print(f'   轉換線 vs 基準線：{"▲" if latest["Tenkan"] > latest["Kijun"] else "▼"}')
    print(f'   MACD vs 信號線：{"▲" if latest["MACD"] > latest["MACD_Signal"] else "▼"}')
    print(f'   信號：{latest["Signal_2_Ichimoku_MACD"]:.1f}')
    
    print(f'\n📊 組合三（Ichimoku + MA）：')
    print(f'   49日線：{latest["SMA_49"]:.2f}')
    print(f'   233日線：{latest["SMA_233"]:.2f}')
    print(f'   49日 vs 233日：{"▲ 黃金交叉" if latest["SMA_49"] > latest["SMA_233"] else "▼ 死亡交叉"}')
    print(f'   信號：{latest["Signal_3_Ichimoku_MA"]:.1f}')
    
    print(f'\n🎯 綜合信號：{latest["Combined_Signal"]:.2f}')
    
    # 信號詮釋
    combined = latest['Combined_Signal']
    if combined >= 0.5:
        interpretation = '✅ 強烈買入信號'
    elif combined >= 0.2:
        interpretation = '🟢 輕微買入信號'
    elif combined <= -0.5:
        interpretation = '❌ 強烈賣出信號'
    elif combined <= -0.2:
        interpretation = '🔴 輕微賣出信號'
    else:
        interpretation = '⚪ 中立觀望'
    
    print(f'\n{interpretation}')
    print(f'{"="*60}\n')


# ============ 主程式 ============

def main():
    """
    主程式：下載數據、計算指標、生成信號、繪製圖表
    """
    
    print('🌤️ 一目均衡表 + 黃金組合分析系統')
    print('=' * 50)
    
    # 設定股票代碼
    ticker = '9992.HK'
    
    print(f'\n📥 正在下載 {ticker} 歷史數據...')
    
    # 下載數據（過去2年，等於有足夠數據計200日線）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2年
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f'❌ 無法下載 {ticker} 數據')
            return
        
        print(f'✅ 成功下載 {len(df)} 日數據')
        print(f'   起始日期：{df.index[0].strftime("%Y-%m-%d")}')
        print(f'   結束日期：{df.index[-1].strftime("%Y-%m-%d")}')
        
    except Exception as e:
        print(f'❌ 下載失敗：{e}')
        return
    
    # -------- 計算各項指標 --------
    print('\n⚙️ 正在計算一目均衡表 (參數: 34, 5, 52, 26)...')
    df = calculate_ichimoku(df, tenkan_period=34, kijun_period=5, senkou_b_period=52, cloud_shift=26)
    
    print('⚙️ 正在計算 RSI...')
    df['RSI'] = calculate_rsi(df)
    
    print('⚙️ 正在計算 MACD...')
    df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = calculate_macd(df)
    
    print('⚙️ 正在計算移動平均線 (49日, 233日)...')
    ma_df = calculate_ma(df, periods=[49, 233])
    df['SMA_49'] = ma_df['SMA_49']
    df['SMA_233'] = ma_df['SMA_233']
    
    # -------- 生成信號 --------
    print('⚙️ 正在生成交易信號...')
    signals = generate_signals(df)
    
    # -------- 繪製圖表 --------
    print('\n📊 正在繪製圖表...')
    plot_ichimoku_combinations(df, signals, ticker)
    
    # -------- 打印摘要 --------
    print_signal_summary(signals, ticker)
    
    print('✅ 分析完成！')


if __name__ == '__main__':
    main()
