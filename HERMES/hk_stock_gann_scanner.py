#!/usr/bin/env python3
"""
HK Stock Scanner - Gann + Technical Analysis
Medium-term buy opportunities with moderate volatility
Runs daily via cronjob → outputs to Telegram
"""
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json, sys

# ============ CONFIG ============
TOP_N = 20  # Top N stocks to report
MIN_PRICE = 5.0  # HKD
RSI_MIN, RSI_MAX = 35, 50
HV_MIN, HV_MAX = 25, 45  # Historical Volatility %
SUPPORT_DISTANCE = 0.10  # Within 10% of Gann support

# ============ Stock Universe ============
def get_large_cap_hk_stocks():
    stocks = [
        ("0700.HK", "騰訊控股", "科網"),
        ("9618.HK", "京東集團", "科網"),
        ("9988.HK", "阿里巴巴", "科網"),
        ("3690.HK", "美團", "科網"),
        ("1810.HK", "小米集團", "科網"),
        ("2382.HK", "聯想集團", "科網"),
        ("0688.HK", "中國海外發展", "地產"),
        ("1109.HK", "華潤置地", "地產"),
        ("1211.HK", "比亞迪股份", "電車"),
        ("175.HK", "吉利汽車", "電車"),
        ("2319.HK", "蒙牛乳業", "消費"),
        ("2020.HK", "安踏體育", "消費"),
        ("1044.HK", "恆安國際", "消費"),
        ("1177.HK", "中國生物製藥", "醫藥"),
        ("2628.HK", "中國人壽", "保險"),
        ("1299.HK", "友邦保險", "保險"),
        ("2318.HK", "中國平安", "保險"),
        ("0939.HK", "建設銀行", "銀行"),
        ("3988.HK", "中國銀行", "銀行"),
        ("3968.HK", "招商銀行", "銀行"),
        ("3328.HK", "交通銀行", "銀行"),
        ("0388.HK", "香港交易所", "金融"),
        ("1038.HK", "長江基建", "公用"),
        ("0001.HK", "長實集團", "地產"),
        ("0016.HK", "新鴻基地產", "地產"),
        ("0012.HK", "恆基地產", "地產"),
        ("0027.HK", "銀河娛樂", "博彩"),
        ("0192.HK", "金沙中國", "博彩"),
        ("1113.HK", "長江實業", "地產"),
        ("0256.HK", "冠君產業", "地產"),
        ("0823.HK", "領展房產", "REIT"),
        ("0883.HK", "香港中華煤氣", "公用"),
        ("1093.HK", "石藥集團", "醫藥"),
        ("1171.HK", "兗礦能源", "能源"),
        ("1882.HK", "海天國際", "機械"),
        ("1988.HK", "民生銀行", "銀行"),
        ("2196.HK", "復星醫藥", "醫藥"),
        ("2600.HK", "中國鋁業", "金屬"),
        ("2888.HK", "恆生銀行", "銀行"),
        ("3818.HK", "金風科技", "替代能源"),
        ("3969.HK", "中國中鐵", "基建"),
        ("4898.HK", "東風集團", "電車"),
        ("6869.HK", "長飛光纖光纜", "電信"),
        ("6949.HK", "北京首都機場", "公用"),
        ("7202.HK", "百濟神州", "生物科技"),
        ("9626.HK", "嗶哩嗶哩", "科網"),
        ("9696.HK", "金山軟件", "科網"),
        ("9818.HK", "網易", "科網"),
        ("9888.HK", "百度集團", "科網"),
        ("9961.HK", "攜程集團", "科網"),
        ("9987.HK", "百勝中國", "消費"),
    ]
    return stocks

# ============ Fetch Data ============
def fetch_stock_data(symbol):
    try:
        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'
        params = {'interval': '1d', 'range': '1y', 'includePrePost': 'false'}
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, params=params, headers=headers, timeout=15)
        data = r.json()
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        closes = np.array(result['indicators']['quote'][0]['close'])
        highs = np.array(result['indicators']['quote'][0]['high'])
        lows = np.array(result['indicators']['quote'][0]['low'])
        dates = [datetime.fromtimestamp(t) for t in timestamps]
        mask = ~np.isnan(closes)
        dates = np.array(dates)[mask]
        closes = closes[mask]
        highs = highs[mask]
        lows = lows[mask]
        return dates, closes, highs, lows
    except Exception:
        return None, None, None, None

def calc_rsi(prices, n=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    if len(gains) < n:
        return None
    avg_gain = np.mean(gains[:n])
    avg_loss = np.mean(losses[:n])
    rsi_list = []
    for i in range(n, len(gains)):
        avg_gain = (avg_gain * (n-1) + gains[i]) / n
        avg_loss = (avg_loss * (n-1) + losses[i]) / n
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_list.append(100 - (100 / (1 + rs)))
    return rsi_list[-1] if rsi_list else None

def calc_hv(closes, n=20):
    if len(closes) < n + 1:
        return None
    log_returns = np.log(closes[-n:] / closes[-n-1:-1])
    return np.std(log_returns) * np.sqrt(252) * 100

def calc_ema200(closes):
    if len(closes) < 200:
        return None
    ema = pd.Series(closes).ewm(span=200, adjust=False).mean().values
    return ema[-1]

def analyze_stock(symbol, name, sector):
    dates, closes, highs, lows = fetch_stock_data(symbol)
    if dates is None or len(closes) < 60:
        return None
    
    current = round(closes[-1], 2)
    major_high = float(np.max(highs))
    major_low = float(np.min(lows))
    
    if current < MIN_PRICE:
        return None
    
    rsi = calc_rsi(closes)
    if rsi is None or rsi < RSI_MIN or rsi > RSI_MAX:
        return None
    
    hv = calc_hv(closes)
    if hv is None or hv < HV_MIN or hv > HV_MAX:
        return None
    
    ema200 = calc_ema200(closes)
    if ema200 is None:
        return None
    
    above_ema200 = current > ema200
    
    # Gann swing levels
    vib = major_high - major_low
    gann_supports = []
    for pct in [0.786, 0.886]:
        level = major_low + vib * (1 - pct)
        if level < current:
            dist = (current - level) / current
            gann_supports.append((level, pct, dist))
    gann_supports.sort(key=lambda x: x[2])  # closest first
    nearest_gann = gann_supports[0] if gann_supports else None
    
    # Square of 9
    sqrt_c = current ** 0.5
    n_floor = int(sqrt_c)
    so9_support = None
    for n in range(n_floor - 2, n_floor + 1):
        sq = n ** 2
        if sq < current:
            diff = current - sq
            if so9_support is None or diff > so9_support[1]:
                so9_support = (sq, diff)
    
    # Gann score
    score = 0.0
    if RSI_MIN <= rsi <= RSI_MAX:
        score += 1.0
    if above_ema200:
        score += 1.0
    if nearest_gann and nearest_gann[2] < SUPPORT_DISTANCE:
        score += 1.0
    if so9_support and so9_support[1] / current < 0.05:
        score += 0.5
    if HV_MIN <= hv <= HV_MAX:
        score += 0.5
    
    return {
        'symbol': symbol,
        'name': name,
        'sector': sector,
        'current': current,
        'rsi': round(rsi, 1),
        'hv': round(hv, 1),
        'ema200': round(ema200, 2),
        'ema200_pct': round((current - ema200) / current * 100, 1),
        'above_ema200': above_ema200,
        'major_high': round(major_high, 2),
        'major_low': round(major_low, 2),
        'nearest_gann_support': round(nearest_gann[0], 2) if nearest_gann else None,
        'nearest_gann_pct': f"{nearest_gann[1]*100:.1f}%" if nearest_gann else None,
        'nearest_gann_dist': round(nearest_gann[2] * 100, 1) if nearest_gann else None,
        'so9_support': so9_support[0] if so9_support else None,
        'score': score,
    }

def run_scan():
    stocks = get_large_cap_hk_stocks()
    results = []
    for symbol, name, sector in stocks:
        r = analyze_stock(symbol, name, sector)
        if r:
            results.append(r)
    results.sort(key=lambda x: (-x['score'], x['rsi']))
    return results[:TOP_N]

def format_report(results, date_str):
    if not results:
        return ("⚠️ 今日掃描完畢，但冇發現符合條件的股票。\n\n"
                "篩選條件：RSI 35~50 | 波幅 25~45% | 現價>200EMA | Gann支撐<10%\n"
                "可能表示市場偏貴，等下次機會。")
    
    lines = []
    lines.append(f"📊 **港股中線買入機會掃描**")
    lines.append(f"📅 {date_str} | 符合條件: {len(results)} 隻")
    lines.append(f"🔍 條件: RSI 35-50 | 波幅 25-45% | 現價>200EMA | Gann支撐<10%")
    lines.append("")
    lines.append("━" * 55)
    
    for i, r in enumerate(results, 1):
        stars = "⭐" * int(r['score'])
        gann_info = (f"{r['nearest_gann_support']} ({r['nearest_gann_pct']}) "
                     f"距{r['nearest_gann_dist']}%") if r['nearest_gann_support'] else "N/A"
        so9_info = f"21²={r['so9_support']}" if r['so9_support'] else ""
        
        lines.append(f"{i:2}. **{r['symbol']}** {r['name']} {stars}")
        lines.append(f"   💰 現價: {r['current']} | RSI: {r['rsi']} | HV: {r['hv']}%")
        lines.append(f"   📈 200EMA: {r['ema200']} (現價{r['ema200_pct']:+.1f}%)")
        lines.append(f"   🛡 Gann支撐: {gann_info} | SO9: {so9_info}")
        lines.append(f"   📊 區間: {r['major_low']} ~ {r['major_high']}")
        lines.append("")
    
    lines.append("━" * 55)
    lines.append("💡 方法: Gann Swing + Square of 9 + RSI + EMA200")
    lines.append("⚠️ 僅供參考，不構成投資建議。")
    
    return "\n".join(lines)

if __name__ == "__main__":
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    results = run_scan()
    report = format_report(results, date_str)
    print(report)
    with open("/tmp/hk_stock_scan_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Saved to /tmp/hk_stock_scan_report.txt]")