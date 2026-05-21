#!/usr/bin/env python3
"""
HK Stock Rebound Scanner
Daily 3 picks with rebound potential at 09:00 on weekdays
"""
import requests
import numpy as np
import pandas as pd
from datetime import datetime

# ============ CONFIG ============
TOP_N = 3
MIN_PRICE = 5.0
RSI_MIN, RSI_MAX = 40, 60
HV_MIN, HV_MAX = 30, 65
SUPPORT_DISTANCE = 0.08  # Within 8% of Gann 50% support

def get_large_cap_hk_stocks():
    return [
        ("0700.HK", "騰訊控股", "科網"),
        ("9618.HK", "京東集團", "科網"),
        ("9988.HK", "阿里巴巴", "科網"),
        ("3690.HK", "美團", "科網"),
        ("1810.HK", "小米集團", "科網"),
        ("2382.HK", "聯想集團", "科網"),
        ("0688.HK", "中海地產", "地產"),
        ("1109.HK", "華潤置地", "地產"),
        ("1211.HK", "比亞迪股份", "電車"),
        ("175.HK", "吉利汽車", "電車"),
        ("2319.HK", "蒙牛乳業", "消費"),
        ("2020.HK", "安踏體育", "消費"),
        ("1177.HK", "中國生物製藥", "醫藥"),
        ("2628.HK", "中國人壽", "保險"),
        ("1299.HK", "友邦保險", "保險"),
        ("2318.HK", "中國平安", "保險"),
        ("0939.HK", "建設銀行", "銀行"),
        ("3988.HK", "中國銀行", "銀行"),
        ("3968.HK", "招商銀行", "銀行"),
        ("3328.HK", "交通銀行", "銀行"),
        ("0388.HK", "港交所", "金融"),
        ("1038.HK", "長江基建", "公用"),
        ("0001.HK", "長實集團", "地產"),
        ("0016.HK", "新鴻基地產", "地產"),
        ("0027.HK", "銀河娛樂", "博彩"),
        ("0192.HK", "金沙中國", "博彩"),
        ("0823.HK", "領展房產", "REIT"),
        ("0883.HK", "煤氣公司", "公用"),
        ("1093.HK", "石藥集團", "醫藥"),
        ("1171.HK", "兗礦能源", "能源"),
        ("1988.HK", "民生銀行", "銀行"),
        ("2196.HK", "復星醫藥", "醫藥"),
        ("2888.HK", "恆生銀行", "銀行"),
        ("3818.HK", "金風科技", "替代能源"),
        ("3969.HK", "中國中鐵", "基建"),
        ("4898.HK", "東風集團", "電車"),
        ("6949.HK", "首都機場", "公用"),
        ("7202.HK", "百濟神州", "生物科技"),
        ("9626.HK", "嗶哩嗶哩", "科網"),
        ("9696.HK", "金山軟件", "科網"),
        ("9818.HK", "網易", "科網"),
        ("9888.HK", "百度集團", "科網"),
        ("9961.HK", "攜程集團", "科網"),
        ("9987.HK", "百勝中國", "消費"),
        ("9966.HK", "康方生物", "生物科技"),
    ]

def fetch_stock_data(symbol):
    try:
        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'
        params = {'interval': '1d', 'range': '2y', 'includePrePost': 'false'}
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, params=params, headers=headers, timeout=15)
        data = r.json()
        result = data['chart']['result'][0]
        closes_raw = result['indicators']['quote'][0]['close']
        highs_raw = result['indicators']['quote'][0]['high']
        lows_raw = result['indicators']['quote'][0]['low']
        volumes_raw = result['indicators']['quote'][0]['volume']
        
        # Filter all together using same mask (aligned arrays)
        mask = np.array([x is not None for x in closes_raw])
        closes = np.array(closes_raw)[mask].astype(float)
        highs = np.array(highs_raw)[mask].astype(float)
        lows = np.array(lows_raw)[mask].astype(float)
        volumes = np.array(volumes_raw)[mask].astype(float)
        
        if len(closes) < 100:
            return None, None, None, None, None
        return closes, highs, lows, volumes
    except Exception:
        return None, None, None, None, None

def calc_rsi(prices, n=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:n])
    avg_loss = np.mean(losses[:n])
    for i in range(n, len(gains)):
        avg_gain = (avg_gain * (n-1) + gains[i]) / n
        avg_loss = (avg_loss * (n-1) + losses[i]) / n
        rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calc_ema(closes, span):
    return pd.Series(closes).ewm(span=span, adjust=False).mean().values

def analyze_stock(symbol, name, sector):
    result = fetch_stock_data(symbol)
    if result[0] is None:
        return None
    closes, highs, lows, volumes = result
    
    current = round(closes[-1], 2)
    major_high = float(np.max(highs))
    major_low = float(np.min(lows))
    
    if current < MIN_PRICE:
        return None
    
    rsi = calc_rsi(closes)
    if rsi < RSI_MIN or rsi > RSI_MAX:
        return None
    
    # HV
    log_returns = np.log(closes[-20:] / closes[-21:-1])
    hv = float(np.std(log_returns) * np.sqrt(252) * 100)
    if hv < HV_MIN or hv > HV_MAX:
        return None
    
    # EMA200
    ema200 = calc_ema(closes, 200)
    ema200_val = float(ema200[-1])
    above_ema200 = current > ema200_val
    ema200_dist_pct = (current - ema200_val) / current * 100
    
    # Gann 50% support (main support level)
    vib = major_high - major_low
    gann_50_level = float(major_low + vib * 0.50)
    gann_50_dist = (current - gann_50_level) / current if gann_50_level < current else None
    
    # Also collect other levels for reference
    gann_supports = []
    for pct in [0.786, 0.886, 0.618, 0.50]:
        level = float(major_low + vib * (1 - pct))
        if level < current:
            dist = (current - level) / current
            gann_supports.append((level, pct, dist))
    gann_supports.sort(key=lambda x: x[2])
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
    
    # Volume
    avg_vol = float(np.mean(volumes[-20:]))
    vol_ratio = float(volumes[-1] / avg_vol) if avg_vol > 0 else 0
    
    # Nearest resistance
    nearest_resist = None
    for pct in [0.236, 0.382, 0.50]:
        level = float(major_low + vib * (1 - pct))
        if level > current:
            dist = (level - current) / current
            if nearest_resist is None or dist < nearest_resist[2]:
                nearest_resist = (level, pct, dist)
    
    target = round(nearest_resist[0], 2) if nearest_resist else round(current * 1.05, 2)
    upside = (target - current) / current * 100
    
    # ============ SCORING ============
    score = 0.0
    
    # RSI in rebound zone (40-60)
    if RSI_MIN <= rsi <= RSI_MAX:
        score += 2.0
        if rsi >= 40 and rsi <= 50:
            score += 0.5  # Extra for being in sweet spot
    
    # Above 200 EMA (rewards but not required)
    if above_ema200:
        score += 1.5
    else:
        if ema200_dist_pct > -20:
            score += 0.5
    
    # Gann 50% support - MAIN CONDITION
    if gann_50_dist is not None and gann_50_dist < 0.08:
        score += 3.0  # Heavy weight on 50% support
        if gann_50_dist < 0.04:
            score += 0.5  # Extra close to 50%
    
    # Nearest Gann support (any level)
    if nearest_gann and nearest_gann[2] < SUPPORT_DISTANCE:
        score += 1.0
    
    # Square of 9 support
    if so9_support and so9_support[1] / current < 0.05:
        score += 0.5
    
    # Volume spike
    if vol_ratio >= 1.15:
        score += 1.0
    
    # HV moderate
    if 30 <= hv <= 65:
        score += 0.5
    
    return {
        'symbol': symbol, 'name': name, 'sector': sector,
        'current': current, 'rsi': round(rsi, 1), 'hv': round(hv, 1),
        'ema200': round(ema200_val, 2), 'ema200_pct': round(ema200_dist_pct, 1),
        'above_ema200': above_ema200,
        'gann_50_level': round(gann_50_level, 2) if gann_50_level else None,
        'gann_50_dist': round(gann_50_dist * 100, 1) if gann_50_dist is not None else None,
        'nearest_gann_support': round(nearest_gann[0], 2) if nearest_gann else None,
        'nearest_gann_pct': f"{nearest_gann[1]*100:.1f}%" if nearest_gann else None,
        'nearest_gann_dist': round(nearest_gann[2]*100, 1) if nearest_gann else None,
        'so9_support': so9_support[0] if so9_support else None,
        'nearest_resist': round(nearest_resist[0], 2) if nearest_resist else None,
        'target': target, 'upside': round(upside, 1),
        'vol_ratio': round(vol_ratio, 2),
        'score': score,
    }

def run_scan():
    results = []
    for symbol, name, sector in get_large_cap_hk_stocks():
        r = analyze_stock(symbol, name, sector)
        if r:
            results.append(r)
    results.sort(key=lambda x: -x['score'])
    return results[:TOP_N]

def get_rebound_reasons(r):
    reasons = []
    if r['above_ema200']:
        reasons.append(f"現價 {r['ema200_pct']:+.1f}% 高於200EMA，中線未壞")
    else:
        reasons.append(f"現價低於200EMA {abs(r['ema200_pct']):.1f}%，但未遠")
    if r.get('gann_50_level') and r.get('gann_50_dist') is not None:
        reasons.append(f"🎯 Gann 50% 支撐 {r['gann_50_level']}（距{r['gann_50_dist']}%）")
    elif r['nearest_gann_support'] and r['nearest_gann_dist']:
        reasons.append(f"Gann {r['nearest_gann_pct']} 支撐 {r['nearest_gann_support']}（距{r['nearest_gann_dist']}%）")
    if r['so9_support']:
        reasons.append(f"Square of 9 {r['so9_support']} 形成支撐")
    if r['vol_ratio'] >= 1.15:
        reasons.append(f"成交量放量（今日量係20日均量的 {r['vol_ratio']}x）")
    if r['rsi'] >= 40 and r['rsi'] <= 55:
        reasons.append(f"RSI {r['rsi']} 調整完畢，準備向上")
    return reasons

def format_report(results, date_str):
    if not results:
        return (f"📈 **今日三隻向好港股** ({date_str})\n\n"
                "⚠️ 今日掃描完畢，冇發現特別理想既反彈機會。\n"
                "可能市場偏弱或偏貴，等下次機會。")
    
    lines = [f"📈 **今日三隻向好港股** ({date_str})",
             "🔍 條件：RSI 40-60 | 波幅 30-65% | Gann 50% 支撐 | 成交量放大",
             ""]
    lines.append("━" * 55)
    
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r['symbol']}** {r['name']}")
        lines.append(f"   📍 現價: {r['current']} | RSI: {r['rsi']} | 波幅: {r['hv']}%")
        lines.append(f"   💡 反彈邏輯：")
        for reason in get_rebound_reasons(r):
            lines.append(f"      • {reason}")
        lines.append(f"   🎯 短線目標: {r['target']}（↑{r['upside']}%）")
        if r['nearest_resist']:
            lines.append(f"   🔴 阻力位: {r['nearest_resist']}")
        lines.append("")
    
    lines.extend(["━" * 55,
                  "⏰ 港股開市9:30，報告9:00發送",
                  "💡 方法: Gann 50% + Square of 9 + RSI + EMA200 + 成交量",
                  "⚠️ 僅供參考，不構成投資建議。"])
    return "\n".join(lines)

if __name__ == "__main__":
    date_str = datetime.now().strftime("%Y-%m-%d")
    results = run_scan()
    report = format_report(results, date_str)
    print(report)
    with open("/tmp/hk_rebound_picks.txt", "w", encoding="utf-8") as f:
        f.write(report)