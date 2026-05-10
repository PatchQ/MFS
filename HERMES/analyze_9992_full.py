import pandas as pd
import glob
from datetime import datetime

base = '/root/GitHub/SData/HKEX/SO/9992_POP/'

def estimate_price(df, date_str):
    """用Put-Call Parity估算股價: S = C - P + K"""
    df2 = df[(df['strike'] >= 120) & (df['strike'] <= 200)]
    df2 = df2[(df2['call_settle_price'] > 0) & (df2['put_settle_price'] > 0)]
    if len(df2) == 0:
        return None
    df2['abs_diff'] = abs(df2['call_settle_price'] - df2['put_settle_price'])
    atm = df2.loc[df2['abs_diff'].idxmin()]
    return atm['strike'] + atm['call_settle_price'] - atm['put_settle_price']

def get_top_oi_strikes(df, month, top_n=5):
    """搵出OI最大的行使價"""
    df_m = df[df['month_abbr'] == month]
    if len(df_m) == 0:
        return []
    oi_by_strike = df_m.groupby('strike')['put_gross'].sum().sort_values(ascending=False)
    return oi_by_strike.head(top_n)

def find_significant_changes(df, month, prev_df=None, threshold=1000):
    """搵出OI變化大的行使價"""
    df_m = df[df['month_abbr'] == month]
    if len(df_m) == 0:
        return []
    
    results = []
    for strike in df_m['strike'].unique():
        curr_oi = df_m[df_m['strike'] == strike]['put_gross'].sum()
        if prev_df is not None:
            prev_m = prev_df[prev_df['month_abbr'] == month]
            prev_oi = prev_m[prev_m['strike'] == strike]['put_gross'].sum() if len(prev_m) > 0 else 0
        else:
            prev_oi = 0
        
        change = curr_oi - prev_oi
        if abs(change) >= threshold:
            results.append((strike, curr_oi, change))
    
    return sorted(results, key=lambda x: abs(x[2]), reverse=True)

# 定義分析區間
periods = [
    ('2026-02-01', '2026-02-28', ['MAR', 'APR']),
    ('2026-03-01', '2026-03-31', ['MAR', 'APR']),
    ('2026-04-01', '2026-04-30', ['APR', 'MAY']),
]

files = sorted(glob.glob(base + 'POP_2026*.csv'))

prev_df = None
prev_date = None

for start_date, end_date, months in periods:
    print('\n' + '='*80)
    print(f'=== {start_date} 至 {end_date} ===')
    print(f'=== 分析月份: {months} ===')
    print('='*80)
    
    for f in files:
        date_str = f.split('_')[-1].replace('.csv', '')
        date = datetime.strptime(date_str, '%Y%m%d')
        
        if not (datetime.strptime(start_date, '%Y-%m-%d') <= date <= datetime.strptime(end_date, '%Y-%m-%d')):
            continue
        
        df = pd.read_csv(f)
        price = estimate_price(df, date_str)
        
        print(f'\n--- {date_str} (收市價: {price:.2f}) ---')
        
        for month in months:
            df_m = df[df['month_abbr'] == month]
            if len(df_m) == 0:
                continue
            
            # 搵最大OI行使價
            top_strikes = get_top_oi_strikes(df, month, 3)
            
            # 搵顯著變化
            prev_df_m = prev_df[prev_df['month_abbr'] == month] if prev_df is not None else None
            changes = find_significant_changes(df, month, prev_df, threshold=500)
            
            # 只顯示有變化的
            if changes:
                print(f'  [{month}月] 大幅變化:')
                for strike, oi, change in changes[:5]:
                    sign = '+' if change > 0 else ''
                    print(f'    Strike {strike}: OI={int(oi)}, 變化={sign}{int(change)}')
            
            if top_strikes is not None and len(top_strikes) > 0:
                print(f'  [{month}月] 最大OI: {dict(top_strikes.head(3))}')
        
        prev_df = df
        prev_date = date_str

print('\n\n=== 完成 ===')
