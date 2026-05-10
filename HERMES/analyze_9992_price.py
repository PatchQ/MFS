import pandas as pd

base = '/root/GitHub/SData/HKEX/SO/9992_POP/'
dates = ['20260401', '20260414', '20260430', '20260505', '20260508']

for date in dates:
    df = pd.read_csv(base + 'POP_' + date + '.csv')
    df = df[(df['strike'] >= 120) & (df['strike'] <= 200) & (df['call_settle_price'] > 0) & (df['put_settle_price'] > 0)]
    df['abs_diff'] = abs(df['call_settle_price'] - df['put_settle_price'])
    atm = df.loc[df['abs_diff'].idxmin()]
    s = atm['strike']
    c = atm['call_settle_price']
    p = atm['put_settle_price']
    print(f'{date}: ATM行使價={s}, Call={c}, Put={p}, 估算股價={s+c-p:.2f}')

# Also show a few more dates around key dates
print('\n=== 4月份關鍵日期股價 ===')
for date in ['20260408', '20260409', '20260410', '20260413', '20260417', '20260421', '20260423', '20260428', '20260429']:
    df = pd.read_csv(base + 'POP_' + date + '.csv')
    df = df[(df['strike'] >= 120) & (df['strike'] <= 200) & (df['call_settle_price'] > 0) & (df['put_settle_price'] > 0)]
    df['abs_diff'] = abs(df['call_settle_price'] - df['put_settle_price'])
    atm = df.loc[df['abs_diff'].idxmin()]
    s = atm['strike']
    c = atm['call_settle_price']
    p = atm['put_settle_price']
    print(f'{date}: ATM行使價={s}, Call={c}, Put={p}, 估算股價={s+c-p:.2f}')