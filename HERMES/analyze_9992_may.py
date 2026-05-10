import pandas as pd
base = '/root/GitHub/SData/HKEX/SO/9992_POP/'

for date in ['20260505', '20260506', '20260507', '20260508']:
    df = pd.read_csv(base + 'POP_' + date + '.csv')
    may = df[df['month_abbr'] == 'MAY']
    
    df2 = df[(df['strike'] >= 120) & (df['strike'] <= 200)]
    df2 = df2[(df2['call_settle_price'] > 0) & (df2['put_settle_price'] > 0)]
    df2['abs_diff'] = abs(df2['call_settle_price'] - df2['put_settle_price'])
    atm = df2.loc[df2['abs_diff'].idxmin()]
    price = atm['strike'] + atm['call_settle_price'] - atm['put_settle_price']
    
    print('=== ' + date + ' (股價: ' + str(round(price, 2)) + ') ===')
    
    may_155 = may[may['strike'] == 155.0]
    oi_155 = may_155['put_gross'].sum()
    change_155 = may_155['put_gross_change'].sum()
    print('  MAY 155 Put OI: ' + str(int(oi_155)) + ', 變化: ' + str(int(change_155)))
    
    for strike in [150, 152.5, 157.5, 160, 162.5, 165, 170, 175]:
        may_s = may[may['strike'] == strike]
        oi = may_s['put_gross'].sum()
        if oi > 0:
            change = may_s['put_gross_change'].sum()
            print('  MAY ' + str(strike) + ' Put OI: ' + str(int(oi)) + ', 變化: ' + str(int(change)))
    
    # Check CALL side
    may_call = may[may['call_gross'] > 0]
    total_call_oi = int(may_call['call_gross'].sum())
    print('  MAY Call 總OI: ' + str(total_call_oi))
    
    for strike in [160, 165, 170, 175, 180]:
        may_c = may[may['strike'] == strike]
        oi_c = may_c['call_gross'].sum()
        if oi_c > 0:
            change_c = may_c['call_gross_change'].sum()
            print('  MAY ' + str(strike) + ' Call OI: ' + str(int(oi_c)) + ', 變化: ' + str(int(change_c)))
    
    print()
