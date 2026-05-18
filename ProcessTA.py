import sys
import os
import json
import argparse

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  

# Core modules import
from Core.MFSDataHub import MFSDataHub
from Core.IndicatorEngine import IndicatorEngine


def process_ta_json_stdin():
    """
    JSON-line stdin/stdout 模式
    
    輸入格式（每行一個 JSON）:
        {"sno": "0001.HK", "stype": "L", "tdate": "2024-01-01", "ai": "False"}
    
    輸出格式（每行一個 JSON）:
        {"sno": "0001.HK", "stype": "L", "success": true, "signal": true, ...}
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            req = json.loads(line)
            sno = req.get('sno')
            stype = req.get('stype')
            tdate = req.get('tdate')
            ai = req.get('ai', 'False')
            
            # 調用原有邏輯（這裡調用 AnalyzeStock 處理單個股票）
            AnalyzeStock(sno, stype, ai)
            
            # 輸出響應
            response = {
                'sno': sno,
                'stype': stype,
                'success': True,
                'tdate': tdate,
                'error': None
            }
            print(json.dumps(response, ensure_ascii=False))
            sys.stdout.flush()
            
        except Exception as e:
            # 輸出錯誤響應
            response = {
                'sno': req.get('sno', 'unknown'),
                'stype': req.get('stype', 'L'),
                'success': False,
                'error': str(e)
            }
            print(json.dumps(response, ensure_ascii=False))
            sys.stdout.flush()

def AnalyzeStock(sno,stype,ai):

    df = cc.pd.read_csv(cc.PATH+"/"+stype+"/"+sno+".csv",index_col=0) 
    df.index = cc.pd.to_datetime(df.index)  

    df = cc.extendData(df)
    df = cc.convertData(df)

    #EMA
    df = cc.calEMA(df)

    # Initialize Core modules
    datahub = MFSDataHub()
    engine = IndicatorEngine(datahub=datahub)

    # HFH - 使用 IndicatorEngine 包裝器，降級到原有函數
    hfh_result = engine.add_indicator(df.copy(), 'hf_h')
    if hfh_result is not None and not hfh_result.empty:
        df = hfh_result
    else:
        df = cc.calHFH(df)

    #cal HHHL
    HHLLdf = cc.calHHLL(df)

    if HHLLdf is not None:
        if len(HHLLdf)>0:        
            #Boss
            df = cc.checkBoss(df, sno, stype, HHLLdf)
            #HHHL
            df = cc.checkWave(df, sno, stype, HHLLdf)                     
    
    #VCP
    df = cc.checkVCP(df)

    #2006 Indicators - Ichimoku, GBS22C, Breakout200, Fisher
    # Ichimoku - 使用 IndicatorEngine 包裝器，降級到原有函數
    ichimoku_result = engine.add_indicator(df.copy(), 'ichimoku', sno=sno, stype=stype)
    if ichimoku_result is not None and not ichimoku_result.empty:
        df = ichimoku_result
    else:
        df = cc.checkIchimoku(df, sno, stype)

    df = cc.checkGBS22C(df, sno, stype)
    df = cc.checkBreakout200(df, sno, stype)
    df = cc.checkFisher(df, sno, stype)

    #AI Signal
    if ai=="True":
        signals = {}
        for modelname in cc.MODELLIST:
            signals[modelname] = cc.loadModel(modelname, sno, df)

        for modelname, sig in signals.items():
            df[modelname] = sig if len(sig) > 0 else False                        

    df = df.reset_index()
    #df = df.sort_values(by=['index'],ascending=[True])
    df = df[:-10].copy()
    df.to_csv(cc.OUTPATH+"/"+stype+"/P_"+sno+".csv",index=False)
    


def ProcessTA(stype,ai):

    snolist = list(map(lambda s: s.replace(".csv", ""), cc.os.listdir(cc.PATH+"/"+stype)))
    SLIST = cc.pd.DataFrame(snolist, columns=["sno"])
    SLIST = SLIST.assign(stype=stype+"")
    SLIST = SLIST.assign(ai=ai+"")
    SLIST = SLIST[:]

    #print(SLIST)

    with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
        list(cc.tqdm(executor.map(AnalyzeStock,SLIST["sno"],SLIST["stype"],SLIST["ai"],chunksize=1),total=len(SLIST)))


if __name__ == '__main__':
    # 命令行參數解析
    parser = argparse.ArgumentParser(description='ProcessTA - 技術分析')
    parser.add_argument('--json-stdin', action='store_true', 
                        help='JSON-line stdin/stdout 模式')
    args = parser.parse_args()
    
    if args.json_stdin:
        # JSON-line 模式
        process_ta_json_stdin()
    else:
        # 原有模式
        start = cc.t.perf_counter()

        # ProcessTA("L",ai="True")    
        # ProcessTA("M",ai="True")    

        ProcessTA("L",ai="False") 
        ProcessTA("M",ai="False")    

        finish = cc.t.perf_counter()
        print(f'It took {round(finish-start,2)} second(s) to finish.')
    