import sys
import os
import json
import argparse

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import UTIL.CommonConfig as cc  


def process_ai_json_stdin():
    """
    JSON-line stdin/stdout 模式
    
    輸入格式（每行一個 JSON）:
        {"sno": "0001.HK", "stype": "L", "tdate": "2024-01-01", "model": "RF"}
    
    輸出格式（每行一個 JSON）:
        {"sno": "0001.HK", "stype": "L", "model": "RF", "success": true, ...}
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
            model = req.get('model', 'RF')
            
            # 調用原有邏輯（這裡調用對應的 model function）
            # 根據 model 選擇對應的函數
            model_map = {
                'RF': cc.RF,
                'SVM': cc.SVM,
                'MLP': cc.MLP
            }
            modelfunction = model_map.get(model, cc.RF)
            
            # 調用 AI 處理函數
            CalSingleAI(sno, stype, tdate, modelfunction)
            
            # 輸出響應
            response = {
                'sno': sno,
                'stype': stype,
                'model': model,
                'success': True,
                'tdate': tdate,
                'error': None
            }
            print(json.dumps(response, ensure_ascii=False))
            sys.stdout.flush()
            
        except Exception as e:
            response = {
                'sno': req.get('sno', 'unknown'),
                'stype': req.get('stype', 'L'),
                'model': req.get('model', 'RF'),
                'success': False,
                'error': str(e)
            }
            print(json.dumps(response, ensure_ascii=False))
            sys.stdout.flush()


def CalSingleAI(sno, stype, tdate, modelfunction):
    """處理單個股票的 AI 分析"""
    try:
        tempdf = modelfunction(sno, stype, tdate)
        return tempdf
    except Exception as e:
        raise RuntimeError(f"CalSingleAI failed for {sno}: {e}")

def ProcessAI(stype,modelfunction,tdate):

   resultdf = cc.pd.DataFrame()

   snolist = list(map(lambda s: s.replace(".csv", ""), cc.os.listdir(cc.OUTPATH+"/"+stype)))
   SLIST = cc.pd.DataFrame(snolist, columns=["sno"])   
   SLIST = SLIST.assign(stype=stype+"")
   SLIST = SLIST.assign(tdate=tdate+"")
   SLIST = SLIST[:]

    #print(SLIST)

   with cc.ExecutorType(max_workers=cc.DEFAULT_MAX_WORKERS) as executor:
       for tempdf in cc.tqdm(executor.map(modelfunction,SLIST["sno"],SLIST["stype"],SLIST["tdate"],chunksize=1),total=len(SLIST)):           
           resultdf = cc.pd.concat([tempdf, resultdf], ignore_index=True)

   #resultdf.to_csv(f"data/{stype}_{model}.csv",index=False)    


def CalAI(stype):
    MODELLIST = [cc.RF,cc.SVM,cc.MLP]    

    for modelfunction in MODELLIST:
        print(modelfunction.__name__)
        ProcessAI(stype,modelfunction,cc.DATADATE)    
    


if __name__ == '__main__':
    # 命令行參數解析
    parser = argparse.ArgumentParser(description='ProcessAI - AI 分析')
    parser.add_argument('--json-stdin', action='store_true',
                        help='JSON-line stdin/stdout 模式')
    args = parser.parse_args()
    
    if args.json_stdin:
        # JSON-line 模式
        process_ai_json_stdin()
    else:
        # 原有模式
        start = cc.t.perf_counter()
        
        CalAI("L")
        CalAI("M")

        finish = cc.t.perf_counter()
        print(f'It took {round(finish-start,2)} second(s) to finish.')
    