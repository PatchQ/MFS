#!/usr/bin/env python3
"""
示例：使用 ScriptRunner 調用 ProcessTA

運行方式:
    cd ~/GitHub/MFS/ScriptRunner
    python examples/run_ta.py
    
或直接調用:
    cd ~/GitHub/MFS
    python -m ScriptRunner.examples.run_ta
"""

import sys
import os

# 確保 ScriptRunner 在路徑中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ScriptRunner import ScriptRunner, TARequest


def main():
    # 創建調度器（指向 ProcessTA.py）
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "ProcessTA.py"
    )
    runner = ScriptRunner(f"python {script_path} --json-stdin", max_workers=4)

    # 準備請求列表
    requests = [
        TARequest(sno="0001.HK", stype="L", tdate="2024-01-01", ai="False"),
        TARequest(sno="0002.HK", stype="L", tdate="2024-01-01", ai="False"),
        TARequest(sno="0003.HK", stype="L", tdate="2024-01-01", ai="False"),
    ]

    print("=" * 50)
    print("ScriptRunner TA Example")
    print("=" * 50)
    
    # 執行並打印結果
    for i, response in enumerate(runner.run_ta(requests)):
        print(f"\n[Request {i+1}] {response.sno} ({response.stype})")
        print(f"  Success: {response.success}")
        if response.error:
            print(f"  Error: {response.error}")
        if response.signal is not None:
            print(f"  Signal: {response.signal}")
        if response.indicators:
            print(f"  Indicators: {response.indicators}")

    print("\n" + "=" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
