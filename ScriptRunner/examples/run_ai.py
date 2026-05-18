#!/usr/bin/env python3
"""
示例：使用 ScriptRunner 調用 ProcessAI

運行方式:
    cd ~/GitHub/MFS/ScriptRunner
    python examples/run_ai.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ScriptRunner import ScriptRunner, AIRequest


def main():
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "ProcessAI.py"
    )
    runner = ScriptRunner(f"python {script_path} --json-stdin", max_workers=4)

    requests = [
        AIRequest(sno="0001.HK", stype="L", tdate="2024-01-01", model="RF"),
        AIRequest(sno="0002.HK", stype="L", tdate="2024-01-01", model="SVM"),
        AIRequest(sno="0003.HK", stype="L", tdate="2024-01-01", model="MLP"),
    ]

    print("=" * 50)
    print("ScriptRunner AI Example")
    print("=" * 50)

    for i, response in enumerate(runner.run_ai(requests)):
        print(f"\n[Request {i+1}] {response.sno} ({response.model})")
        print(f"  Success: {response.success}")
        if response.error:
            print(f"  Error: {response.error}")
        if response.prediction is not None:
            print(f"  Prediction: {response.prediction}")
        if response.probability is not None:
            print(f"  Probability: {response.probability:.4f}")

    print("\n" + "=" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
