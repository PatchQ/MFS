"""
Test script to compare backtesting.py vs vectorbt results
"""
import sys
import os
sys.path.append('..')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import UTIL.CommonConfig as cc
import pandas as pd
import numpy as np

# Test parameters
TEST_SNO = "P_0700.HK"
TEST_STYPE = "L"
TEST_SIGNAL = "HFH"
MAX_HOLDBARS = 100
SL = -10.0
TP = 20.0
DD = 0.0

print("=" * 60)
print(f"Testing with stock: {TEST_SNO}, Signal: {TEST_SIGNAL}")
print("=" * 60)

# Load data
file_path = f"{cc.OUTPATH}/{TEST_STYPE}/{TEST_SNO}.csv"
print(f"\nLoading: {file_path}")
df = pd.read_csv(file_path)
df.set_index("index", inplace=True)
df.index = pd.to_datetime(df.index)

print(f"Data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"BOSSB signal count: {df['BOSSB'].sum()}")

# Check signal exists
if TEST_SIGNAL not in df.columns:
    print(f"ERROR: Signal {TEST_SIGNAL} not found in data")
    sys.exit(1)

# Run old backtesting.py version
print("\n" + "=" * 60)
print("Running OLD backtesting.py version...")
print("=" * 60)

from Run_Backtest_bk import runBacktest as runBacktest_old

old_result = runBacktest_old(TEST_SNO, TEST_STYPE, TEST_SIGNAL, MAX_HOLDBARS, SL, TP, DD)
print("\nOLD version result:")
print(old_result.to_string() if len(old_result) > 0 else "No result (no trades)")

# Run new vectorbt version
print("\n" + "=" * 60)
print("Running NEW vectorbt version...")
print("=" * 60)

from Run_Backtest import runBacktest as runBacktest_new

new_result = runBacktest_new(TEST_SNO, TEST_STYPE, TEST_SIGNAL, MAX_HOLDBARS, SL, TP, DD)
print("\nNEW version result:")
print(new_result.to_string() if len(new_result) > 0 else "No result (no trades)")

# Compare results
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

if len(old_result) > 0 and len(new_result) > 0:
    metrics = ['returns', 'trades_counts', 'win_rates', 'RR', 'SQN', 
               'sharpe_ratios', 'sortino_ratios', 'calmar_ratios',
               'avg_trade', 'best_trade', 'worst_trade']
    
    print(f"\n{'Metric':<20} {'Old':>15} {'New':>15} {'Diff':>15}")
    print("-" * 65)
    for m in metrics:
        if m in old_result.columns and m in new_result.columns:
            old_val = old_result[m].values[0]
            new_val = new_result[m].values[0]
            diff = new_val - old_val
            print(f"{m:<20} {old_val:>15.4f} {new_val:>15.4f} {diff:>15.4f}")
else:
    print("\nCannot compare - one or both versions returned no result")

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
