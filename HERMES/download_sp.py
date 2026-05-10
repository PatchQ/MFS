"""
HKEX 收市結算價資料下載器 (sp*.dat)
===================================
URL Pattern: https://www.hkex.com.hk/eng/stat/dmstat/datadownload/sp{YYYYMMDD}.dat

格式分析:
- Field 1 (11 chars): [line_number(8)] + [stock_code(3)]
- Field 2 (可變): [C/P] + [YYMM] + [strike]
  - 短 strike (<1000): 分開做第三個 field
  - 長 strike (>=1000): 合併到第二個 field
- Field 3 (可變, 短 strike): strike price (5 digits, 值/100)
- Field 4 (9 或 12 chars): 
  - 12 chars: settle(6 digits, 值/100) + volume(6 digits)
  - 9 chars: volume(6 or 9 digits) only

用法:
  python download_sp.py                      # 今日
  python download_sp.py --date 20260508     # 指定日期
  python download_sp.py --date 20260508 --output ./data  # 指定輸出目錄
"""
import sys
import os
import re
import argparse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import urllib.request
import zipfile
import io

HK_TZ = ZoneInfo('Asia/Hong_Kong')

# ============================================================
# 路徑設定
# ============================================================
DEFAULT_OUTPUT = Path("/root/GitHub/SData/HKEX/SP")
DEFAULT_BASE_URL = "https://www.hkex.com.hk/eng/stat/dmstat/datadownload"

# ============================================================
# 解析函數
# ============================================================
def parse_sp_record(line: str) -> dict | None:
    """
    解析 sp*.dat 的一行記錄
    返回: {
        'line_no': int,
        'stock_code': str,      # 如 "A50", "TCH"
        'type': str,            # "Call" or "Put"
        'expiry': str,          # "YYYY-MM"
        'strike': float,        # 行使價 (已除100)
        'settle': float | None, # 結算價 (已除100)
        'volume': int,          # 成交量
    }
    """
    parts = line.split()
    if len(parts) == 0:
        return None
    
    try:
        if len(parts) == 4:
            # 短 strike 格式: [line+code] [C/P+YYMM] [strike] [settle+vol]
            raw = parts[0]
            line_no = int(raw[:8])
            stock_code = raw[8:].strip()
            cp = parts[1][0]
            expiry = parts[1][1:5]
            strike_raw = parts[2].strip()
            settle_vol = parts[3].strip()
            
        elif len(parts) == 3:
            # 長 strike 格式: [line+code] [C/P+YYMM+strike] [settle+vol]
            raw = parts[0]
            line_no = int(raw[:8])
            stock_code = raw[8:].strip()
            cp = parts[1][0]
            expiry = parts[1][1:5]
            strike_raw = parts[1][5:].strip()
            settle_vol = parts[2].strip()
        else:
            return None
        
        # 解析 strike
        strike = int(strike_raw) / 100.0
        
        # 解析 expiry (YYMM -> YYYY-MM)
        expiry_yy = expiry[:2]
        expiry_mm = expiry[2:4]
        expiry_formatted = f"20{expiry_yy}-{expiry_mm}"
        
        # 解析 settle + volume
        if len(settle_vol) == 12:
            settle_raw = settle_vol[:6]
            vol_raw = settle_vol[6:]
            settle = int(settle_raw) / 100.0 if settle_raw != '000000' else 0.0
            volume = int(vol_raw)
        elif len(settle_vol) == 9:
            settle = None
            volume = int(settle_vol)
        elif len(settle_vol) == 6:
            settle = None
            volume = int(settle_vol)
        else:
            settle = None
            volume = 0
        
        return {
            'line_no': line_no,
            'stock_code': stock_code,
            'type': 'Call' if cp == 'C' else 'Put',
            'expiry': expiry_formatted,
            'strike': strike,
            'settle': settle,
            'volume': volume,
        }
        
    except Exception as e:
        print(f"    解析失敗: {line[:50]}... Error: {e}")
        return None


def parse_sp_file(content: bytes) -> list[dict]:
    """解析完整的 sp*.dat 文件"""
    lines = content.split(b'\r\n')
    records = []
    eof_found = False
    
    for line_bytes in lines:
        line = line_bytes.decode('ascii', errors='replace').strip()
        
        if line.startswith('EOF'):
            eof_found = True
            break
        
        if not line:
            continue
        
        rec = parse_sp_record(line)
        if rec:
            records.append(rec)
    
    return records, eof_found


# ============================================================
# 下載函數
# ============================================================
def download_sp_file(date_str: str, output_dir: Path = None) -> Path | None:
    """
    下載指定日期的 sp*.dat 文件
    date_str: YYYYMMDD 格式
    返回: 下載的檔案路徑
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"sp{date_str}.dat"
    url = f"{DEFAULT_BASE_URL}/{filename}"
    output_path = output_dir / filename
    
    print(f"  下載: {url}")
    print(f"  保存: {output_path}")
    
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()
        
        with open(output_path, 'wb') as f:
            f.write(content)
        
        print(f"  下載成功: {len(content):,} bytes")
        return output_path
        
    except urllib.error.HTTPError as e:
        print(f"  HTTP 錯誤: {e.code} {e.reason}")
        return None
    except Exception as e:
        print(f"  下載失敗: {e}")
        return None


def process_sp_file(filepath: Path, output_dir: Path = None) -> int:
    """
    解析 sp*.dat 文件並輸出 CSV
    返回: 解析的記錄數
    """
    if output_dir is None:
        output_dir = filepath.parent
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  讀取: {filepath}")
    
    with open(filepath, 'rb') as f:
        content = f.read()
    
    records, eof_found = parse_sp_file(content)
    
    print(f"  解析: {len(records):,} 條記錄, EOF={'是' if eof_found else '否'}")
    
    if not records:
        print("  警告: 沒有解析到任何記錄")
        return 0
    
    # 生成 CSV
    date_str = filepath.stem[2:]  # sp20260508 -> 20260508
    csv_path = output_dir / f"SP_{date_str}.csv"
    
    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'line_no', 'stock_code', 'type', 'expiry', 'strike', 'settle', 'volume'
        ])
        writer.writeheader()
        writer.writerows(records)
    
    print(f"  保存: {csv_path}")
    
    # 統計
    from collections import Counter
    stock_counts = Counter(r['stock_code'] for r in records)
    print(f"  股票數: {len(stock_counts)}")
    print(f"  記錄數: {len(records):,}")
    
    return len(records)


# ============================================================
# 主程式
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="HKEX 收市結算價資料下載器")
    parser.add_argument('--date', type=str, help='日期 (YYYYMMDD)，預設今日')
    parser.add_argument('--output', type=str, help='輸出目錄', default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    
    # 確定日期 (URL 用 YYMMDD 格式，但輸入接受 YYYYMMDD)
    if args.date:
        date_str = args.date
        if len(date_str) == 8 and date_str.isdigit():
            # Convert YYYYMMDD -> YYMMDD
            date_str = date_str[2:]
    else:
        now = datetime.now(HK_TZ)
        date_str = now.strftime("%y%m%d")
    
    print("=" * 50)
    print("  HKEX SP 收市結算價下載器")
    print(f"  日期: {date_str}")
    print("=" * 50)
    
    output_dir = Path(args.output)
    
    # 下載
    filepath = download_sp_file(date_str, output_dir)
    if not filepath:
        print("下載失敗，退出")
        sys.exit(1)
    
    # 處理
    print()
    count = process_sp_file(filepath, output_dir)
    
    print()
    print("=" * 50)
    print(f"  完成! 共 {count:,} 條記錄")
    print("=" * 50)


if __name__ == "__main__":
    main()
