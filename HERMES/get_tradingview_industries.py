#!/usr/bin/env python3
"""
從 TradingView 爬取港股行業分類名單
用法：python get_tradingview_industries.py
輸出：data/indlist_tv.csv
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

SECTORS_URL = "https://www.tradingview.com/markets/stocks-hong-kong/sectorandindustry-sector/"
INDUSTRIES_URL = "https://www.tradingview.com/markets/stocks-hong-kong/sectorandindustry-industry/"

def fetch_page(url):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text

def parse_sectors(html):
    """解析 Sectors 頁面（sector → industries 數量）"""
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("table tbody tr")
    sectors = []
    for row in rows:
        cells = row.find_all("td")
        if len(cells) >= 7:
            sector_link = cells[0].find("a")
            if sector_link:
                sector_name = sector_link.get_text(strip=True)
                industries_count = cells[5].get_text(strip=True)
                stocks_count = cells[6].get_text(strip=True)
                sectors.append({
                    "sector": sector_name,
                    "industries": industries_count,
                    "stocks": stocks_count,
                })
    return sectors

def parse_industries(html):
    """解析 Industries 頁面（industry → sector, stocks count）"""
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("table tbody tr")
    industries = []
    for row in rows:
        cells = row.find_all("td")
        if len(cells) >= 7:
            industry_link = cells[0].find("a")
            if industry_link:
                industry_name = industry_link.get_text(strip=True)
                href = industry_link.get("href", "")
                # 從 href 提取 industry slug
                slug = href.rstrip("/").split("/")[-1]
                sector_name = cells[5].get_text(strip=True)
                stocks_count = cells[6].get_text(strip=True)
                industries.append({
                    "industry": industry_name,
                    "slug": slug,
                    "sector": sector_name,
                    "stocks": stocks_count,
                })
    return industries

def main():
    print("🌐 爬取 TradingView 港股行業分類...")

    print("  [1/2] 取得 Industries 列表...")
    html = fetch_page(INDUSTRIES_URL)
    industries = parse_industries(html)
    print(f"  ✅ 取得 {len(industries)} 個行業")

    df = pd.DataFrame(industries)
    df = df[["industry", "sector", "stocks", "slug"]]
    df = df.sort_values("industry").reset_index(drop=True)

    output_path = "Data/indlist_tv.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ 已儲存 → {output_path}")
    print(f"\n行數: {len(df)}")
    print(df.head(10).to_string())

    # 統計每個 sector 的行業數
    sector_stats = df.groupby("sector").agg(
        industries=("industry", "count"),
        total_stocks=("stocks", lambda x: pd.to_numeric(x, errors="coerce").sum())
    ).reset_index()
    print(f"\n=== Sector 統計 ===")
    print(sector_stats.sort_values("industries", ascending=False).to_string())

if __name__ == "__main__":
    main()
