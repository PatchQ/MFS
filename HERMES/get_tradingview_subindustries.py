#!/usr/bin/env python3
"""
抓取 TradingView 香港股票全部 20 個部門的子行業（繁體中文）
用法：先在 browser 登入 TW TradingView，然後手動運行每個部門頁面
或用 Selenium automated。
"""
import csv, time, re

# 20 個部門的 slug（英文）
SECTOR_SLUGS = [
    "finance",
    "technology-services",
    "retail-trade",
    "energy-minerals",
    "producer-manufacturing",
    "communications",
    "health-technology",
    "electronic-technology",
    "non-energy-minerals",
    "consumer-durables",
    "utilities",
    "consumer-non-durables",
    "industrial-services",
    "distribution-services",
    "transportation",
    "process-industries",
    "consumer-services",
    "commercial-services",
    "miscellaneous",
    "health-services",
]

# TW 繁體名稱 mapping
SECTOR_NAMES_TW = {
    "finance": "金融",
    "technology-services": "科技服務",
    "retail-trade": "零售業",
    "energy-minerals": "能源礦產",
    "producer-manufacturing": "生產製造",
    "communications": "通訊",
    "health-technology": "健康科技",
    "electronic-technology": "電子科技",
    "non-energy-minerals": "非能源礦產",
    "consumer-durables": "耐用消費品",
    "utilities": "公用事業",
    "consumer-non-durables": "非耐用消費品",
    "industrial-services": "工業服務",
    "distribution-services": "配送服務",
    "transportation": "運輸",
    "process-industries": "加工業",
    "consumer-services": "消費者服務",
    "commercial-services": "商業服務",
    "miscellaneous": "其他類",
    "health-services": "健康服務",
}

BASE_URL = "https://tw.tradingview.com/markets/stocks-hong-kong/sectorandindustry-sector/{slug}/"

if __name__ == "__main__":
    print("部門總數:", len(SECTOR_SLUGS))
    for slug in SECTOR_SLUGS:
        url = BASE_URL.format(slug=slug)
        name = SECTOR_NAMES_TW.get(slug, slug)
        print(f"  {name}: {url}")
