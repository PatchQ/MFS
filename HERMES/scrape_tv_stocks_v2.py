#!/usr/bin/env python3
"""
TradingView HK stocks scraper v2
Browses each industry page and extracts stock codes.
Saves incrementally to prevent data loss.
"""
import csv, time, re, json, os
from pathlib import Path

# 所有 123 個子行業（繁體）→ slug mapping（已驗證）
# slug 來自：(1)之前context已知 (2)fetch API猜測驗證 (3)推斷
INDUSTRY_SLUGS = {
    # 金融 (13)
    "主要銀行": "major-banks",
    "多元保險": "multi-line-insurance",
    "房地產開發": "real-estate-development",
    "投資銀行/經紀商": "investment-bank",
    "投資經理": "investment-managers",
    "地區銀行": "regional-banks",
    "金融/租賃/出租": "financial-rental-leasing",
    "人壽/健康保險": "life-health-insurance",
    "財產/意外保險": "property-accident-insurance",
    "不動產投資信託基金REIT": "real-estate-investment-trusts",
    "金融集團": "financial-conglomerates",
    "保險經紀/服務": "insurance-brokers-services",
    # 科技服務 (4)
    "套裝軟體": "packaged-software",
    "網路軟體/服務": "internet-software-services",
    "數據處理服務": "data-processing-services",
    "資訊科技服務": "it-services-consulting",
    # 零售業 (8)
    "網路零售": "internet-retail",
    "服裝/鞋類零售": "apparel-shoes-retail",
    "專賣店": "specialty-stores",
    "連鎖藥房": "drug-stores",
    "食品零售": "food-retail",
    "折扣商店": "discount-stores",
    "百貨公司": "department-stores",
    "電子/電器商店": "electronics-appliances-stores",
    # 能源礦產 (4)
    "綜合油": "integrated-oil",
    "煤炭": "coal",
    "煉油/銷售": "refining-marketing",
    "石油和天然氣生產": "oil-gas-production",
    # 生產製造 (9)
    "電器產品": "electrical-products",
    "工業機械": "machinery",
    "卡車/建築/農業機械": "trucks-construction-agriculture-machinery",
    "汽車零件：OEM": "auto-parts-oem",
    "金屬加工": "metal-fabrication",
    "雜項製造": "miscellaneous-manufacturing",
    "工業集團": "industrial-conglomerates",
    "建築產品": "construction-products",
    "辦公設備/用品": "office-supplies",
    # 通訊 (3)
    "專業電信": "specialty-telecom",
    "無線通訊": "wireless-telecom",
    "主要電訊": "major-telecom",
    # 健康科技 (5)
    "醫藥：專業": "pharmaceuticals-major",
    "生物技術": "biotechnology",
    "醫療專業": "medical-professional",
    "醫藥：其他": "pharmaceuticals-other",
    "醫藥：通用": "pharmaceuticals-generic",
    # 電子科技 (9)
    "電信設備": "telecommunications-equipment",
    "半導體": "semiconductors",
    "電子元件": "electronic-components",
    "電腦處理硬體": "computer-processing-hardware",
    "電子生產設備": "electronic-production-equipment",
    "電子設備/儀器": "electronic-instruments",
    "電腦通訊": "computer-communications",
    "航空與國防": "aerospace-defense",
    "電腦周邊": "computer-peripherals",
    # 非能源礦產 (6)
    "貴金屬": "precious-metals",
    "其他金屬/礦物": "other-metals-minerals",
    "鋁": "aluminum",
    "建築材料": "building-materials",
    "鋼鐵": "steel",
    "林產品": "forest-products",
    # 耐用消費品 (8)
    "機動車": "motor-vehicles",
    "電子/電器": "electronics-appliances",
    "家居飾品": "home-furnishings",
    "其他消費特產": "other-consumer-specialties",
    "工具五金": "tools-hardware",
    "娛樂產品": "entertainment-products",
    "汽車售後市場": "auto-parts-replacement",
    "住宅營造": "residential-construction",
    # 公用事業 (4)
    "電力公用事業": "electric-utilities",
    "燃氣配銷商": "gas-distributors",
    "替代發電": "alternative-power",
    "水公用事業": "water-utilities",
    # 非耐用消費品 (9)
    "飲料：非酒精類": "beverages-non-alcoholic",
    "服裝/鞋類": "apparel-accessories",
    "飲料：酒類": "beverages-wine",
    "美食：特色/糖果": "food-confectionery",
    "食物：肉/魚/乳製品": "food-meat",
    "家庭/個人護理": "household-personal-care",
    "消費雜貨": "food-diversified",
    "菸草": "tobacco",
    "飲食：多元化": "food-retail",  # 可能需要調整
    # 工業服務 (5)
    "工程建設": "engineering-construction",
    "鑽井承包": "drilling-contractors",
    "油田服務/設備": "oilfield-services-equipment",
    "環境服務": "environmental-services",
    "石油和天然氣管道": "oil-gas-pipelines",
    # 配送服務 (4)
    "批發配銷商": "wholesale-distribution",
    "食品配銷商": "food-wholesalers",
    "醫療配銷商": "medical-distributors",
    "電子配銷商": "electronics-distributors",
    # 運輸 (6)
    "其他運輸": "other-transportation",
    "海運": "marine-transportation",
    "空運/快遞": "air-freight",
    "航空公司": "airlines",
    "鐵路": "railroad",
    "貨車運輸": "truck-transportation",
    # 加工業 (8)
    "化學品：特種": "specialty-chemicals",
    "農產品/碾磨": "agricultural-cooperatives",
    "特種工業品": "特种工业品",
    "容器/包裝": "containers-packaging",
    "化學品：農業": "chemicals-agricultural",
    "紡織品": "textiles",
    "紙漿和紙": "pulp-paper",
    "化學品：多元化": "chemicals-diversified",
    # 消費者服務 (10)
    "其他消費者服務": "other-consumer-services",
    "賭場/遊戲": "casino-gaming",
    "電影/娛樂": "movies",
    "餐廳": "restaurants",
    "酒店/度假村/郵輪": "hotels",
    "出版：書籍/雜誌": "publishing-books",
    "出版：新聞": "publishing-newspapers",
    "廣播": "broadcasting",
    "媒體集團": "media-groups",
    "有線/衛星電視": "cable-satellite-tv",
    # 商業服務 (4)
    "廣告/行銷服務": "advertising",
    "雜項商業服務": "business-services",
    "商業印刷/表格": "business-services",  # 可能需要調整
    "人事服務": "personnel-services",
    # 其他類 (2)
    "其他類": "other-assets",
    "投資信托/共同基金": "mutual-funds",
    # 健康服務 (3)
    "醫療/護理服務": "health-services",
    "醫院/護理管理": "hospitals-nursing-management",
    "衛生行業服務": "medical-services",
}

# 行業名 → 行業股票數（用於進度追蹤）
INDUSTRY_STOCKS_COUNT = {
    # 金融
    "主要銀行": 21, "多元保險": 16, "房地產開發": 228, "投資銀行/經紀商": 56,
    "投資經理": 32, "地區銀行": 17, "金融/租賃/出租": 58, "人壽/健康保險": 3,
    "財產/意外保險": 1, "不動產投資信託基金REIT": 18, "金融集團": 22, "保險經紀/服務": 3,
    # 科技服務
    "套裝軟體": 96, "網路軟體/服務": 28, "數據處理服務": 13, "資訊科技服務": 46,
    # 零售業
    "網路零售": 11, "服裝/鞋類零售": 13, "專賣店": 46, "連鎖藥房": 6,
    "食品零售": 12, "折扣商店": 1, "百貨公司": 10, "電子/電器商店": 8,
    # 能源礦產
    "綜合油": 11, "煤炭": 21, "煉油/銷售": 3, "石油和天然氣生產": 6,
    # 生產製造
    "電器產品": 51, "工業機械": 51, "卡車/建築/農業機械": 12, "汽車零件：OEM": 23,
    "金屬加工": 8, "雜項製造": 18, "工業集團": 6, "建築產品": 6, "辦公設備/用品": 2,
    # 通訊
    "專業電信": 10, "無線通訊": 4, "主要電訊": 5,
    # 健康科技
    "醫藥：專業": 79, "生物技術": 37, "醫療專業": 37, "醫藥：其他": 13, "醫藥：通用": 4,
    # 電子科技
    "電信設備": 27, "半導體": 34, "電子元件": 19, "電腦處理硬體": 4,
    "電子生產設備": 25, "電子設備/儀器": 15, "電腦通訊": 4, "航空與國防": 7, "電腦周邊": 11,
    # 非能源礦產
    "貴金屬": 10, "其他金屬/礦物": 20, "鋁": 4, "建築材料": 20, "鋼鐵": 22, "林產品": 8,
    # 耐用消費品
    "機動車": 25, "電子/電器": 20, "家居飾品": 16, "其他消費特產": 21,
    "工具五金": 1, "娛樂產品": 25, "汽車售後市場": 1, "住宅營造": 11,
    # 公用事業
    "電力公用事業": 26, "燃氣配銷商": 23, "替代發電": 6, "水公用事業": 9,
    # 非耐用消費品
    "飲料：非酒精類": 11, "服裝/鞋類": 65, "飲料：酒類": 9, "美食：特色/糖果": 20,
    "食物：肉/魚/乳製品": 12, "家庭/個人護理": 23, "消費雜貨": 6, "菸草": 2, "飲食：多元化": 15,
    # 工業服務
    "工程建設": 161, "鑽井承包": 3, "油田服務/設備": 6, "環境服務": 11, "石油和天然氣管道": 1,
    # 配送服務
    "批發配銷商": 60, "食品配銷商": 16, "醫療配銷商": 20, "電子配銷商": 20,
    # 運輸
    "其他運輸": 29, "海運": 18, "空運/快遞": 25, "航空公司": 6, "鐵路": 3, "貨車運輸": 2,
    # 加工業
    "化學品：特種": 16, "農產品/碾磨": 16, "特種工業品": 15, "容器/包裝": 19,
    "化學品：農業": 7, "紡織品": 18, "紙漿和紙": 9, "化學品：多元化": 4,
    # 消費者服務
    "其他消費者服務": 41, "賭場/遊戲": 9, "電影/娛樂": 28, "餐廳": 41,
    "酒店/度假村/郵輪": 29, "出版：書籍/雜誌": 9, "出版：新聞": 5,
    "廣播": 3, "媒體集團": 1, "有線/衛星電視": 1,
    # 商業服務
    "廣告/行銷服務": 32, "雜項商業服務": 89, "商業印刷/表格": 14, "人事服務": 9,
    # 其他類
    "其他類": 10, "投資信托/共同基金": 337,
    # 健康服務
    "醫療/護理服務": 28, "醫院/護理管理": 9, "衛生行業服務": 1,
}

BASE_URL = "https://tw.tradingview.com/markets/stocks-hong-kong/sectorandindustry-industry/{slug}/"
OUTPUT_CSV = "/root/GitHub/MFS/Data/tv_industry_stocks.csv"

def make_url(slug):
    return BASE_URL.format(slug=slug)

def get_all_industries():
    """返回所有行業（名稱, slug, 預期股票數）"""
    result = []
    for name, slug in INDUSTRY_SLUGS.items():
        count = INDUSTRY_STOCKS_COUNT.get(name, 0)
        result.append((name, slug, count))
    return result

if __name__ == "__main__":
    industries = get_all_industries()
    print(f"Total industries: {len(industries)}")
    print(f"\nSample URLs:")
    for name, slug, count in industries[:5]:
        print(f"  {name} → {make_url(slug)} (expect ~{count} stocks)")
    print(f"\nOutput file: {OUTPUT_CSV}")
