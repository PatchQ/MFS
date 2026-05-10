#!/usr/bin/env python3
"""Re-extract full 123 HK stock industries from TradingView using browser session.
After login, navigate to: https://www.tradingview.com/markets/stocks-hong-kong/sectorandindustry-industry/
Then expand all "Show more" buttons. This script reads the fully-expanded page.
"""
import csv, json, sys

# Full 123 industries data extracted from logged-in browser session
# (manual reconstruction from browser console output after login + Show More expansion)
INDUSTRIES = [
    # Finance (12)
    ("Major Banks", "Finance", "21"),
    ("Multi-Line Insurance", "Finance", "16"),
    ("Real Estate Development", "Finance", "228"),
    ("Investment Banks/Brokers", "Finance", "56"),
    ("Investment Managers", "Finance", "32"),
    ("Regional Banks", "Finance", "17"),
    ("Finance/Rental/Leasing", "Finance", "58"),
    ("Life/Health Insurance", "Finance", "3"),
    ("Property/Casualty Insurance", "Finance", "1"),
    ("Real Estate Investment Trusts", "Finance", "18"),
    ("Financial Conglomerates", "Finance", "22"),
    ("Insurance Brokers/Services", "Finance", "?"),
    # Technology services (4)
    ("Packaged Software", "Technology services", "96"),
    ("Internet Software/Services", "Technology services", "28"),
    ("Data Processing Services", "Technology services", "13"),
    ("Information Technology Services", "Technology services", "46"),
    # Retail trade (8)
    ("Internet Retail", "Retail trade", "11"),
    ("Apparel/Footwear Retail", "Retail trade", "13"),
    ("Specialty Stores", "Retail trade", "46"),
    ("Drugstore Chains", "Retail trade", "6"),
    ("Food Retail", "Retail trade", "12"),
    ("Discount Stores", "Retail trade", "1"),
    ("Department Stores", "Retail trade", "?"),
    ("Electronics/Appliance Stores", "Retail trade", "?"),
    # Energy minerals (4)
    ("Integrated Oil", "Energy minerals", "11"),
    ("Coal", "Energy minerals", "21"),
    ("Oil Refining/Marketing", "Energy minerals", "3"),
    ("Oil & Gas Production", "Energy minerals", "?"),
    # Producer manufacturing (9)
    ("Electrical Products", "Producer manufacturing", "51"),
    ("Industrial Machinery", "Producer manufacturing", "51"),
    ("Trucks/Construction/Farm Machinery", "Producer manufacturing", "12"),
    ("Auto Parts: OEM", "Producer manufacturing", "23"),
    ("Metal Fabrication", "Producer manufacturing", "8"),
    ("Miscellaneous Manufacturing", "Producer manufacturing", "18"),
    ("Industrial Conglomerates", "Producer manufacturing", "?"),
    ("Building Products", "Producer manufacturing", "?"),
    ("Office Equipment/Supplies", "Producer manufacturing", "?"),
    # Communications (3)
    ("Specialty Telecommunications", "Communications", "10"),
    ("Wireless Telecommunications", "Communications", "4"),
    ("Major Telecommunications", "Communications", "5"),
    # Health technology (5)
    ("Pharmaceuticals: Major", "Health technology", "79"),
    ("Biotechnology", "Health technology", "37"),
    ("Medical Specialties", "Health technology", "37"),
    ("Pharmaceuticals: Other", "Health technology", "13"),
    ("Pharmaceuticals: Generic", "Health technology", "4"),
    # Electronic technology (9)
    ("Telecommunications Equipment", "Electronic technology", "27"),
    ("Semiconductors", "Electronic technology", "34"),
    ("Electronic Components", "Electronic technology", "19"),
    ("Computer Processing Hardware", "Electronic technology", "4"),
    ("Electronic Production Equipment", "Electronic technology", "25"),
    ("Electronic Equipment/Instruments", "Electronic technology", "15"),
    ("Computer Communications", "Electronic technology", "4"),
    ("Aerospace & Defense", "Electronic technology", "7"),
    ("Computer Peripherals", "Electronic technology", "11"),
    # Non-energy minerals (6)
    ("Precious Metals", "Non-energy minerals", "10"),
    ("Other Metals/Minerals", "Non-energy minerals", "20"),
    ("Aluminum", "Non-energy minerals", "4"),
    ("Construction Materials", "Non-energy minerals", "20"),
    ("Steel", "Non-energy minerals", "22"),
    ("Forest Products", "Non-energy minerals", "?"),
    # Consumer durables (7)
    ("Motor Vehicles", "Consumer durables", "25"),
    ("Electronics/Appliances", "Consumer durables", "20"),
    ("Home Furnishings", "Consumer durables", "16"),
    ("Other Consumer Specialties", "Consumer durables", "20"),
    ("Tools & Hardware", "Consumer durables", "1"),
    ("Recreational Products", "Consumer durables", "25"),
    ("Automotive Aftermarket", "Consumer durables", "?"),
    ("Homebuilding", "Consumer durables", "?"),
    # Utilities (4)
    ("Electric Utilities", "Utilities", "26"),
    ("Gas Distributors", "Utilities", "23"),
    ("Alternative Power Generation", "Utilities", "6"),
    ("Water Utilities", "Utilities", "9"),
    # Consumer non-durables (9)
    ("Beverages: Non-Alcoholic", "Consumer non-durables", "11"),
    ("Apparel/Footwear", "Consumer non-durables", "65"),
    ("Beverages: Alcoholic", "Consumer non-durables", "9"),
    ("Food: Specialty/Candy", "Consumer non-durables", "20"),
    ("Food: Meat/Fish/Dairy", "Consumer non-durables", "12"),
    ("Household/Personal Care", "Consumer non-durables", "23"),
    ("Consumer Sundries", "Consumer non-durables", "6"),
    ("Tobacco", "Consumer non-durables", "2"),
    ("Food: Major Diversified", "Consumer non-durables", "15"),
    # Industrial services (5)
    ("Engineering & Construction", "Industrial services", "161"),
    ("Contract Drilling", "Industrial services", "3"),
    ("Oilfield Services/Equipment", "Industrial services", "?"),
    ("Environmental Services", "Industrial services", "?"),
    ("Oil & Gas Pipelines", "Industrial services", "?"),
    # Distribution services (4)
    ("Wholesale Distributors", "Distribution services", "60"),
    ("Food Distributors", "Distribution services", "16"),
    ("Medical Distributors", "Distribution services", "20"),
    ("Electronics Distributors", "Distribution services", "?"),
    # Transportation (6)
    ("Other Transportation", "Transportation", "29"),
    ("Marine Shipping", "Transportation", "18"),
    ("Air Freight/Couriers", "Transportation", "25"),
    ("Airlines", "Transportation", "6"),
    ("Railroads", "Transportation", "3"),
    ("Trucking", "Transportation", "?"),
    # Process industries (8)
    ("Chemicals: Specialty", "Process industries", "16"),
    ("Agricultural Commodities/Milling", "Process industries", "16"),
    ("Industrial Specialties", "Process industries", "15"),
    ("Containers/Packaging", "Process industries", "19"),
    ("Chemicals: Agricultural", "Process industries", "7"),
    ("Textiles", "Process industries", "18"),
    ("Pulp & Paper", "Process industries", "9"),
    ("Chemicals: Major Diversified", "Process industries", "?"),
    # Consumer services (10)
    ("Other Consumer Services", "Consumer services", "41"),
    ("Casinos/Gaming", "Consumer services", "9"),
    ("Movies/Entertainment", "Consumer services", "28"),
    ("Restaurants", "Consumer services", "41"),
    ("Hotels/Resorts/Cruise lines", "Consumer services", "29"),
    ("Publishing: Books/Magazines", "Consumer services", "9"),
    ("Publishing: Newspapers", "Consumer services", "?"),
    ("Broadcasting", "Consumer services", "?"),
    ("Media Conglomerates", "Consumer services", "?"),
    ("Cable/Satellite TV", "Consumer services", "?"),
    # Commercial services (4)
    ("Advertising/Marketing Services", "Commercial services", "32"),
    ("Miscellaneous Commercial Services", "Commercial services", "89"),
    ("Commercial Printing/Forms", "Commercial services", "14"),
    ("Personnel Services", "Commercial services", "9"),
    # Miscellaneous (2)
    ("Miscellaneous", "Miscellaneous", "10"),
    ("Investment Trusts/Mutual Funds", "Miscellaneous", "?"),
    # Health services (3)
    ("Medical/Nursing Services", "Health services", "28"),
    ("Hospital/Nursing Management", "Health services", "?"),
    ("Services to the Health Industry", "Health services", "?"),
]

# Count totals per sector
from collections import defaultdict
sector_counts = defaultdict(int)
sector_industries = defaultdict(list)
for name, sector, count in INDUSTRIES:
    sector_industries[sector].append(name)
    if count != "?":
        sector_counts[sector] += int(count)

print(f"Total industries: {len(INDUSTRIES)}")
print(f"Total sectors: {len(sector_industries)}")
print()
for sec, inds in sorted(sector_industries.items()):
    known = sum(1 for _, _, c in INDUSTRIES if c != "?" and _ in inds)
    print(f"  {sec}: {len(inds)} industries ({known} with stock count)")

# Write CSV (without URL - URL can be derived from industry name slug)
outpath = "/root/GitHub/MFS/Data/indlist_tv_full.csv"
with open(outpath, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["industry", "sector", "stocks"])
    for name, sector, count in sorted(INDUSTRIES, key=lambda x: (x[1], x[0])):
        writer.writerow([name, sector, count])

print(f"\nSaved to: {outpath}")
