#!/usr/bin/env python3
"""
Fixed CU download — uses FRED to pull OECD CU series.
Replaces broken OECD SDMX API block.
Run this standalone, then continue with full download script.
"""

import os
import pandas as pd
import requests
import time
from pathlib import Path

RAW_DIR = Path.home() / "projects" / "global-cu-inflation" / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# OECD CU from FRED (quarterly, manufacturing, % balance, SA)
# Source: OECD Main Economic Indicators via FRED
FRED_CU_SERIES = {
    "USA": "BSCURT02USQ160S",
    "DEU": "BSCURT02DEQ160S",
    "FRA": "BSCURT02FRQ160S",
    "JPN": "BSCURT02JPQ160S",
    "GBR": "BSCURT02GBQ160S",
    "CAN": "BSCURT02CAQ160S",
}

# Also pull US TCU (monthly, manufacturing utilisation rate %)
# This is the primary US CU series — different concept but complementary
FRED_US_TCU = "TCU"

COUNTRY_NAMES = {
    "USA": "United States", "DEU": "Germany", "FRA": "France",
    "JPN": "Japan", "GBR": "United Kingdom", "CAN": "Canada"
}

def fetch_fred(series_id):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": "2010-01-01",
        "observation_end": "2024-12-31",
    }
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code == 200:
        obs = resp.json().get("observations", [])
        return [(o["date"], float(o["value"])) for o in obs if o["value"] != "."]
    else:
        print(f"  FRED error {resp.status_code} for {series_id}")
        return []

print("Downloading OECD CU via FRED...\n")

cu_records = []

for iso3, series_id in FRED_CU_SERIES.items():
    print(f"  {COUNTRY_NAMES[iso3]} ({series_id})...")
    obs = fetch_fred(series_id)
    for date, val in obs:
        cu_records.append({
            "date": date,
            "country": COUNTRY_NAMES[iso3],
            "country_code": iso3,
            "cu_pct": val,
            "series": series_id,
            "frequency": "Q",
            "source": "OECD via FRED"
        })
    print(f"    ✓ {len(obs)} observations")
    time.sleep(0.5)

# US TCU monthly (separate — different concept)
print(f"\n  US TCU monthly (manufacturing utilisation rate)...")
obs_tcu = fetch_fred(FRED_US_TCU)
tcu_records = []
for date, val in obs_tcu:
    tcu_records.append({
        "date": date,
        "country": "United States",
        "country_code": "USA",
        "cu_rate_pct": val,
        "series": FRED_US_TCU,
        "frequency": "M",
        "source": "Federal Reserve via FRED"
    })
print(f"    ✓ {len(obs_tcu)} observations")

# Save
if cu_records:
    df_cu = pd.DataFrame(cu_records)
    out = RAW_DIR / "oecd_cu_quarterly.csv"
    df_cu.to_csv(out, index=False)
    print(f"\n✓ Saved: {out} ({len(df_cu)} records, {df_cu['country_code'].nunique()} countries)")
    print(df_cu.groupby("country_code")["cu_pct"].count())

if tcu_records:
    df_tcu = pd.DataFrame(tcu_records)
    out = RAW_DIR / "fred_us_cu_monthly.csv"
    df_tcu.to_csv(out, index=False)
    print(f"\n✓ Saved: {out} ({len(df_tcu)} records)")

print("\nNote: NZL has no OECD BTS CU series. Will use industrial production proxies only.")
