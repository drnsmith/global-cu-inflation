#!/usr/bin/env python3
"""
Global CU Inflation — Data Collection Script v2
Downloads OECD CU, FRED industrial production, Ember electricity,
World Bank inflation and macro controls.
Countries: USA, DEU, FRA, JPN, GBR, CAN, NZL
Output: ~/projects/global-cu-inflation/data/raw/
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import json
import logging
from pathlib import Path
import time

# ==================== CONFIG ====================
BASE_DIR = Path.home() / "projects" / "global-cu-inflation"
RAW_DIR = BASE_DIR / "data" / "raw"
LOG_FILE = BASE_DIR / "download_log.txt"

COUNTRIES = ["USA", "DEU", "FRA", "JPN", "GBR", "CAN", "NZL"]
COUNTRY_NAMES = {
    "USA": "United States", "DEU": "Germany", "FRA": "France",
    "JPN": "Japan", "GBR": "United Kingdom", "CAN": "Canada",
    "NZL": "New Zealand"
}

# World Bank country codes
WB_CODES = {
    "USA": "US", "DEU": "DE", "FRA": "FR",
    "JPN": "JP", "GBR": "GB", "CAN": "CA", "NZL": "NZ"
}

# OECD country codes (used in new API)
OECD_CODES = {
    "USA": "USA", "DEU": "DEU", "FRA": "FRA",
    "JPN": "JPN", "GBR": "GBR", "CAN": "CAN", "NZL": "NZL"
}

# FRED series IDs for industrial production (monthly index)
FRED_IP_SERIES = {
    "USA": "INDPRO",        # US Industrial Production Index
    "DEU": "DEUPROINDMISMEI",  # Germany industrial production
    "FRA": "FRAPROINDMISMEI",  # France
    "JPN": "JPNPROINDMISMEI",  # Japan
    "GBR": "GBRPROINDMISMEI",  # UK
    "CAN": "CANPROINDMISMEI",  # Canada
    "NZL": "NZLPROINDMISMEI",  # New Zealand
}

# FRED series for CU (manufacturing)
FRED_CU_SERIES = {
    "USA": "TCU",  # US Total Capacity Utilization (manufacturing)
}

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
WORLD_BANK_API = "https://api.worldbank.org/v2"
START_YEAR = 2010
END_YEAR = 2024

# ==================== SETUP ====================
def setup():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup()

# ==================== BLOCK 1A: OECD CU (NEW API) ====================
def download_oecd_cu():
    """
    Download OECD Manufacturing Capacity Utilisation.
    Uses the NEW OECD SDMX REST API (sdmx.oecd.org).
    Old API (stats.oecd.org) was deprecated — this replaces it.
    """
    logger.info("=" * 60)
    logger.info("BLOCK 1A: OECD Capacity Utilisation (new API)")
    logger.info("=" * 60)

    cu_data = []

    for iso3, oecd_code in OECD_CODES.items():
        try:
            # New OECD SDMX 2.1 REST API
            # Dataset: MEI = Main Economic Indicators
            # Series: PRINTO01 = Capacity utilisation, manufacturing
            url = (
                f"https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_STES,"
                f"4.0/{oecd_code}.PRINTO01.........?startPeriod={START_YEAR}"
                f"&endPeriod={END_YEAR}&dimensionAtObservation=TIME_PERIOD&format=jsondata"
            )

            logger.info(f"Fetching CU: {COUNTRY_NAMES[iso3]}...")
            resp = requests.get(url, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                try:
                    obs = data["data"]["dataSets"][0]["series"]
                    time_periods = data["data"]["structure"]["dimensions"]["observation"][0]["values"]

                    for series_key, series_val in obs.items():
                        for t_idx, val_list in series_val["observations"].items():
                            period = time_periods[int(t_idx)]["id"]
                            value = val_list[0]
                            if value is not None:
                                cu_data.append({
                                    "date": period,
                                    "country": COUNTRY_NAMES[iso3],
                                    "country_code": iso3,
                                    "cu_pct": float(value)
                                })
                except (KeyError, IndexError) as e:
                    logger.warning(f"Parse error for {iso3}: {e}")
            elif resp.status_code == 404:
                logger.warning(f"No OECD CU data for {iso3} (404 — country may not publish this series)")
            else:
                logger.warning(f"OECD API returned {resp.status_code} for {iso3}")

            time.sleep(1)  # polite crawling

        except Exception as e:
            logger.warning(f"Error fetching OECD CU for {iso3}: {e}")

    if cu_data:
        df = pd.DataFrame(cu_data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["country_code", "date"]).reset_index(drop=True)
        out = RAW_DIR / "oecd_cu_monthly.csv"
        df.to_csv(out, index=False)
        logger.info(f"✓ Saved: {out} ({len(df)} records, {df['country_code'].nunique()} countries)")
        return df
    else:
        logger.error("No OECD CU data retrieved.")
        return None


# ==================== BLOCK 1B: FRED — US CU + INDUSTRIAL PRODUCTION ====================
def download_fred_series():
    """
    Download from FRED:
    - US Total Capacity Utilisation (TCU) — monthly
    - Industrial Production indices for all 7 countries — monthly
    FRED is free, reliable, no rate limit issues.
    """
    logger.info("=" * 60)
    logger.info("BLOCK 1B: FRED — Industrial Production + US CU")
    logger.info("=" * 60)

    if not FRED_API_KEY:
        logger.error("FRED_API_KEY not set. Export it before running.")
        return None, None

    def fetch_fred(series_id, label):
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": f"{START_YEAR}-01-01",
            "observation_end": f"{END_YEAR}-12-31",
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                obs = resp.json().get("observations", [])
                records = []
                for o in obs:
                    if o["value"] != ".":
                        records.append({
                            "date": o["date"],
                            "value": float(o["value"])
                        })
                logger.info(f"  ✓ {label}: {len(records)} observations")
                return pd.DataFrame(records)
            else:
                logger.warning(f"  FRED {series_id}: HTTP {resp.status_code}")
                return None
        except Exception as e:
            logger.warning(f"  FRED {series_id} error: {e}")
            return None

    # US Capacity Utilisation
    logger.info("Fetching US Capacity Utilisation (TCU)...")
    df_us_cu = fetch_fred("TCU", "US CU")
    if df_us_cu is not None:
        df_us_cu["country"] = "United States"
        df_us_cu["country_code"] = "USA"
        df_us_cu.rename(columns={"value": "cu_pct"}, inplace=True)
        out = RAW_DIR / "fred_us_cu_monthly.csv"
        df_us_cu.to_csv(out, index=False)
        logger.info(f"✓ Saved: {out}")

    # Industrial Production for all countries
    logger.info("Fetching Industrial Production indices...")
    ip_frames = []
    for iso3, series_id in FRED_IP_SERIES.items():
        logger.info(f"  {COUNTRY_NAMES[iso3]} ({series_id})...")
        df_ip = fetch_fred(series_id, COUNTRY_NAMES[iso3])
        if df_ip is not None:
            df_ip["country"] = COUNTRY_NAMES[iso3]
            df_ip["country_code"] = iso3
            df_ip.rename(columns={"value": "ip_index"}, inplace=True)
            ip_frames.append(df_ip)
        time.sleep(0.5)

    if ip_frames:
        df_ip_all = pd.concat(ip_frames, ignore_index=True)
        df_ip_all["date"] = pd.to_datetime(df_ip_all["date"])
        df_ip_all = df_ip_all.sort_values(["country_code", "date"]).reset_index(drop=True)
        out = RAW_DIR / "fred_industrial_production.csv"
        df_ip_all.to_csv(out, index=False)
        logger.info(f"✓ Saved: {out} ({len(df_ip_all)} records)")
        return df_us_cu, df_ip_all

    return df_us_cu, None


# ==================== BLOCK 1C: EMBER ELECTRICITY ====================
def download_ember_electricity():
    """
    Download Ember global electricity data.
    Ember changed their data endpoint — this uses the correct current URL.
    Falls back to direct CSV download from GitHub mirror if API fails.
    """
    logger.info("=" * 60)
    logger.info("BLOCK 1C: Ember Electricity Data")
    logger.info("=" * 60)

    country_map = {v: k for k, v in COUNTRY_NAMES.items()}

    # Try Ember's current bulk download
    urls_to_try = [
        "https://ember-climate.org/app/uploads/2022/07/monthly_full_release_long_format.csv",
        "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv",  # OWID fallback
    ]

    for url in urls_to_try:
        try:
            logger.info(f"Trying: {url}")
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                df_raw = pd.read_csv(pd.io.common.StringIO(resp.text))
                logger.info(f"  Columns: {list(df_raw.columns[:10])}")

                # Handle OWID format
                if "country" in df_raw.columns and "electricity_generation" in df_raw.columns:
                    df_filtered = df_raw[df_raw["country"].isin(COUNTRY_NAMES.values())].copy()
                    df_filtered["country_code"] = df_filtered["country"].map(country_map)
                    df_filtered = df_filtered[["year", "country", "country_code",
                                               "electricity_generation"]].dropna()
                    df_filtered = df_filtered[
                        (df_filtered["year"] >= START_YEAR) &
                        (df_filtered["year"] <= END_YEAR)
                    ]
                    if len(df_filtered) > 0:
                        out = RAW_DIR / "electricity_generation_annual.csv"
                        df_filtered.to_csv(out, index=False)
                        logger.info(f"✓ Saved: {out} ({len(df_filtered)} records) [OWID format]")
                        return df_filtered

                # Handle Ember format
                elif "Country" in df_raw.columns:
                    df_filtered = df_raw[
                        df_raw["Country"].isin(COUNTRY_NAMES.values())
                    ].copy()
                    if len(df_filtered) > 0:
                        df_filtered["country_code"] = df_filtered["Country"].map(country_map)
                        out = RAW_DIR / "ember_electricity_monthly.csv"
                        df_filtered.to_csv(out, index=False)
                        logger.info(f"✓ Saved: {out} ({len(df_filtered)} records)")
                        return df_filtered

        except Exception as e:
            logger.warning(f"  Failed: {e}")
            continue

    logger.warning("Ember/OWID electricity download failed. Add manual download instructions.")
    logger.warning("Manual: https://ember-climate.org/data/data-tools/data-explorer/")
    return None


# ==================== BLOCK 1D: WORLD BANK — INFLATION + CONTROLS ====================
def download_world_bank():
    """
    Download from World Bank WDI:
    - CPI Inflation (annual %)
    - GDP per capita (USD)
    - Unemployment rate
    - Trade openness (% GDP)
    - M2 money supply (% GDP)
    """
    logger.info("=" * 60)
    logger.info("BLOCK 1D: World Bank — Inflation + Macro Controls")
    logger.info("=" * 60)

    indicators = {
        "FP.CPI.TOTL.ZG": "inflation_cpi_pct",
        "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
        "SL.UEM.TOTL.ZS": "unemployment_pct",
        "NE.TRD.GNFS.ZS": "trade_pct_gdp",
        "FM.LBL.BMNY.GD.ZS": "m2_pct_gdp",
    }

    all_frames = []

    for wb_indicator, var_name in indicators.items():
        logger.info(f"Fetching {var_name}...")
        country_string = ";".join(WB_CODES.values())

        url = f"{WORLD_BANK_API}/country/{country_string}/indicators/{wb_indicator}"
        params = {
            "format": "json",
            "per_page": 1000,
            "date": f"{START_YEAR}:{END_YEAR}"
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if len(data) > 1 and data[1]:
                    records = []
                    wb_to_iso3 = {v: k for k, v in WB_CODES.items()}
                    for record in data[1]:
                        if record["value"] is not None:
                            wb_code = record["countryiso3code"]
                            # Map WB iso3 to our codes
                            iso3 = wb_code if wb_code in COUNTRIES else None
                            if iso3:
                                records.append({
                                    "year": int(record["date"]),
                                    "country": COUNTRY_NAMES.get(iso3, record["country"]["value"]),
                                    "country_code": iso3,
                                    var_name: float(record["value"])
                                })
                    if records:
                        df_ind = pd.DataFrame(records)
                        all_frames.append(df_ind)
                        logger.info(f"  ✓ {var_name}: {len(records)} records")
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"  World Bank {wb_indicator} error: {e}")

    if all_frames:
        # Merge all indicators on country_code + year
        df_merged = all_frames[0]
        for df_next in all_frames[1:]:
            df_merged = pd.merge(
                df_merged, df_next,
                on=["year", "country", "country_code"],
                how="outer"
            )
        df_merged = df_merged.sort_values(["country_code", "year"]).reset_index(drop=True)
        out = RAW_DIR / "world_bank_macro.csv"
        df_merged.to_csv(out, index=False)
        logger.info(f"✓ Saved: {out} ({len(df_merged)} records)")
        return df_merged
    else:
        logger.error("No World Bank data retrieved.")
        return None


# ==================== BLOCK 1E: NIGHTLIGHTS (FROM LACIE) ====================
def link_nightlights():
    """
    Link pre-processed VIIRS nightlights panel from LaCie
    (already collected in cross-national-dev-analytics project).
    Copies relevant countries to this project's data directory.
    """
    logger.info("=" * 60)
    logger.info("BLOCK 1E: VIIRS Nightlights (from LaCie)")
    logger.info("=" * 60)

    source = Path("/Volumes/LaCie/cross-national-dev-analytics/data/processed/satellite/nighttime_lights_panel.csv")

    if not source.exists():
        logger.warning(f"Nightlights panel not found at {source}")
        logger.warning("Connect LaCie drive and re-run this block.")
        return None

    try:
        df = pd.read_csv(source)
        logger.info(f"  Loaded nightlights panel: {df.shape}")
        logger.info(f"  Columns: {list(df.columns)}")

        out = RAW_DIR / "viirs_nightlights_panel.csv"
        df.to_csv(out, index=False)
        logger.info(f"✓ Copied: {out}")
        return df
    except Exception as e:
        logger.warning(f"Error loading nightlights: {e}")
        return None


# ==================== BLOCK 1F: DATA QUALITY CLASSIFICATION ====================
def download_data_quality():
    """
    Download World Bank Statistical Performance Indicators (SPI).
    Used to classify countries by statistical system quality —
    critical for the transfer validity tests in Paper 1.
    """
    logger.info("=" * 60)
    logger.info("BLOCK 1F: World Bank Statistical Performance Indicators")
    logger.info("=" * 60)

    # SPI overall score
    url = f"{WORLD_BANK_API}/country/all/indicators/IQ.SPI.OVRL"
    params = {
        "format": "json",
        "per_page": 5000,
        "date": f"{START_YEAR}:{END_YEAR}"
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            if len(data) > 1 and data[1]:
                records = []
                for record in data[1]:
                    if record["value"] is not None:
                        records.append({
                            "year": int(record["date"]),
                            "country_code": record["countryiso3code"],
                            "country": record["country"]["value"],
                            "spi_score": float(record["value"])
                        })
                if records:
                    df = pd.DataFrame(records)
                    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)
                    out = RAW_DIR / "wb_spi_scores.csv"
                    df.to_csv(out, index=False)
                    logger.info(f"✓ Saved: {out} ({len(df)} records, {df['country_code'].nunique()} countries)")
                    return df
    except Exception as e:
        logger.warning(f"SPI download error: {e}")

    return None


# ==================== MAIN ====================
def main():
    print("\n" + "=" * 60)
    print("GLOBAL CU INFLATION — DATA COLLECTION v2")
    print(f"Countries: {', '.join(COUNTRIES)}")
    print(f"Period: {START_YEAR}–{END_YEAR}")
    print("=" * 60 + "\n")

    logger.info(f"Started: {datetime.now()}")
    logger.info(f"Output: {RAW_DIR}")

    if not FRED_API_KEY:
        print("\n⚠️  WARNING: FRED_API_KEY not set.")
        print("   Run: export FRED_API_KEY='your_key'")
        print("   Then re-run this script.\n")

    results = {}

    print("\n[1/6] OECD Capacity Utilisation (new API)...")
    results["oecd_cu"] = download_oecd_cu()

    print("\n[2/6] FRED Industrial Production + US CU...")
    results["fred_cu"], results["fred_ip"] = download_fred_series()

    print("\n[3/6] Ember Electricity...")
    results["electricity"] = download_ember_electricity()

    print("\n[4/6] World Bank Macro Controls...")
    results["world_bank"] = download_world_bank()

    print("\n[5/6] VIIRS Nightlights (LaCie)...")
    results["nightlights"] = link_nightlights()

    print("\n[6/6] Statistical Performance Indicators...")
    results["spi"] = download_data_quality()

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for name, df in results.items():
        if df is not None:
            print(f"  ✓ {name}: {len(df)} records")
        else:
            print(f"  ✗ {name}: FAILED — check log")

    print(f"\nLog: {LOG_FILE}")
    print(f"Raw data: {RAW_DIR}")
    print("\nNext: run notebooks/01_explore_raw.ipynb")
    logger.info(f"Completed: {datetime.now()}")


if __name__ == "__main__":
    main()
