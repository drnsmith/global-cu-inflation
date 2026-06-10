#!/usr/bin/env python3
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import time

RAW = Path.home() / "projects" / "global-cu-inflation" / "data" / "raw"

CITIES = {
    "USA": {"lat": 39.95,  "lon": -75.17},
    "DEU": {"lat": 52.52,  "lon": 13.40},
    "FRA": {"lat": 48.85,  "lon": 2.35},
    "JPN": {"lat": 35.68,  "lon": 139.69},
    "GBR": {"lat": 51.51,  "lon": -0.13},
    "CAN": {"lat": 45.42,  "lon": -75.69},
    "NZL": {"lat": -41.29, "lon": 174.78},
}

BASE_TEMP = 18.0

def fetch_year(lat, lon, year, retries=3):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "daily": "temperature_2m_mean",
        "timezone": "UTC"
    }
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return pd.DataFrame({
                    "date": pd.to_datetime(data["daily"]["time"]),
                    "temp_mean": data["daily"]["temperature_2m_mean"]
                })
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            time.sleep(3)
    return None

all_records = []

for iso3, info in CITIES.items():
    print(f"\n{iso3}...")
    country_daily = []
    for year in range(2010, 2025):
        print(f"  {year}", end=" ", flush=True)
        df_y = fetch_year(info["lat"], info["lon"], year)
        if df_y is not None:
            df_y["country_code"] = iso3
            country_daily.append(df_y)
        time.sleep(0.5)
    print()

    if country_daily:
        df_c = pd.concat(country_daily, ignore_index=True)
        df_c["hdd"] = (BASE_TEMP - df_c["temp_mean"]).clip(lower=0)
        df_c["cdd"] = (df_c["temp_mean"] - BASE_TEMP).clip(lower=0)
        df_c["year"] = df_c["date"].dt.year
        df_c["quarter"] = df_c["date"].dt.quarter
        q = df_c.groupby(["country_code","year","quarter"]).agg(
            hdd_sum=("hdd","sum"),
            cdd_sum=("cdd","sum"),
            temp_mean_q=("temp_mean","mean")
        ).reset_index()
        all_records.append(q)
        print(f"  ✓ {len(q)} quarterly records")

        # Save incrementally
        pd.concat(all_records).to_csv(RAW / "weather_hdd_cdd_quarterly.csv", index=False)

print(f"\n✓ Done. {RAW / 'weather_hdd_cdd_quarterly.csv'}")
