import os, requests, time
import pandas as pd
from pathlib import Path

RAW_DIR = Path.home() / "projects" / "global-cu-inflation" / "data" / "raw"
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

CORRECTIONS = {
    "NZL": ("BSCURT02NZQ160S", "New Zealand"),
    "CAN": ("CAPUTILQ", "Canada"),
}

def fetch_fred(series_id):
    resp = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": series_id, "api_key": FRED_API_KEY,
                "file_type": "json", "observation_start": "2010-01-01",
                "observation_end": "2024-12-31"},
        timeout=15
    )
    if resp.status_code == 200:
        return [(o["date"], float(o["value"]))
                for o in resp.json().get("observations", [])
                if o["value"] != "."]
    print(f"  Error {resp.status_code} for {series_id}")
    return []

new_records = []
for iso3, (series_id, name) in CORRECTIONS.items():
    print(f"  {name} ({series_id})...")
    obs = fetch_fred(series_id)
    for date, val in obs:
        new_records.append({
            "date": date, "country": name, "country_code": iso3,
            "cu_pct": val, "series": series_id, "frequency": "Q",
            "source": "OECD via FRED" if iso3 == "NZL" else "Statistics Canada via FRED"
        })
    print(f"    ✓ {len(obs)} observations")
    time.sleep(0.5)

if new_records:
    existing = pd.read_csv(RAW_DIR / "oecd_cu_quarterly.csv")
    combined = pd.concat([existing, pd.DataFrame(new_records)], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["country_code", "date"]).reset_index(drop=True)
    combined.to_csv(RAW_DIR / "oecd_cu_quarterly.csv", index=False)
    print(combined.groupby("country_code")["cu_pct"].count())
