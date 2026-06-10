#!/usr/bin/env python3
"""
Build harmonised quarterly panel for CU estimation.
Aggregates all raw files to country x quarter.
Output: data/harmonised/cu_panel_quarterly.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path.home() / "projects" / "global-cu-inflation" / "data" / "raw"
HARM = Path.home() / "projects" / "global-cu-inflation" / "data" / "harmonised"
HARM.mkdir(parents=True, exist_ok=True)

COUNTRIES = ["USA", "DEU", "FRA", "JPN", "GBR", "CAN", "NZL"]

# ── 1. OECD CU (quarterly) ──────────────────────────────────────
cu = pd.read_csv(RAW / "oecd_cu_quarterly.csv")
cu["date"] = pd.to_datetime(cu["date"])
cu["year"] = cu["date"].dt.year
cu["quarter"] = cu["date"].dt.quarter
cu = cu[["country_code", "year", "quarter", "cu_pct"]].copy()
print(f"CU: {cu.shape}, countries: {sorted(cu.country_code.unique())}")

# ── 2. US TCU monthly → quarterly average ───────────────────────
tcu = pd.read_csv(RAW / "fred_us_cu_monthly.csv")
tcu["date"] = pd.to_datetime(tcu["date"])
tcu["year"] = tcu["date"].dt.year
tcu["quarter"] = tcu["date"].dt.quarter
tcu_q = tcu.groupby(["country_code", "year", "quarter"])["cu_rate_pct"].mean().reset_index()
tcu_q.rename(columns={"cu_rate_pct": "cu_rate_us_pct"}, inplace=True)
print(f"US TCU quarterly: {tcu_q.shape}")

# ── 3. Industrial production monthly → quarterly average ────────
ip = pd.read_csv(RAW / "fred_industrial_production.csv")
ip["date"] = pd.to_datetime(ip["date"])
ip["year"] = ip["date"].dt.year
ip["quarter"] = ip["date"].dt.quarter
ip_q = ip.groupby(["country_code", "year", "quarter"])["ip_index"].mean().reset_index()
print(f"IP quarterly: {ip_q.shape}, countries: {sorted(ip_q.country_code.unique())}")

# ── 4. Electricity annual → expand to quarters ──────────────────
elec = pd.read_csv(RAW / "electricity_generation_annual.csv")
# Expand annual to 4 quarters (same value per quarter — annual data)
elec_q = []
for _, row in elec.iterrows():
    for q in [1, 2, 3, 4]:
        elec_q.append({
            "country_code": row["country_code"],
            "year": int(row["year"]),
            "quarter": q,
            "electricity_generation_twh": row["electricity_generation"]
        })
elec_q = pd.DataFrame(elec_q)
print(f"Electricity quarterly: {elec_q.shape}")

# ── 5. World Bank macro annual → expand to quarters ─────────────
wb = pd.read_csv(RAW / "world_bank_macro.csv")
wb_q = []
for _, row in wb.iterrows():
    for q in [1, 2, 3, 4]:
        wb_q.append({
            "country_code": row["country_code"],
            "year": int(row["year"]),
            "quarter": q,
            "inflation_cpi_pct": row["inflation_cpi_pct"],
            "gdp_per_capita_usd": row["gdp_per_capita_usd"],
            "unemployment_pct": row["unemployment_pct"],
            "trade_pct_gdp": row["trade_pct_gdp"],
            "m2_pct_gdp": row["m2_pct_gdp"],
        })
wb_q = pd.DataFrame(wb_q)
print(f"World Bank quarterly: {wb_q.shape}")

# ── 6. Nightlights annual → expand to quarters ──────────────────
nl = pd.read_csv(RAW / "viirs_nightlights_panel.csv")
nl = nl.rename(columns={"iso3": "country_code"})
nl = nl[nl["country_code"].isin(COUNTRIES)]
nl_q = []
for _, row in nl.iterrows():
    for q in [1, 2, 3, 4]:
        nl_q.append({
            "country_code": row["country_code"],
            "year": int(row["year"]),
            "quarter": q,
            "nightlights_mean": row["luminosity_mean"],
            "nightlights_sum": row["luminosity_sum"],
        })
nl_q = pd.DataFrame(nl_q)
print(f"Nightlights quarterly: {nl_q.shape}")

# ── 7. Build skeleton: all country x year x quarter combos ──────
years = list(range(2010, 2025))
quarters = [1, 2, 3, 4]
skeleton = pd.MultiIndex.from_product(
    [COUNTRIES, years, quarters],
    names=["country_code", "year", "quarter"]
).to_frame(index=False)
print(f"\nSkeleton: {skeleton.shape}")

# ── 8. Merge all ────────────────────────────────────────────────
panel = skeleton.copy()
for df, label in [
    (cu, "CU"),
    (tcu_q, "US TCU"),
    (ip_q, "IP"),
    (elec_q, "Electricity"),
    (wb_q, "World Bank"),
    (nl_q, "Nightlights"),
]:
    panel = pd.merge(panel, df, on=["country_code", "year", "quarter"], how="left")
    print(f"After merging {label}: {panel.shape}")

# ── 9. Coverage summary ─────────────────────────────────────────
print("\n── COVERAGE SUMMARY ──────────────────────────────────")
key_vars = ["cu_pct", "ip_index", "electricity_generation_twh",
            "inflation_cpi_pct", "nightlights_mean"]
for var in key_vars:
    if var in panel.columns:
        n = panel[var].notna().sum()
        pct = 100 * n / len(panel)
        print(f"  {var}: {n}/{len(panel)} ({pct:.1f}%)")

print("\n── BY COUNTRY ────────────────────────────────────────")
print(panel.groupby("country_code")[["cu_pct", "ip_index", "inflation_cpi_pct"]].apply(
    lambda x: x.notna().sum()
))

# ── 10. Save ────────────────────────────────────────────────────
out = HARM / "cu_panel_quarterly.csv"
panel.to_csv(out, index=False)
print(f"\n✓ Saved: {out}")
print(f"  Shape: {panel.shape}")
print(f"  Columns: {list(panel.columns)}")
