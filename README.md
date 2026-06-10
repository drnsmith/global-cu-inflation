# Global Capacity Utilisation and Inflation Forecasting

**Status: Active empirical implementation**

## The Problem

Inflation models require a measure of real-side pressure. Capacity utilisation (CU) is the natural candidate — but official CU surveys exist mainly in advanced economies. Many countries have no reliable series, intermittent coverage, or weak industrial statistics.

This project asks: can missing industrial capacity pressure be recovered from observable physical and activity proxies, and does recovering it improve what we know about inflation?

**Organising claim:** Inflation models may fail partly because real-side pressure is badly measured, not only because pressure no longer matters.

**One-sentence question:** Does fixing a measurement problem change what we know about inflation?

---

## Three-Paper Research Programme

| Paper | Question | Status |
|-------|----------|--------|
| **Paper 1: Measurement** | Can country-level industrial CU be estimated from electricity, industrial production, nightlights, and trade proxies with enough accuracy to recover official CU dynamics where observed data exist? | Active |
| **Paper 2: Forecasting** | Does estimated CU improve out-of-sample inflation forecasts relative to standard benchmark predictors? | Planned |
| **Paper 3: Global Transmission** | Does domestic inflation respond to trade-weighted foreign CU — not only domestic slack? | Planned |

---

## Repository Structure

```
global-cu-inflation/
├── data/
│   ├── raw/                    # Downloaded from APIs — do not edit
│   ├── processed/              # Cleaned, harmonised panel datasets
│   └── harmonised/             # Analysis-ready panel (country × quarter)
├── notebooks/
│   ├── 01_explore_raw.ipynb    # Initial data audit
│   ├── 02_construct_panel.ipynb
│   ├── 03_estimate_cu.ipynb    # Paper 1: ML estimation models
│   └── 04_forecast_eval.ipynb  # Paper 2: Inflation forecasting
├── src/
│   ├── download.py             # Data collection (this script)
│   ├── clean.py                # Harmonisation pipeline
│   ├── models.py               # CU estimation models
│   └── evaluate.py             # Validation and forecast evaluation
├── outputs/
│   ├── figures/
│   └── tables/
├── docs/
│   └── data_sources.md
├── requirements.txt
└── README.md
```

---

## Data Sources

| Block | Source | Variables | Coverage |
|-------|--------|-----------|----------|
| Observed CU | OECD MEI (new SDMX API) | Manufacturing CU (%) | Advanced economies, monthly |
| Observed CU | FRED (TCU) | Total CU, manufacturing | USA, monthly |
| Industrial production | FRED (MEI series) | IP index | 7 countries, monthly |
| Electricity | Ember / OWID | Generation, TWh | Global, annual/monthly |
| Nightlights | NASA VIIRS (processed) | Radiance panel | 10 countries, annual |
| Inflation | World Bank WDI | CPI annual % | 7 countries, annual |
| Macro controls | World Bank WDI | GDP pc, unemployment, trade, M2 | 7 countries, annual |
| Data quality | World Bank SPI | Statistical performance score | Global |

---

## Countries

**Training set (observed CU available):** USA, DEU, FRA, JPN, GBR, CAN, NZL

**Extension (Paper 1 prediction targets):** Countries without official CU surveys — identified using World Bank SPI classification.

---

## Empirical Design (Paper 1)

**Stage 1 — Estimate CU from proxies:**
Train models on countries with observed CU. Predictors: electricity generation, industrial production, nightlights, trade volumes, weather/calendar adjustments, structural controls.

**Validation strategy:**
- Random country holdout
- Leave-one-region-out
- Leave-one-income-group-out
- High-income → middle-income transfer test

**Models:**
- Panel OLS (econometric baseline)
- Random Forest
- Gradient Boosting (XGBoost)
- Ensemble

**Pre-specified success criteria:** ≥3% lower RMSE against benchmark, supported by Diebold-Mariano tests, robust across countries and horizons.

---

## Setup

```bash
# Clone and install
git clone https://github.com/[username]/global-cu-inflation.git
cd global-cu-inflation
pip install -r requirements.txt

# Set FRED API key (free: https://fred.stlouisfed.org/docs/api/api_key.html)
export FRED_API_KEY="your_key_here"

# Download all data
python src/download.py
```

---

## Related Projects

- [Institutional Legibility Index](https://github.com/[username]/institutional-legibility-index) — companion research programme on measurement validity in institutional data systems

---

## Author

Dr Natasha Smith | [Valimetrica](https://valimetrica.com) | Independent Economic Research

*Research focus: measurement validity in applied economics — whether the instruments used to measure economic phenomena are themselves valid.*
