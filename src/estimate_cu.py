#!/usr/bin/env python3
"""
Paper 1: Capacity Without a Survey
Estimate industrial CU from observable proxies.
Models: Panel OLS baseline + Gradient Boosting
Validation: Leave-one-country-out holdout
Output: outputs/figures/ + outputs/tables/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Modelling
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ───────────────────────────────────────────────────────
BASE = Path.home() / "projects" / "global-cu-inflation"
HARM = BASE / "data" / "harmonised"
FIG  = BASE / "outputs" / "figures"
TAB  = BASE / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# ── Load panel ──────────────────────────────────────────────────
panel = pd.read_csv(HARM / "cu_panel_quarterly.csv")
print(f"Panel loaded: {panel.shape}")

# ── Feature engineering ─────────────────────────────────────────
# Log transforms for scale-invariance
panel["log_ip"]    = np.log(panel["ip_index"].clip(lower=0.01))
panel["log_elec"]  = np.log(panel["electricity_twh"].clip(lower=0.01))
panel["log_gdp"]   = np.log(panel["gdp_per_capita_usd"].clip(lower=1))

# IP growth rate (quarter-on-quarter within country)
panel = panel.sort_values(["country_code","year","quarter"]).reset_index(drop=True)
panel["ip_growth"] = panel.groupby("country_code")["ip_index"].pct_change()

# Electricity growth
panel["elec_growth"] = panel.groupby("country_code")["electricity_twh"].pct_change()

# Time trend
panel["time_trend"] = (panel["year"] - 2010) * 4 + panel["quarter"]

# Country dummies (for OLS fixed effects)
panel = pd.get_dummies(panel, columns=["country_code"], prefix="c", drop_first=True)
country_dummies = [c for c in panel.columns if c.startswith("c_")]

# Restore country_code from dummies for splits
# (we kept it before get_dummies via sort, reconstruct from skeleton)
countries_order = ["CAN","DEU","FRA","GBR","JPN","NZL","USA"]
panel["country_code"] = np.repeat(countries_order, 60)

# ── Training set: rows with observed CU ─────────────────────────
FEATURES = ["log_ip","log_elec","log_gdp","ip_growth","elec_growth",
            "unemployment_pct","trade_pct_gdp","time_trend"] + country_dummies

df_train_full = panel[panel["cu_pct"].notna()].copy()
df_train_full = df_train_full.dropna(subset=FEATURES)

print(f"Training observations (observed CU): {len(df_train_full)}")
print(f"Countries in training set: {sorted(df_train_full['country_code'].unique())}")

# ── Leave-one-country-out validation ────────────────────────────
countries_with_full_cu = ["DEU","FRA","JPN","NZL","USA"]  # GBR partial, CAN partial — still include
countries_to_holdout   = ["DEU","FRA","JPN","NZL","USA","GBR"]

results = []

print("\n── LEAVE-ONE-COUNTRY-OUT VALIDATION ───────────────────")
print(f"{'Country':<8} {'Model':<20} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'N':>6}")
print("-" * 62)

ols_preds_all  = []
gbm_preds_all  = []

for holdout in countries_to_holdout:
    # Split
    mask_train = (df_train_full["country_code"] != holdout)
    mask_test  = (df_train_full["country_code"] == holdout)

    X_train = df_train_full.loc[mask_train, FEATURES].values
    y_train = df_train_full.loc[mask_train, "cu_pct"].values
    X_test  = df_train_full.loc[mask_test,  FEATURES].values
    y_test  = df_train_full.loc[mask_test,  "cu_pct"].values

    if len(y_test) == 0:
        continue

    # Scale
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # ── OLS ──
    ols = LinearRegression()
    ols.fit(X_tr_sc, y_train)
    y_pred_ols = ols.predict(X_te_sc)

    rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))
    mae_ols  = mean_absolute_error(y_test, y_pred_ols)
    r2_ols   = r2_score(y_test, y_pred_ols)

    print(f"{holdout:<8} {'Panel OLS':<20} {rmse_ols:>8.2f} {mae_ols:>8.2f} {r2_ols:>8.3f} {len(y_test):>6}")

    results.append({"country": holdout, "model": "OLS",
                    "rmse": rmse_ols, "mae": mae_ols, "r2": r2_ols, "n": len(y_test)})

    ols_preds_all.append(pd.DataFrame({
        "country": holdout,
        "y_true": y_test,
        "y_pred": y_pred_ols,
        "model": "OLS",
        "quarter_idx": df_train_full.loc[mask_test, "time_trend"].values
    }))

    # ── Gradient Boosting ──
    gbm = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=5, subsample=0.8, random_state=42
    )
    gbm.fit(X_train, y_train)  # GBM doesn't need scaling
    y_pred_gbm = gbm.predict(X_test)

    rmse_gbm = np.sqrt(mean_squared_error(y_test, y_pred_gbm))
    mae_gbm  = mean_absolute_error(y_test, y_pred_gbm)
    r2_gbm   = r2_score(y_test, y_pred_gbm)

    print(f"{holdout:<8} {'Gradient Boosting':<20} {rmse_gbm:>8.2f} {mae_gbm:>8.2f} {r2_gbm:>8.3f} {len(y_test):>6}")

    results.append({"country": holdout, "model": "GBM",
                    "rmse": rmse_gbm, "mae": mae_gbm, "r2": r2_gbm, "n": len(y_test)})

    gbm_preds_all.append(pd.DataFrame({
        "country": holdout,
        "y_true": y_test,
        "y_pred": y_pred_gbm,
        "model": "GBM",
        "quarter_idx": df_train_full.loc[mask_test, "time_trend"].values
    }))

# ── Summary table ────────────────────────────────────────────────
print("\n── SUMMARY (mean across holdout countries) ─────────────")
df_results = pd.DataFrame(results)
summary = df_results.groupby("model")[["rmse","mae","r2"]].mean().round(3)
print(summary)
df_results.to_csv(TAB / "holdout_validation_results.csv", index=False)
print(f"\n✓ Saved: {TAB / 'holdout_validation_results.csv'}")

# ── CHART 1: Predicted vs Actual by country ──────────────────────
print("\nGenerating charts...")

all_ols = pd.concat(ols_preds_all, ignore_index=True)
all_gbm = pd.concat(gbm_preds_all, ignore_index=True)

countries_plotted = sorted(all_ols["country"].unique())
n_countries = len(countries_plotted)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    "Paper 1: Estimated vs Observed Capacity Utilisation\nLeave-One-Country-Out Validation",
    fontsize=13, fontweight="bold", y=1.01
)
axes_flat = axes.flatten()

for i, country in enumerate(countries_plotted):
    ax = axes_flat[i]

    d_ols = all_ols[all_ols["country"] == country].sort_values("quarter_idx")
    d_gbm = all_gbm[all_gbm["country"] == country].sort_values("quarter_idx")

    ax.plot(d_ols["quarter_idx"], d_ols["y_true"],
            color="#2c3e50", linewidth=2, label="Observed CU", zorder=3)
    ax.plot(d_ols["quarter_idx"], d_ols["y_pred"],
            color="#e74c3c", linewidth=1.5, linestyle="--", label="OLS", zorder=2)
    ax.plot(d_gbm["quarter_idx"], d_gbm["y_pred"],
            color="#2980b9", linewidth=1.5, linestyle="-.", label="GBM", zorder=2)

    r2_o = df_results[(df_results["country"]==country)&(df_results["model"]=="OLS")]["r2"].values[0]
    r2_g = df_results[(df_results["country"]==country)&(df_results["model"]=="GBM")]["r2"].values[0]

    ax.set_title(f"{country}  |  OLS R²={r2_o:.2f}  GBM R²={r2_g:.2f}", fontsize=10)
    ax.set_xlabel("Quarter (2010=1)", fontsize=8)
    ax.set_ylabel("CU (%)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

# Hide unused subplot
if n_countries < 6:
    axes_flat[-1].set_visible(False)

plt.tight_layout()
chart1_path = FIG / "01_predicted_vs_actual_by_country.png"
plt.savefig(chart1_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Chart 1: {chart1_path}")

# ── CHART 2: Model comparison — RMSE and R² bar chart ────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    "Paper 1: Validation Performance — OLS vs Gradient Boosting\n"
    "Leave-One-Country-Out Holdout",
    fontsize=12, fontweight="bold"
)

x = np.arange(len(countries_plotted))
w = 0.35

ols_rmse = [df_results[(df_results["country"]==c)&(df_results["model"]=="OLS")]["rmse"].values[0]
            for c in countries_plotted]
gbm_rmse = [df_results[(df_results["country"]==c)&(df_results["model"]=="GBM")]["rmse"].values[0]
            for c in countries_plotted]
ols_r2   = [df_results[(df_results["country"]==c)&(df_results["model"]=="OLS")]["r2"].values[0]
            for c in countries_plotted]
gbm_r2   = [df_results[(df_results["country"]==c)&(df_results["model"]=="GBM")]["r2"].values[0]
            for c in countries_plotted]

# RMSE
ax1.bar(x - w/2, ols_rmse, w, label="Panel OLS", color="#e74c3c", alpha=0.85)
ax1.bar(x + w/2, gbm_rmse, w, label="Gradient Boosting", color="#2980b9", alpha=0.85)
ax1.set_xlabel("Holdout Country", fontsize=10)
ax1.set_ylabel("RMSE (percentage points)", fontsize=10)
ax1.set_title("Out-of-Sample RMSE\n(lower is better)", fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels(countries_plotted)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis="y")
ax1.axhline(y=np.mean(ols_rmse), color="#e74c3c", linestyle=":", alpha=0.6, linewidth=1)
ax1.axhline(y=np.mean(gbm_rmse), color="#2980b9", linestyle=":", alpha=0.6, linewidth=1)

# R²
ax2.bar(x - w/2, ols_r2, w, label="Panel OLS", color="#e74c3c", alpha=0.85)
ax2.bar(x + w/2, gbm_r2, w, label="Gradient Boosting", color="#2980b9", alpha=0.85)
ax2.set_xlabel("Holdout Country", fontsize=10)
ax2.set_ylabel("R²", fontsize=10)
ax2.set_title("Out-of-Sample R²\n(higher is better)", fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(countries_plotted)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")
ax2.axhline(y=0, color="black", linewidth=0.8)

plt.tight_layout()
chart2_path = FIG / "02_model_comparison_rmse_r2.png"
plt.savefig(chart2_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Chart 2: {chart2_path}")

# ── Final summary ─────────────────────────────────────────────────
print("\n── FINAL RESULTS ────────────────────────────────────────")
print(summary)
print(f"\nOutputs:")
print(f"  {chart1_path}")
print(f"  {chart2_path}")
print(f"  {TAB / 'holdout_validation_results.csv'}")
print("\nNext: add CAN CU prediction (no observed CU — full prediction target)")
