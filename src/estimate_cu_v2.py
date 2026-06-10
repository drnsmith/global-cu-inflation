#!/usr/bin/env python3
"""
Paper 1: Capacity Without a Survey — v2
Within-country z-score normalisation before training.
CU scales are not comparable across countries (different survey methodologies).
Normalisation lets the model learn cycle shape, not level.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

BASE = Path.home() / "projects" / "global-cu-inflation"
HARM = BASE / "data" / "harmonised"
FIG  = BASE / "outputs" / "figures"
TAB  = BASE / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────
panel = pd.read_csv(HARM / "cu_panel_quarterly.csv")
panel = panel.sort_values(["country_code","year","quarter"]).reset_index(drop=True)

# ── Within-country z-score normalisation of CU ──────────────────
# CU surveys use incompatible scales across countries:
#   JPN: net balance (-25 to +6)
#   NZL: percentage balance (87–97)
#   USA: utilisation rate (66–79)
# Z-scoring within country removes level differences,
# preserving cycle shape for cross-country model training.
cu_stats = panel.groupby("country_code")["cu_pct"].agg(["mean","std"]).rename(
    columns={"mean":"cu_mean","std":"cu_std"})
panel = panel.join(cu_stats, on="country_code")
panel["cu_z"] = (panel["cu_pct"] - panel["cu_mean"]) / panel["cu_std"]

print("CU z-score stats by country (should all be ~mean=0, std=1):")
print(panel.groupby("country_code")["cu_z"].agg(["mean","std"]).round(3))

# ── Feature engineering ──────────────────────────────────────────
panel["log_ip"]      = np.log(panel["ip_index"].clip(lower=0.01))
panel["log_elec"]    = np.log(panel["electricity_twh"].clip(lower=0.01))
panel["log_gdp"]     = np.log(panel["gdp_per_capita_usd"].clip(lower=1))
panel["ip_growth"]   = panel.groupby("country_code")["ip_index"].pct_change()
panel["elec_growth"] = panel.groupby("country_code")["electricity_twh"].pct_change()
panel["time_trend"]  = (panel["year"] - 2010) * 4 + panel["quarter"]

# Country dummies for OLS fixed effects
panel_enc = pd.get_dummies(panel, columns=["country_code"], prefix="c", drop_first=True)
country_dummies = [c for c in panel_enc.columns if c.startswith("c_")]

# Restore country_code
countries_order = ["CAN","DEU","FRA","GBR","JPN","NZL","USA"]
panel_enc["country_code"] = np.repeat(countries_order, 60)

FEATURES = ["log_ip","log_elec","log_gdp","ip_growth","elec_growth",
            "unemployment_pct","trade_pct_gdp","time_trend"] + country_dummies

# Training set: observed CU only
df = panel_enc[panel_enc["cu_z"].notna()].dropna(subset=FEATURES).copy()
print(f"\nTraining observations: {len(df)}")
print(f"Countries: {sorted(df['country_code'].unique())}")

# ── Leave-one-country-out validation ─────────────────────────────
countries_to_holdout = ["DEU","FRA","JPN","NZL","USA","GBR"]
results = []
ols_preds_all = []
gbm_preds_all = []

print("\n── LEAVE-ONE-COUNTRY-OUT (normalised CU) ───────────────")
print(f"{'Country':<8} {'Model':<20} {'RMSE_z':>8} {'MAE_z':>8} {'R²':>8} {'N':>6}")
print("-" * 62)

for holdout in countries_to_holdout:
    mask_train = df["country_code"] != holdout
    mask_test  = df["country_code"] == holdout

    X_train = df.loc[mask_train, FEATURES].values
    y_train = df.loc[mask_train, "cu_z"].values
    X_test  = df.loc[mask_test,  FEATURES].values
    y_test  = df.loc[mask_test,  "cu_z"].values

    if len(y_test) == 0:
        continue

    # Scale features
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    # OLS
    ols = LinearRegression()
    ols.fit(X_tr_sc, y_train)
    y_pred_ols = ols.predict(X_te_sc)

    rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))
    mae_ols  = mean_absolute_error(y_test, y_pred_ols)
    r2_ols   = r2_score(y_test, y_pred_ols)
    print(f"{holdout:<8} {'Panel OLS':<20} {rmse_ols:>8.3f} {mae_ols:>8.3f} {r2_ols:>8.3f} {len(y_test):>6}")
    results.append({"country":holdout,"model":"OLS",
                    "rmse":rmse_ols,"mae":mae_ols,"r2":r2_ols,"n":len(y_test)})

    ols_preds_all.append(pd.DataFrame({
        "country": holdout, "y_true": y_test, "y_pred": y_pred_ols,
        "model": "OLS", "time_trend": df.loc[mask_test,"time_trend"].values
    }))

    # GBM
    gbm = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=5, subsample=0.8, random_state=42
    )
    gbm.fit(X_train, y_train)
    y_pred_gbm = gbm.predict(X_test)

    rmse_gbm = np.sqrt(mean_squared_error(y_test, y_pred_gbm))
    mae_gbm  = mean_absolute_error(y_test, y_pred_gbm)
    r2_gbm   = r2_score(y_test, y_pred_gbm)
    print(f"{holdout:<8} {'Gradient Boosting':<20} {rmse_gbm:>8.3f} {mae_gbm:>8.3f} {r2_gbm:>8.3f} {len(y_test):>6}")
    results.append({"country":holdout,"model":"GBM",
                    "rmse":rmse_gbm,"mae":mae_gbm,"r2":r2_gbm,"n":len(y_test)})

    gbm_preds_all.append(pd.DataFrame({
        "country": holdout, "y_true": y_test, "y_pred": y_pred_gbm,
        "model": "GBM", "time_trend": df.loc[mask_test,"time_trend"].values
    }))

# ── Summary ───────────────────────────────────────────────────────
df_results = pd.DataFrame(results)
summary = df_results.groupby("model")[["rmse","mae","r2"]].mean().round(3)
print("\n── SUMMARY (mean across holdout countries) ─────────────")
print(summary)
df_results.to_csv(TAB / "holdout_validation_results_v2.csv", index=False)

# ── CHART 1: Predicted vs Actual (normalised) ────────────────────
all_ols = pd.concat(ols_preds_all, ignore_index=True)
all_gbm = pd.concat(gbm_preds_all, ignore_index=True)
countries_plotted = sorted(all_ols["country"].unique())

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    "Estimated vs Observed Capacity Utilisation (z-score normalised)\n"
    "Leave-One-Country-Out Validation | Panel OLS vs Gradient Boosting",
    fontsize=12, fontweight="bold"
)

for i, country in enumerate(countries_plotted):
    ax = axes.flatten()[i]
    d_ols = all_ols[all_ols["country"]==country].sort_values("time_trend")
    d_gbm = all_gbm[all_gbm["country"]==country].sort_values("time_trend")

    ax.plot(d_ols["time_trend"], d_ols["y_true"],
            color="#2c3e50", lw=2, label="Observed", zorder=3)
    ax.plot(d_ols["time_trend"], d_ols["y_pred"],
            color="#e74c3c", lw=1.5, ls="--", label="OLS", zorder=2)
    ax.plot(d_gbm["time_trend"], d_gbm["y_pred"],
            color="#2980b9", lw=1.5, ls="-.", label="GBM", zorder=2)

    r2_o = df_results[(df_results["country"]==country)&(df_results["model"]=="OLS")]["r2"].values[0]
    r2_g = df_results[(df_results["country"]==country)&(df_results["model"]=="GBM")]["r2"].values[0]

    ax.set_title(f"{country}  |  OLS R²={r2_o:.2f}  GBM R²={r2_g:.2f}", fontsize=10)
    ax.set_xlabel("Quarter index (2010 Q1 = 1)", fontsize=8)
    ax.set_ylabel("CU (z-score)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="grey", lw=0.8, ls=":")

if len(countries_plotted) < 6:
    axes.flatten()[-1].set_visible(False)

plt.tight_layout()
p1 = FIG / "01_predicted_vs_actual_normalised.png"
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✓ Chart 1: {p1}")

# ── CHART 2: RMSE and R² comparison ──────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    "Validation Performance: Panel OLS vs Gradient Boosting\n"
    "Leave-One-Country-Out | CU z-score",
    fontsize=12, fontweight="bold"
)

x = np.arange(len(countries_plotted))
w = 0.35

ols_rmse = [df_results[(df_results["country"]==c)&(df_results["model"]=="OLS")]["rmse"].values[0] for c in countries_plotted]
gbm_rmse = [df_results[(df_results["country"]==c)&(df_results["model"]=="GBM")]["rmse"].values[0] for c in countries_plotted]
ols_r2   = [df_results[(df_results["country"]==c)&(df_results["model"]=="OLS")]["r2"].values[0] for c in countries_plotted]
gbm_r2   = [df_results[(df_results["country"]==c)&(df_results["model"]=="GBM")]["r2"].values[0] for c in countries_plotted]

ax1.bar(x-w/2, ols_rmse, w, label="Panel OLS", color="#e74c3c", alpha=0.85)
ax1.bar(x+w/2, gbm_rmse, w, label="Gradient Boosting", color="#2980b9", alpha=0.85)
ax1.axhline(np.mean(ols_rmse), color="#e74c3c", ls=":", lw=1.2, alpha=0.7, label=f"OLS mean={np.mean(ols_rmse):.2f}")
ax1.axhline(np.mean(gbm_rmse), color="#2980b9", ls=":", lw=1.2, alpha=0.7, label=f"GBM mean={np.mean(gbm_rmse):.2f}")
ax1.set_xticks(x); ax1.set_xticklabels(countries_plotted)
ax1.set_ylabel("RMSE (z-score units)"); ax1.set_title("Out-of-Sample RMSE\n(lower is better)")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3, axis="y")

ax2.bar(x-w/2, ols_r2, w, label="Panel OLS", color="#e74c3c", alpha=0.85)
ax2.bar(x+w/2, gbm_r2, w, label="Gradient Boosting", color="#2980b9", alpha=0.85)
ax2.axhline(0, color="black", lw=0.8)
ax2.axhline(np.mean(ols_r2), color="#e74c3c", ls=":", lw=1.2, alpha=0.7)
ax2.axhline(np.mean(gbm_r2), color="#2980b9", ls=":", lw=1.2, alpha=0.7)
ax2.set_xticks(x); ax2.set_xticklabels(countries_plotted)
ax2.set_ylabel("R²"); ax2.set_title("Out-of-Sample R²\n(higher is better)")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
p2 = FIG / "02_model_comparison_rmse_r2_normalised.png"
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Chart 2: {p2}")

print("\n── FINAL SUMMARY ────────────────────────────────────────")
print(summary)
print("\nNote: metrics in z-score units. R²>0 means model beats the mean.")
print("GBM improvement over OLS:")
ols_mean_r2 = summary.loc["OLS","r2"]
gbm_mean_r2 = summary.loc["GBM","r2"]
print(f"  Mean R²: OLS={ols_mean_r2:.3f} → GBM={gbm_mean_r2:.3f}")
