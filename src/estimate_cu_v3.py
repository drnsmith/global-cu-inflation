#!/usr/bin/env python3
"""
Paper 1: Capacity Without a Survey — v3
Fixes applied:
- CAN included in holdout
- No preprocessing leakage (scaler fitted on in-fold training only)
- Ridge regression replacing OLS (handles multicollinearity)
- Cyclical features only (growth rates, gaps — no repeated annual levels)
- Weather HDD/CDD included
- Simple baselines: zero, IP-only, electricity-only
- Pre/post COVID split results
- Japan excluded (diffusion index, not CU rate)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

BASE = Path.home() / "projects" / "global-cu-inflation"
HARM = BASE / "data" / "harmonised"
RAW  = BASE / "data" / "raw"
FIG  = BASE / "outputs" / "figures"
TAB  = BASE / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# ── Load and merge ────────────────────────────────────────────────
panel = pd.read_csv(HARM / "cu_panel_quarterly.csv")
weather = pd.read_csv(RAW / "weather_hdd_cdd_quarterly.csv")

panel = panel.merge(weather[["country_code","year","quarter","hdd_sum","cdd_sum"]],
                    on=["country_code","year","quarter"], how="left")

# Drop Japan — confirmed diffusion index
panel = panel[panel["country_code"] != "JPN"].reset_index(drop=True)
panel = panel.sort_values(["country_code","year","quarter"]).reset_index(drop=True)

COUNTRIES = sorted(panel["country_code"].unique())
print(f"Countries: {COUNTRIES}")

# ── Within-country z-score of CU ─────────────────────────────────
cu_stats = panel.groupby("country_code")["cu_pct"].agg(["mean","std"]).rename(
    columns={"mean":"cu_mean","std":"cu_std"})
panel = panel.join(cu_stats, on="country_code")
panel["cu_z"] = (panel["cu_pct"] - panel["cu_mean"]) / panel["cu_std"]

# ── Cyclical features only ────────────────────────────────────────
# Growth rates and deviations — no annual levels repeated across quarters
panel = panel.sort_values(["country_code","year","quarter"]).reset_index(drop=True)

# IP: quarter-on-quarter growth and year-on-year growth
panel["ip_qoq"]  = panel.groupby("country_code")["ip_index"].pct_change(1)
panel["ip_yoy"]  = panel.groupby("country_code")["ip_index"].pct_change(4)

# IP gap: deviation from 8-quarter rolling mean (cyclical component)
panel["ip_gap"]  = panel.groupby("country_code")["ip_index"].transform(
    lambda x: x - x.rolling(8, min_periods=4).mean())

# Electricity: growth and gap
panel["elec_qoq"] = panel.groupby("country_code")["electricity_twh"].pct_change(1)
panel["elec_yoy"] = panel.groupby("country_code")["electricity_twh"].pct_change(4)
panel["elec_gap"] = panel.groupby("country_code")["electricity_twh"].transform(
    lambda x: x - x.rolling(8, min_periods=4).mean())

# Weather: normalise within country
panel["hdd_z"] = panel.groupby("country_code")["hdd_sum"].transform(
    lambda x: (x - x.mean()) / x.std())
panel["cdd_z"] = panel.groupby("country_code")["cdd_sum"].transform(
    lambda x: (x - x.mean()) / x.std())

# Unemployment change (cyclical, not level)
panel["unemp_chg"] = panel.groupby("country_code")["unemployment_pct"].pct_change(4)

# Time trend (captures secular drift)
panel["time_trend"] = (panel["year"] - 2010) * 4 + panel["quarter"]

# Quarter dummies (seasonality)
for q in [2,3,4]:
    panel[f"q{q}"] = (panel["quarter"] == q).astype(int)

FEATURES = [
    "ip_qoq", "ip_yoy", "ip_gap",
    "elec_qoq", "elec_yoy", "elec_gap",
    "hdd_z", "cdd_z",
    "unemp_chg", "time_trend",
    "q2", "q3", "q4"
]

# Training set: observed CU, drop rows with missing features
df = panel[panel["cu_z"].notna()].dropna(subset=FEATURES).copy()
df = df.reset_index(drop=True)
print(f"Training observations: {len(df)}")
print(f"Countries with observed CU: {sorted(df['country_code'].unique())}")

# ── Baselines ─────────────────────────────────────────────────────
def baseline_zero(y_test):
    """Predict zero (= country mean after z-scoring)"""
    y_pred = np.zeros(len(y_test))
    return y_pred

def eval_metrics(y_true, y_pred, label):
    return {
        "model": label,
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae":  mean_absolute_error(y_true, y_pred),
        "r2":   r2_score(y_true, y_pred)
    }

# ── Leave-one-country-out validation ─────────────────────────────
countries_to_holdout = sorted(df["country_code"].unique())
print(f"\nHoldout countries: {countries_to_holdout}")

all_results = []
all_preds   = []

print(f"\n{'Country':<8} {'Model':<22} {'RMSE':>7} {'MAE':>7} {'R²':>8} {'N':>5}")
print("-" * 60)

for holdout in countries_to_holdout:
    mask_train = df["country_code"] != holdout
    mask_test  = df["country_code"] == holdout

    X_train = df.loc[mask_train, FEATURES].values
    y_train = df.loc[mask_train, "cu_z"].values
    X_test  = df.loc[mask_test,  FEATURES].values
    y_test  = df.loc[mask_test,  "cu_z"].values
    tt_test = df.loc[mask_test,  "time_trend"].values
    yr_test = df.loc[mask_test,  "year"].values

    if len(y_test) < 4:
        continue

    n = len(y_test)

    # Baseline: zero (predict country mean)
    y_zero = baseline_zero(y_test)
    r = eval_metrics(y_test, y_zero, "Zero baseline")
    r.update({"country": holdout, "n": n})
    all_results.append(r)

    # Baseline: IP-only Ridge (no leakage — fit scaler on train)
    ip_features = ["ip_qoq","ip_yoy","ip_gap"]
    X_tr_ip = df.loc[mask_train, ip_features].values
    X_te_ip = df.loc[mask_test,  ip_features].values
    pipe_ip = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
    pipe_ip.fit(X_tr_ip, y_train)
    y_ip = pipe_ip.predict(X_te_ip)
    r = eval_metrics(y_test, y_ip, "IP-only Ridge")
    r.update({"country": holdout, "n": n})
    all_results.append(r)

    # Baseline: electricity-only Ridge
    elec_features = ["elec_qoq","elec_yoy","elec_gap","hdd_z","cdd_z"]
    X_tr_el = df.loc[mask_train, elec_features].values
    X_te_el = df.loc[mask_test,  elec_features].values
    pipe_el = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
    pipe_el.fit(X_tr_el, y_train)
    y_el = pipe_el.predict(X_te_el)
    r = eval_metrics(y_test, y_el, "Elec-only Ridge")
    r.update({"country": holdout, "n": n})
    all_results.append(r)

    # Ridge (full features — no leakage)
    pipe_ridge = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
    pipe_ridge.fit(X_train, y_train)
    y_ridge = pipe_ridge.predict(X_test)
    r = eval_metrics(y_test, y_ridge, "Ridge (full)")
    r.update({"country": holdout, "n": n})
    all_results.append(r)

    # GBM (no scaling needed, no leakage risk)
    gbm = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        min_samples_leaf=4, subsample=0.8, random_state=42
    )
    gbm.fit(X_train, y_train)
    y_gbm = gbm.predict(X_test)
    r = eval_metrics(y_test, y_gbm, "GBM (full)")
    r.update({"country": holdout, "n": n})
    all_results.append(r)

    # Print
    for res in all_results[-5:]:
        print(f"{holdout:<8} {res['model']:<22} {res['rmse']:>7.3f} {res['mae']:>7.3f} {res['r2']:>8.3f} {n:>5}")

    # Store predictions for plotting
    all_preds.append(pd.DataFrame({
        "country": holdout,
        "time_trend": tt_test,
        "year": yr_test,
        "y_true": y_test,
        "y_zero": y_zero,
        "y_ip": y_ip,
        "y_el": y_el,
        "y_ridge": y_ridge,
        "y_gbm": y_gbm,
    }))
    print()

# ── Summary table ─────────────────────────────────────────────────
df_results = pd.DataFrame(all_results)
summary = df_results.groupby("model")[["rmse","mae","r2"]].mean().round(3).sort_values("r2", ascending=False)
print("\n── SUMMARY (mean across holdout countries) ─────────────")
print(summary)
df_results.to_csv(TAB / "holdout_results_v3.csv", index=False)

# ── Pre/post COVID split ───────────────────────────────────────────
print("\n── PRE-COVID (2010–2019) vs POST-COVID (2021–2024) ─────")
df_preds = pd.concat(all_preds, ignore_index=True)

for period, yr_min, yr_max in [("Pre-COVID", 2010, 2019), ("Post-COVID", 2021, 2024)]:
    mask = (df_preds["year"] >= yr_min) & (df_preds["year"] <= yr_max)
    sub = df_preds[mask]
    if len(sub) == 0:
        continue
    print(f"\n{period}:")
    for model_col, label in [("y_ridge","Ridge"),("y_gbm","GBM")]:
        r2  = r2_score(sub["y_true"], sub[model_col])
        rmse = np.sqrt(mean_squared_error(sub["y_true"], sub[model_col]))
        print(f"  {label:<8} RMSE={rmse:.3f}  R²={r2:.3f}")

# ── CHART 1: Predicted vs Actual — GBM and Ridge ─────────────────
countries_plotted = sorted(df_preds["country"].unique())
n_c = len(countries_plotted)
ncols = 3
nrows = int(np.ceil(n_c / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows*4))
fig.suptitle(
    "Estimated vs Observed CU (z-score) — v3 with weather, cyclical features\n"
    "Leave-One-Country-Out | Ridge vs GBM vs Baselines",
    fontsize=12, fontweight="bold"
)

axes_flat = axes.flatten() if nrows > 1 else axes

for i, country in enumerate(countries_plotted):
    ax = axes_flat[i]
    d = df_preds[df_preds["country"]==country].sort_values("time_trend")

    ax.plot(d["time_trend"], d["y_true"],  color="#2c3e50", lw=2,   label="Observed", zorder=4)
    ax.plot(d["time_trend"], d["y_ip"],    color="#95a5a6", lw=1,   ls=":",  label="IP-only", zorder=1)
    ax.plot(d["time_trend"], d["y_el"],    color="#f39c12", lw=1,   ls=":",  label="Elec-only", zorder=1)
    ax.plot(d["time_trend"], d["y_ridge"], color="#e74c3c", lw=1.5, ls="--", label="Ridge", zorder=2)
    ax.plot(d["time_trend"], d["y_gbm"],   color="#2980b9", lw=1.5, ls="-.", label="GBM", zorder=3)

    ax.axvline(x=40, color="grey", lw=0.8, ls=":", alpha=0.7)  # COVID Q1 2020
    ax.axhline(y=0,  color="grey", lw=0.6, ls=":")

    r2_ridge = df_results[(df_results["country"]==country)&(df_results["model"]=="Ridge (full)")]["r2"].values
    r2_gbm   = df_results[(df_results["country"]==country)&(df_results["model"]=="GBM (full)")]["r2"].values
    r2_r = r2_ridge[0] if len(r2_ridge) else np.nan
    r2_g = r2_gbm[0]   if len(r2_gbm)   else np.nan

    ax.set_title(f"{country}  |  Ridge R²={r2_r:.2f}  GBM R²={r2_g:.2f}", fontsize=10)
    ax.set_xlabel("Quarter (2010 Q1=1)", fontsize=8)
    ax.set_ylabel("CU (z-score)", fontsize=8)
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(True, alpha=0.25)

for j in range(i+1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.tight_layout()
p1 = FIG / "05_predicted_vs_actual_v3.png"
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✓ Chart 1: {p1}")

# ── CHART 2: Model comparison bar chart ───────────────────────────
models_order = ["Zero baseline","IP-only Ridge","Elec-only Ridge","Ridge (full)","GBM (full)"]
colors       = ["#bdc3c7","#95a5a6","#f39c12","#e74c3c","#2980b9"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Validation Performance — All Models vs Baselines\n"
    "Leave-One-Country-Out Mean | CU z-score",
    fontsize=12, fontweight="bold"
)

x = np.arange(len(models_order))
rmse_vals = [summary.loc[m,"rmse"] if m in summary.index else np.nan for m in models_order]
r2_vals   = [summary.loc[m,"r2"]   if m in summary.index else np.nan for m in models_order]

bars1 = ax1.bar(x, rmse_vals, color=colors, alpha=0.85, edgecolor="white")
ax1.set_xticks(x)
ax1.set_xticklabels([m.replace(" ","\n") for m in models_order], fontsize=8)
ax1.set_ylabel("Mean RMSE (z-score)")
ax1.set_title("Out-of-Sample RMSE\n(lower is better)")
ax1.grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars1, rmse_vals):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
             f"{val:.2f}", ha="center", va="bottom", fontsize=8)

bars2 = ax2.bar(x, r2_vals, color=colors, alpha=0.85, edgecolor="white")
ax2.axhline(0, color="black", lw=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels([m.replace(" ","\n") for m in models_order], fontsize=8)
ax2.set_ylabel("Mean R²")
ax2.set_title("Out-of-Sample R²\n(higher is better, >0 beats mean)")
ax2.grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars2, r2_vals):
    ypos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.08
    ax2.text(bar.get_x()+bar.get_width()/2, ypos,
             f"{val:.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
p2 = FIG / "06_model_comparison_v3.png"
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Chart 2: {p2}")

print("\n── FINAL SUMMARY ────────────────────────────────────────")
print(summary.to_string())
print("\nNote: R²>0 = beats zero baseline (country mean). COVID marker at Q40.")
