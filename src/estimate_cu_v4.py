#!/usr/bin/env python3
"""
Paper 1: Capacity Without a Survey — v4
Implements decomposition steps from review:
1. Country-specific correlations: CU vs IP, CU vs electricity
2. Exclude 2020–2021 entirely
3. Add lagged predictors (1q and 4q)
4. Test level vs growth vs gap transformations
5. IP-only with lags as clean benchmark
6. Interaction: does electricity help in manufacturing-intensive countries?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE = Path.home() / "projects" / "global-cu-inflation"
HARM = BASE / "data" / "harmonised"
RAW  = BASE / "data" / "raw"
FIG  = BASE / "outputs" / "figures"
TAB  = BASE / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# ── Load ─────────────────────────────────────────────────────────
panel = pd.read_csv(HARM / "cu_panel_quarterly.csv")
weather = pd.read_csv(RAW / "weather_hdd_cdd_quarterly.csv")
panel = panel.merge(weather[["country_code","year","quarter","hdd_sum","cdd_sum"]],
                    on=["country_code","year","quarter"], how="left")
panel = panel[panel["country_code"] != "JPN"].reset_index(drop=True)
panel = panel.sort_values(["country_code","year","quarter"]).reset_index(drop=True)

COUNTRIES = sorted(panel["country_code"].unique())

# Within-country z-score
cu_stats = panel.groupby("country_code")["cu_pct"].agg(["mean","std"]).rename(
    columns={"mean":"cu_mean","std":"cu_std"})
panel = panel.join(cu_stats, on="country_code")
panel["cu_z"] = (panel["cu_pct"] - panel["cu_mean"]) / panel["cu_std"]

# ── Feature construction ──────────────────────────────────────────
panel["ip_qoq"]    = panel.groupby("country_code")["ip_index"].pct_change(1)
panel["ip_yoy"]    = panel.groupby("country_code")["ip_index"].pct_change(4)
panel["ip_gap"]    = panel.groupby("country_code")["ip_index"].transform(
    lambda x: x - x.rolling(8, min_periods=4).mean())
panel["ip_lag1"]   = panel.groupby("country_code")["ip_index"].shift(1)
panel["ip_lag4"]   = panel.groupby("country_code")["ip_index"].shift(4)
panel["ip_qoq_lag1"] = panel.groupby("country_code")["ip_qoq"].shift(1)
panel["ip_qoq_lag4"] = panel.groupby("country_code")["ip_qoq"].shift(4)

panel["elec_qoq"]  = panel.groupby("country_code")["electricity_twh"].pct_change(1)
panel["elec_yoy"]  = panel.groupby("country_code")["electricity_twh"].pct_change(4)
panel["elec_gap"]  = panel.groupby("country_code")["electricity_twh"].transform(
    lambda x: x - x.rolling(8, min_periods=4).mean())

panel["hdd_z"] = panel.groupby("country_code")["hdd_sum"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8))
panel["cdd_z"] = panel.groupby("country_code")["cdd_sum"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8))

panel["unemp_chg"] = panel.groupby("country_code")["unemployment_pct"].pct_change(4)
panel["time_trend"] = (panel["year"] - 2010) * 4 + panel["quarter"]
for q in [2,3,4]:
    panel[f"q{q}"] = (panel["quarter"] == q).astype(int)

# COVID flag
panel["covid"] = panel["year"].isin([2020,2021]).astype(int)

# ── CHART 1: Country-specific correlations ────────────────────────
print("── COUNTRY CORRELATIONS: CU vs IP and ELECTRICITY ──────")
df_obs = panel[panel["cu_z"].notna()].copy()

fig, axes = plt.subplots(2, 6, figsize=(18, 8))
fig.suptitle("CU Cycle vs Industrial Production and Electricity — By Country\n"
             "(z-score normalised within country)", fontsize=12, fontweight="bold")

corr_records = []
for i, country in enumerate(COUNTRIES):
    d = df_obs[df_obs["country_code"] == country].dropna(subset=["ip_qoq","elec_qoq"])

    # CU vs IP growth
    ax1 = axes[0, i]
    ax1.scatter(d["ip_qoq"], d["cu_z"], alpha=0.6, s=20, color="#2980b9")
    if len(d) > 3:
        m, b = np.polyfit(d["ip_qoq"].dropna(), d.loc[d["ip_qoq"].notna(),"cu_z"], 1)
        x_line = np.linspace(d["ip_qoq"].min(), d["ip_qoq"].max(), 50)
        ax1.plot(x_line, m*x_line + b, color="#e74c3c", lw=1.5)
        corr_ip = d[["ip_qoq","cu_z"]].dropna().corr().iloc[0,1]
    else:
        corr_ip = np.nan
    ax1.set_title(f"{country}\nIP growth vs CU\nr={corr_ip:.2f}", fontsize=9)
    ax1.set_xlabel("IP QoQ growth", fontsize=7)
    ax1.set_ylabel("CU z-score", fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=7)

    # CU vs electricity growth
    ax2 = axes[1, i]
    ax2.scatter(d["elec_qoq"], d["cu_z"], alpha=0.6, s=20, color="#f39c12")
    if len(d) > 3:
        valid = d[["elec_qoq","cu_z"]].dropna()
        if len(valid) > 3:
            m2, b2 = np.polyfit(valid["elec_qoq"], valid["cu_z"], 1)
            x2 = np.linspace(valid["elec_qoq"].min(), valid["elec_qoq"].max(), 50)
            ax2.plot(x2, m2*x2 + b2, color="#e74c3c", lw=1.5)
            corr_el = valid.corr().iloc[0,1]
        else:
            corr_el = np.nan
    else:
        corr_el = np.nan
    ax2.set_title(f"{country}\nElec growth vs CU\nr={corr_el:.2f}", fontsize=9)
    ax2.set_xlabel("Elec QoQ growth", fontsize=7)
    ax2.set_ylabel("CU z-score", fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=7)

    corr_records.append({"country": country, "corr_ip": corr_ip, "corr_elec": corr_el})
    print(f"  {country}: IP r={corr_ip:.3f}  Elec r={corr_el:.3f}")

plt.tight_layout()
p_corr = FIG / "07_country_correlations.png"
plt.savefig(p_corr, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Chart 1: {p_corr}")

df_corr = pd.DataFrame(corr_records)
df_corr.to_csv(TAB / "country_correlations.csv", index=False)

# ── LOCO validation function ──────────────────────────────────────
def run_loco(df_in, features, label, alpha=1.0, use_gbm=False):
    countries = sorted(df_in["country_code"].unique())
    results = []
    preds_all = []

    for holdout in countries:
        mask_tr = df_in["country_code"] != holdout
        mask_te = df_in["country_code"] == holdout

        X_tr = df_in.loc[mask_tr, features].values
        y_tr = df_in.loc[mask_tr, "cu_z"].values
        X_te = df_in.loc[mask_te, features].values
        y_te = df_in.loc[mask_te, "cu_z"].values
        tt   = df_in.loc[mask_te, "time_trend"].values
        yr   = df_in.loc[mask_te, "year"].values

        if len(y_te) < 4:
            continue

        if use_gbm:
            model = GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.03,
                min_samples_leaf=4, subsample=0.8, random_state=42)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
        else:
            pipe = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=alpha))])
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)

        results.append({
            "country": holdout, "model": label,
            "rmse": np.sqrt(mean_squared_error(y_te, y_pred)),
            "mae":  mean_absolute_error(y_te, y_pred),
            "r2":   r2_score(y_te, y_pred),
            "n":    len(y_te)
        })
        preds_all.append(pd.DataFrame({
            "country": holdout, "model": label,
            "y_true": y_te, "y_pred": y_pred,
            "time_trend": tt, "year": yr
        }))

    return pd.DataFrame(results), pd.concat(preds_all, ignore_index=True) if preds_all else pd.DataFrame()

# ── Feature sets ──────────────────────────────────────────────────
FEAT_IP_BASIC  = ["ip_qoq","ip_yoy","ip_gap","q2","q3","q4"]
FEAT_IP_LAGGED = ["ip_qoq","ip_yoy","ip_gap","ip_qoq_lag1","ip_qoq_lag4","q2","q3","q4"]
FEAT_FULL      = ["ip_qoq","ip_yoy","ip_gap","ip_qoq_lag1","ip_qoq_lag4",
                  "elec_qoq","elec_yoy","elec_gap","hdd_z","cdd_z",
                  "unemp_chg","time_trend","q2","q3","q4"]

# ── Experiment 1: Full sample ─────────────────────────────────────
print("\n── EXPERIMENT 1: FULL SAMPLE (2010–2024) ───────────────")
df_full = panel[panel["cu_z"].notna()].dropna(subset=FEAT_FULL).copy()

all_results = []
all_preds   = []

for feats, label, gbm in [
    (FEAT_IP_BASIC,  "IP-basic Ridge",  False),
    (FEAT_IP_LAGGED, "IP-lagged Ridge", False),
    (FEAT_FULL,      "Full Ridge",      False),
    (FEAT_FULL,      "Full GBM",        True),
]:
    r, p = run_loco(df_full, feats, label, use_gbm=gbm)
    all_results.append(r)
    all_preds.append(p)

df_results_full = pd.concat(all_results, ignore_index=True)
summary_full = df_results_full.groupby("model")[["rmse","r2"]].mean().round(3).sort_values("r2", ascending=False)
print(summary_full)

# ── Experiment 2: Exclude 2020–2021 ──────────────────────────────
print("\n── EXPERIMENT 2: EXCLUDING 2020–2021 ───────────────────")
df_nocovid = panel[
    (panel["cu_z"].notna()) & (~panel["year"].isin([2020,2021]))
].dropna(subset=FEAT_FULL).copy()

all_results_nc = []
for feats, label, gbm in [
    (FEAT_IP_BASIC,  "IP-basic Ridge",  False),
    (FEAT_IP_LAGGED, "IP-lagged Ridge", False),
    (FEAT_FULL,      "Full Ridge",      False),
    (FEAT_FULL,      "Full GBM",        True),
]:
    r, _ = run_loco(df_nocovid, feats, label, use_gbm=gbm)
    r["sample"] = "No COVID"
    all_results_nc.append(r)

df_results_nc = pd.concat(all_results_nc, ignore_index=True)
summary_nc = df_results_nc.groupby("model")[["rmse","r2"]].mean().round(3).sort_values("r2", ascending=False)
print(summary_nc)

# ── Experiment 3: Pre-COVID only (2010–2019) ──────────────────────
print("\n── EXPERIMENT 3: PRE-COVID ONLY (2010–2019) ────────────")
df_pre = panel[
    (panel["cu_z"].notna()) & (panel["year"] <= 2019)
].dropna(subset=FEAT_FULL).copy()

all_results_pre = []
for feats, label, gbm in [
    (FEAT_IP_BASIC,  "IP-basic Ridge",  False),
    (FEAT_IP_LAGGED, "IP-lagged Ridge", False),
    (FEAT_FULL,      "Full Ridge",      False),
    (FEAT_FULL,      "Full GBM",        True),
]:
    r, _ = run_loco(df_pre, feats, label, use_gbm=gbm)
    r["sample"] = "Pre-COVID"
    all_results_pre.append(r)

df_results_pre = pd.concat(all_results_pre, ignore_index=True)
summary_pre = df_results_pre.groupby("model")[["rmse","r2"]].mean().round(3).sort_values("r2", ascending=False)
print(summary_pre)

# ── Country-level breakdown ───────────────────────────────────────
print("\n── COUNTRY BREAKDOWN: IP-lagged Ridge ──────────────────")
ip_lag_full = df_results_full[df_results_full["model"]=="IP-lagged Ridge"]
ip_lag_nc   = df_results_nc[df_results_nc["model"]=="IP-lagged Ridge"]
ip_lag_pre  = df_results_pre[df_results_pre["model"]=="IP-lagged Ridge"]

country_breakdown = ip_lag_full[["country","r2","rmse"]].merge(
    ip_lag_nc[["country","r2"]].rename(columns={"r2":"r2_nocovid"}), on="country", how="left"
).merge(
    ip_lag_pre[["country","r2"]].rename(columns={"r2":"r2_precovid"}), on="country", how="left"
)
country_breakdown.columns = ["country","r2_full","rmse_full","r2_nocovid","r2_precovid"]
print(country_breakdown.round(3).to_string(index=False))
country_breakdown.to_csv(TAB / "country_breakdown_v4.csv", index=False)

# ── Save all results ──────────────────────────────────────────────
combined = pd.concat([
    df_results_full.assign(sample="Full"),
    df_results_nc,
    df_results_pre
], ignore_index=True)
combined.to_csv(TAB / "holdout_results_v4.csv", index=False)

# ── CHART 2: R² comparison across experiments ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle("Leave-One-Country-Out R² — Model and Sample Comparison\n"
             "IP-lagged Ridge vs Full Ridge vs GBM | CU z-score",
             fontsize=12, fontweight="bold")

samples = [
    ("Full sample\n(2010–2024)", df_results_full),
    ("Excluding 2020–2021", df_results_nc),
    ("Pre-COVID only\n(2010–2019)", df_results_pre),
]
colors_model = {
    "IP-basic Ridge":  "#95a5a6",
    "IP-lagged Ridge": "#2980b9",
    "Full Ridge":      "#e74c3c",
    "Full GBM":        "#27ae60",
}
models_order = ["IP-basic Ridge","IP-lagged Ridge","Full Ridge","Full GBM"]

for ax, (title, df_s) in zip(axes, samples):
    summary_s = df_s.groupby("model")["r2"].mean()
    vals  = [summary_s.get(m, np.nan) for m in models_order]
    cols  = [colors_model[m] for m in models_order]
    bars  = ax.bar(range(len(models_order)), vals, color=cols, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(models_order)))
    ax.set_xticklabels([m.replace(" ","\n") for m in models_order], fontsize=8)
    ax.set_ylabel("Mean R²")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ypos = bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.04
            ax.text(bar.get_x()+bar.get_width()/2, ypos, f"{val:.2f}",
                    ha="center", fontsize=8)

plt.tight_layout()
p2 = FIG / "08_r2_by_sample_and_model.png"
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✓ Chart 2: {p2}")

# ── CHART 3: Country breakdown across samples ─────────────────────
countries_plot = sorted(country_breakdown["country"].unique())
x = np.arange(len(countries_plot))
w = 0.25

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("IP-lagged Ridge: R² by Country and Sample\n"
             "Leave-One-Country-Out | CU z-score", fontsize=12, fontweight="bold")

ax.bar(x - w,   country_breakdown.set_index("country").loc[countries_plot,"r2_full"].values,
       w, label="Full (2010–2024)", color="#2980b9", alpha=0.85)
ax.bar(x,       country_breakdown.set_index("country").loc[countries_plot,"r2_nocovid"].values,
       w, label="Excl. 2020–2021", color="#e74c3c", alpha=0.85)
ax.bar(x + w,   country_breakdown.set_index("country").loc[countries_plot,"r2_precovid"].values,
       w, label="Pre-COVID only", color="#27ae60", alpha=0.85)

ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(countries_plot, fontsize=10)
ax.set_ylabel("R²")
ax.set_title("R² > 0 = beats country mean | R² < 0 = worse than mean", fontsize=9)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
p3 = FIG / "09_country_r2_by_sample.png"
plt.savefig(p3, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Chart 3: {p3}")

print("\n── FINAL SUMMARY ────────────────────────────────────────")
print("\nFull sample:")
print(summary_full[["rmse","r2"]].to_string())
print("\nExcluding 2020–2021:")
print(summary_nc[["rmse","r2"]].to_string())
print("\nPre-COVID only:")
print(summary_pre[["rmse","r2"]].to_string())
