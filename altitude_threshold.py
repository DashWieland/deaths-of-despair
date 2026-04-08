"""
Altitude threshold analysis.

Tests whether the elevation-suicide/K70 relationship is linear or has
a threshold around 2000-2500m (where hypoxia effects on serotonin become
clinically significant per the published literature).

Uses: existing elevation_cache.json + WONDER death rates.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import UnivariateSpline

# ── Load data ──────────────────────────────────────────────────────────────────

with open("data/elevation_cache.json") as f:
    elevations = json.load(f)

def load_wonder(path, rate_col):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip().str.strip('"')
    df = df[df["County Code"].str.match(r"^\d{5}$", na=False)].copy()
    df = df.rename(columns={"County Code": "fips"})
    df["rate"] = pd.to_numeric(df["Crude Rate"], errors="coerce")
    return df[["fips", "rate"]].rename(columns={"rate": rate_col})

suicide  = load_wonder("data/suicide_county_2018_2024.csv",      "suicide_rate")
overdose = load_wonder("data/overdose_county_2018_2024.csv",     "overdose_rate")
k70      = load_wonder("data/alcohol_liver_county_2018_2024.csv","k70_rate")

df = (suicide
      .merge(overdose, on="fips", how="outer")
      .merge(k70,      on="fips", how="outer"))
df["elevation_m"] = df["fips"].map(elevations)
df = df.dropna(subset=["elevation_m"]).copy()

# ── Binned analysis ────────────────────────────────────────────────────────────

# Bins chosen to reflect natural breaks and the published hypoxia threshold
BINS = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 3000, 5000]
LABELS = ["0-300","300-600","600-900","900-1200","1200-1500",
          "1500-1800","1800-2100","2100-2400","2400-3000","3000+"]
df["elev_bin"] = pd.cut(df["elevation_m"], bins=BINS, labels=LABELS)

print("Binned mean rates by elevation (m):")
print(f"{'Bin (m)':<14} {'n':>5} {'Suicide':>10} {'K70':>10} {'Overdose':>10}")
print("-"*52)
for label in LABELS:
    sub = df[df["elev_bin"] == label]
    n = len(sub)
    su = sub["suicide_rate"].mean()
    k7 = sub["k70_rate"].mean()
    od = sub["overdose_rate"].mean()
    print(f"{label:<14} {n:>5} {su:>10.2f} {k7:>10.2f} {od:>10.2f}")

# ── Threshold test: linear vs. kinked at 2000m ────────────────────────────────

THRESHOLD = 2000  # metres — literature-suggested hypoxia threshold

for col, label in [("suicide_rate","Suicide"), ("k70_rate","K70"), ("overdose_rate","Overdose")]:
    sub = df[["elevation_m", col]].dropna()
    x = sub["elevation_m"].values
    y = sub[col].values

    # Linear model
    X_lin = np.column_stack([np.ones(len(x)), x])
    b_lin, _, _, _ = np.linalg.lstsq(X_lin, y, rcond=None)
    yh_lin = X_lin @ b_lin
    ss_res_lin = np.sum((y - yh_lin)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2_lin = 1 - ss_res_lin/ss_tot

    # Kinked (piecewise linear) model: slope changes at threshold
    x_above = np.maximum(x - THRESHOLD, 0)
    X_kink = np.column_stack([np.ones(len(x)), x, x_above])
    b_kink, _, _, _ = np.linalg.lstsq(X_kink, y, rcond=None)
    yh_kink = X_kink @ b_kink
    ss_res_kink = np.sum((y - yh_kink)**2)
    r2_kink = 1 - ss_res_kink/ss_tot

    # F-test: does adding the kink improve fit?
    n, k = len(y), 3
    f_stat = ((ss_res_lin - ss_res_kink) / 1) / (ss_res_kink / (n - k))
    p_f = stats.f.sf(f_stat, 1, n - k)

    slope_below = b_kink[1]
    slope_above = b_kink[1] + b_kink[2]

    print(f"\n{label}:")
    print(f"  Linear R²={r2_lin:.4f}")
    print(f"  Kinked R²={r2_kink:.4f}  (threshold={THRESHOLD}m)")
    print(f"  Slope below {THRESHOLD}m: {slope_below:+.5f} per m")
    print(f"  Slope above {THRESHOLD}m: {slope_above:+.5f} per m  (ratio: {slope_above/slope_below:.2f}x)")
    print(f"  F-test for kink: F={f_stat:.2f}, p={p_f:.4f}")

# ── Figure ─────────────────────────────────────────────────────────────────────

print("\nGenerating figure...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor("#f8f8f6")

configs = [
    ("suicide_rate",  "Suicide rate (per 100K)",  "#4d9de0"),
    ("k70_rate",      "K70 rate (per 100K)",       "#a44a3f"),
    ("overdose_rate", "Overdose rate (per 100K)",  "#e84855"),
]

for ax, (col, ylabel, color) in zip(axes, configs):
    sub = df[["elevation_m", col]].dropna()
    x = sub["elevation_m"].values
    y = sub[col].values

    # Scatter (thin, transparent)
    ax.scatter(x, y, s=3, alpha=0.25, color=color, linewidths=0)

    # Binned means with error bars
    bin_means = df.groupby("elev_bin", observed=True)[col].agg(["mean","sem","count"])
    bin_centers = [150, 450, 750, 1050, 1350, 1650, 1950, 2250, 2700, 3500]
    valid = bin_means["count"] >= 10
    ax.errorbar(
        [bin_centers[i] for i, v in enumerate(valid) if v],
        bin_means.loc[valid, "mean"].values,
        yerr=bin_means.loc[valid, "sem"].values * 1.96,
        fmt="o", color="black", markersize=5, linewidth=1.5,
        capsize=3, zorder=5, label="Bin mean ± 95% CI"
    )

    # Kinked fit line
    x_range = np.linspace(x.min(), x.max(), 300)
    x_above = np.maximum(x_range - THRESHOLD, 0)
    X_kink = np.column_stack([np.ones(len(x)), x, np.maximum(x - THRESHOLD, 0)])
    b_kink, _, _, _ = np.linalg.lstsq(X_kink, y, rcond=None)
    X_plot = np.column_stack([np.ones(len(x_range)), x_range, x_above])
    y_fit = X_plot @ b_kink
    ax.plot(x_range, y_fit, color="black", linewidth=2, alpha=0.9, label="Piecewise fit")

    # Threshold line
    ax.axvline(THRESHOLD, color="grey", linewidth=1, linestyle="--", alpha=0.7)
    ax.text(THRESHOLD + 50, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1,
            "2000m", fontsize=8, color="grey", va="top")

    ax.set_xlabel("County centroid elevation (m)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_facecolor("#f8f8f6")

    r, _ = stats.spearmanr(x, y)
    ax.set_title(f"Spearman ρ = {r:.3f}", fontsize=11, fontweight="bold")

axes[0].legend(fontsize=8, loc="upper left")

plt.suptitle("Elevation vs. Death Rate: Testing for Threshold at 2000m\n"
             "(black dots = binned means ± 95% CI; dashed = published hypoxia threshold)",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("figures/fig_altitude_threshold.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig_altitude_threshold.png")
print("\nDone.")
