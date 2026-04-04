"""
Altitude hypothesis: does county mean elevation predict suicide + K70 rates
independently of overdose rates?

Method:
  - County centroids (WGS84) → SRTM90m elevation via OpenTopoData API
  - Correlate elevation with each death axis
  - OLS with elevation + log(pop density) as predictors
  - Partial correlations: elevation ~ suicide/K70 controlling for rurality
"""

import warnings
warnings.filterwarnings("ignore")

import time
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── 1. Load shapefile and compute centroids ────────────────────────────────────

print("Loading county shapefile and computing centroids...")
counties = gpd.read_file(
    "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
)
counties = counties.rename(columns={"GEOID": "fips"})
EXCLUDE = {"02", "15", "60", "66", "69", "72", "78"}
counties = counties[~counties["STATEFP"].isin(EXCLUDE)].copy()

# Centroids in WGS84
centroids_wgs84 = counties.to_crs("EPSG:4326").copy()
centroids_wgs84["lon"] = centroids_wgs84.geometry.centroid.x
centroids_wgs84["lat"] = centroids_wgs84.geometry.centroid.y
coords = centroids_wgs84[["fips", "lat", "lon"]].reset_index(drop=True)

print(f"  {len(coords)} counties to query")

# ── 2. Batch-query OpenTopoData (SRTM90m, 100 locations/request) ──────────────

BATCH = 100
BASE  = "https://api.opentopodata.org/v1/srtm90m"

elevations = {}
n_batches = (len(coords) + BATCH - 1) // BATCH

import os
if os.path.exists("elevation_cache.json"):
    print("  Loading from cache...")
    with open("elevation_cache.json") as f:
        elevations = json.load(f)
    print(f"  {len(elevations)} counties loaded from cache")
else:
    print(f"Querying elevation API ({n_batches} batches, ~1 req/sec)...")
    for i in range(n_batches):
        batch = coords.iloc[i*BATCH:(i+1)*BATCH]
        loc_str = "|".join(f"{r.lat:.4f},{r.lon:.4f}" for r in batch.itertuples())
        try:
            resp = requests.get(BASE, params={"locations": loc_str}, timeout=30)
            data = resp.json()
            if data["status"] == "OK":
                for row, result in zip(batch.itertuples(), data["results"]):
                    elevations[row.fips] = result["elevation"]
            else:
                print(f"  Batch {i}: API error: {data}")
        except Exception as e:
            print(f"  Batch {i}: request failed: {e}")
        if i < n_batches - 1:
            time.sleep(1.1)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{n_batches} batches done")

    print(f"  Elevation retrieved for {len(elevations)} counties")
    with open("elevation_cache.json", "w") as f:
        json.dump(elevations, f)
    print("  Cached to elevation_cache.json")

# ── 3. Load death rates and join ──────────────────────────────────────────────

def load_wonder(path, rate_col):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip().str.strip('"')
    df = df[df["County Code"].str.match(r"^\d{5}$", na=False)].copy()
    df = df.rename(columns={"County Code": "fips"})
    df["rate"] = pd.to_numeric(df["Crude Rate"], errors="coerce")
    df["pop"]  = pd.to_numeric(df["Population"], errors="coerce")
    return df[["fips", "rate", "pop"]].rename(columns={"rate": rate_col, "pop": f"pop_{rate_col}"})

suicide  = load_wonder("suicide_county_2018_2024.csv",      "suicide_rate")
overdose = load_wonder("overdose_county_2018_2024.csv",     "overdose_rate")
k70      = load_wonder("alcohol_liver_county_2018_2024.csv","k70_rate")

df = (suicide
      .merge(overdose[["fips","overdose_rate"]], on="fips", how="outer")
      .merge(k70[["fips","k70_rate"]],           on="fips", how="outer"))

df["elevation_m"] = df["fips"].map(elevations)
aland = counties.set_index("fips")["ALAND"].reindex(df["fips"].values).values
df["pop_density"] = df["pop_suicide_rate"] / aland * 1e6
df["log_density"] = np.log1p(df["pop_density"])

df = df.dropna(subset=["elevation_m"]).copy()
print(f"\nWorking dataset: {len(df)} counties with elevation data")

# ── 4. Correlations ────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("PEARSON CORRELATIONS WITH ELEVATION")
print("="*60)
print(f"{'Axis':<25} {'r':>8} {'p':>10} {'n':>6}")
print("-"*60)

for col, label in [("suicide_rate","Suicide"), ("k70_rate","K70 (liver)"), ("overdose_rate","Overdose")]:
    sub = df[["elevation_m", col]].dropna()
    r, p = stats.pearsonr(sub["elevation_m"], sub[col])
    print(f"{label:<25} {r:>8.4f} {p:>10.4f} {len(sub):>6}")

# Spearman too (rates are skewed)
print("\n" + "="*60)
print("SPEARMAN CORRELATIONS WITH ELEVATION")
print("="*60)
print(f"{'Axis':<25} {'rho':>8} {'p':>10} {'n':>6}")
print("-"*60)

for col, label in [("suicide_rate","Suicide"), ("k70_rate","K70 (liver)"), ("overdose_rate","Overdose")]:
    sub = df[["elevation_m", col]].dropna()
    r, p = stats.spearmanr(sub["elevation_m"], sub[col])
    print(f"{label:<25} {r:>8.4f} {p:>10.4f} {len(sub):>6}")

# ── 5. Partial correlations (elevation ~ rate controlling for log pop density) ─

def partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z."""
    def resid(a, b):
        slope, intercept, *_ = stats.linregress(b, a)
        return a - (slope * b + intercept)
    rx = resid(x, z)
    ry = resid(y, z)
    return stats.pearsonr(rx, ry)

print("\n" + "="*60)
print("PARTIAL CORRELATIONS: elevation ~ rate | log(pop density)")
print("="*60)
print(f"{'Axis':<25} {'r_partial':>10} {'p':>10}")
print("-"*60)

for col, label in [("suicide_rate","Suicide"), ("k70_rate","K70 (liver)"), ("overdose_rate","Overdose")]:
    sub = df[["elevation_m", col, "log_density"]].dropna()
    r, p = partial_corr(sub["elevation_m"].values, sub[col].values, sub["log_density"].values)
    print(f"{label:<25} {r:>10.4f} {p:>10.4f}")

# ── 6. OLS: each rate ~ elevation + log(density) ──────────────────────────────

print("\n" + "="*60)
print("OLS: rate ~ elevation_m + log(pop_density)")
print("="*60)

for col, label in [("suicide_rate","Suicide"), ("k70_rate","K70 (liver)"), ("overdose_rate","Overdose")]:
    sub = df[["elevation_m", col, "log_density"]].dropna()
    X = np.column_stack([np.ones(len(sub)), sub["elevation_m"], sub["log_density"]])
    y = sub[col].values
    # OLS via lstsq
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot
    # t-stats for elevation coefficient
    n, k = len(y), 3
    mse = ss_res / (n - k)
    XtXinv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(mse * XtXinv))
    t_elev = coef[1] / se[1]
    p_elev = 2 * stats.t.sf(abs(t_elev), df=n-k)
    print(f"\n{label}:")
    print(f"  intercept={coef[0]:.3f}  elev_coef={coef[1]:.5f}  density_coef={coef[2]:.3f}")
    print(f"  R²={r2:.4f}  t(elevation)={t_elev:.3f}  p={p_elev:.4f}")

# ── 7. Figure ──────────────────────────────────────────────────────────────────

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
    elev = sub["elevation_m"].values
    rate = sub[col].values
    r, p  = stats.spearmanr(elev, rate)
    sig   = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

    ax.scatter(elev, rate, s=3, alpha=0.35, color=color, linewidths=0)

    # Trend line
    m, b, *_ = stats.linregress(elev, rate)
    xr = np.array([elev.min(), elev.max()])
    ax.plot(xr, m*xr + b, color="black", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("County centroid elevation (m)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"ρ = {r:.3f}{sig}", fontsize=12, fontweight="bold")
    ax.set_facecolor("#f8f8f6")

plt.suptitle("Elevation vs. Death Rates by Axis\n(Spearman ρ, county centroids, SRTM90m)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("fig_altitude.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig_altitude.png")
print("\nDone.")
