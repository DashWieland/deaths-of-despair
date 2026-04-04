"""
Temporal analysis: was the regime split always there?

Compares bivariate Moran's I between suicide and overdose in:
  - 1999-2005 (pre-opioid epidemic peak, 7-year pool)
  - 2018-2024 (fentanyl era, 7-year pool)

If the spatial independence (BV I ≈ 0) was present in both periods,
the two-regime structure is structural — pre-existing the manufactured
opioid crisis. If the early period shows positive co-clustering that
collapsed by 2018-2024, then the opioid epidemic actively split the
geography. Either result is interesting.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import libpysal
from esda import Moran, Moran_BV, Moran_Local

# ── Load data ──────────────────────────────────────────────────────────────────

def load_wonder(path, rate_col):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip().str.strip('"')
    df = df[df["County Code"].str.match(r"^\d{5}$", na=False)].copy()
    df = df.rename(columns={"County Code": "fips"})
    df["rate"] = pd.to_numeric(df["Crude Rate"], errors="coerce")
    return df[["fips", "rate"]].rename(columns={"rate": rate_col})

# Current era
su_now = load_wonder("suicide_county_2018_2024.csv",      "suicide_rate")
od_now = load_wonder("overdose_county_2018_2024.csv",     "overdose_rate")
k7_now = load_wonder("alcohol_liver_county_2018_2024.csv","k70_rate")

# Historical era
su_old = load_wonder("suicide_county_1999_2005.csv",      "suicide_rate")
od_old = load_wonder("overdose_county_1999_2005.csv",     "overdose_rate")
k7_old = load_wonder("alcohol_liver_county_1999_2005.csv","k70_rate")

for label, su, od, k7 in [("1999-2005", su_old, od_old, k7_old),
                            ("2018-2024", su_now, od_now, k7_now)]:
    print(f"{label}: suicide={len(su)}, overdose={len(od)}, K70={len(k7)}")

# ── Spatial setup ──────────────────────────────────────────────────────────────

print("\nLoading shapefile...")
counties = gpd.read_file(
    "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
)
counties = counties.rename(columns={"GEOID": "fips"})
EXCLUDE = {"02", "15", "60", "66", "69", "72", "78"}
counties = counties[~counties["STATEFP"].isin(EXCLUDE)].copy()
counties = counties.to_crs("EPSG:5070")

def build_gdf(su_df, od_df, k7_df):
    merged = (su_df
              .merge(od_df, on="fips", how="outer")
              .merge(k7_df, on="fips", how="outer"))
    gdf = counties.merge(merged, on="fips", how="left")
    gdf_su_od = gdf[gdf["suicide_rate"].notna() & gdf["overdose_rate"].notna()].copy().reset_index(drop=True)
    gdf_all   = gdf[gdf["suicide_rate"].notna() & gdf["overdose_rate"].notna() &
                    gdf["k70_rate"].notna()].copy().reset_index(drop=True)
    return gdf_su_od, gdf_all

gdf_su_od_old, gdf_all_old = build_gdf(su_old, od_old, k7_old)
gdf_su_od_now, gdf_all_now = build_gdf(su_now, od_now, k7_now)

print(f"\nSpatial subsets:")
print(f"  1999-2005 suicide+overdose: {len(gdf_su_od_old)} | all three: {len(gdf_all_old)}")
print(f"  2018-2024 suicide+overdose: {len(gdf_su_od_now)} | all three: {len(gdf_all_now)}")

def build_weights(gdf):
    w = libpysal.weights.Queen.from_dataframe(gdf, silence_warnings=True)
    w.transform = "r"
    return w

print("Building weights...")
w_old = build_weights(gdf_su_od_old)
w_now = build_weights(gdf_su_od_now)
w_old_all = build_weights(gdf_all_old)
w_now_all = build_weights(gdf_all_now)

# ── Univariate Moran's I: both eras ───────────────────────────────────────────

print("\nComputing Moran's I (999 permutations)...")

mi = {}
for era, gdf_su_od, gdf_all, w, w_all in [
    ("1999-2005", gdf_su_od_old, gdf_all_old, w_old, w_old_all),
    ("2018-2024", gdf_su_od_now, gdf_all_now, w_now, w_now_all),
]:
    mi[era] = {
        "suicide":  Moran(gdf_su_od["suicide_rate"],  w,     permutations=999),
        "overdose": Moran(gdf_su_od["overdose_rate"], w,     permutations=999),
        "k70":      Moran(gdf_all["k70_rate"],         w_all, permutations=999),
    }

print("\n" + "="*65)
print("UNIVARIATE MORAN'S I — ERA COMPARISON")
print("="*65)
print(f"{'Axis':<20} {'1999-2005 I':>12} {'2018-2024 I':>12}  {'Change':>8}")
print("-"*65)
for axis in ["suicide", "overdose", "k70"]:
    i_old = mi["1999-2005"][axis].I
    i_now = mi["2018-2024"][axis].I
    print(f"{axis:<20} {i_old:>12.4f} {i_now:>12.4f}  {i_now-i_old:>+8.4f}")

# ── Bivariate Moran's I: both eras ────────────────────────────────────────────

print("\nComputing bivariate Moran's I (999 permutations)...")

bv = {}
for era, gdf_su_od, gdf_all, w, w_all in [
    ("1999-2005", gdf_su_od_old, gdf_all_old, w_old, w_old_all),
    ("2018-2024", gdf_su_od_now, gdf_all_now, w_now, w_now_all),
]:
    bv[era] = {
        "su_od": Moran_BV(gdf_su_od["suicide_rate"],  gdf_su_od["overdose_rate"], w,     permutations=999),
        "od_su": Moran_BV(gdf_su_od["overdose_rate"], gdf_su_od["suicide_rate"],  w,     permutations=999),
        "su_k7": Moran_BV(gdf_all["suicide_rate"],    gdf_all["k70_rate"],        w_all, permutations=999),
        "od_k7": Moran_BV(gdf_all["overdose_rate"],   gdf_all["k70_rate"],        w_all, permutations=999),
    }

print("\n" + "="*80)
print("BIVARIATE MORAN'S I — ERA COMPARISON")
print("="*80)
print(f"{'Pair':<35} {'1999-2005':>10} {'p':>7} {'2018-2024':>10} {'p':>7} {'Change':>8}")
print("-"*80)
pairs = [
    ("su_od", "Suicide → Overdose neighbors"),
    ("od_su", "Overdose → Suicide neighbors"),
    ("su_k7", "Suicide → K70 neighbors"),
    ("od_k7", "Overdose → K70 neighbors"),
]
for key, label in pairs:
    old_I = bv["1999-2005"][key].I
    old_p = bv["1999-2005"][key].p_sim
    now_I = bv["2018-2024"][key].I
    now_p = bv["2018-2024"][key].p_sim
    print(f"{label:<35} {old_I:>10.4f} {old_p:>7.4f} {now_I:>10.4f} {now_p:>7.4f} {now_I-old_I:>+8.4f}")
print("="*80)

# ── Rate comparison: how much did each axis change? ───────────────────────────

print("\n" + "="*55)
print("NATIONAL RATE COMPARISON (mean crude rate per 100K)")
print("="*55)
print(f"{'Axis':<20} {'1999-2005':>12} {'2018-2024':>12} {'change':>8}")
print("-"*55)
for axis, old_gdf, now_gdf, col in [
    ("Suicide",  gdf_su_od_old, gdf_su_od_now, "suicide_rate"),
    ("Overdose", gdf_su_od_old, gdf_su_od_now, "overdose_rate"),
    ("K70",      gdf_all_old,   gdf_all_now,   "k70_rate"),
]:
    old_mean = old_gdf[col].mean()
    now_mean = now_gdf[col].mean()
    print(f"{axis:<20} {old_mean:>12.2f} {now_mean:>12.2f} {now_mean-old_mean:>+8.2f}")

# ── Figure: LISA side-by-side, two eras ───────────────────────────────────────

print("\nGenerating figure...")

CLUSTER_COLORS = {"HH":"#d7191c","LL":"#2c7bb6","LH":"#abd9e9","HL":"#fdae61","NS":"#cccccc"}

def cluster_labels(lisa, sig=0.05):
    labels = np.full(len(lisa.q), "NS", dtype=object)
    quad_map = {1:"HH", 2:"LH", 3:"LL", 4:"HL"}
    for i, q in enumerate(lisa.q):
        if lisa.p_sim[i] < sig:
            labels[i] = quad_map.get(q, "NS")
    return labels

lisa_su_old = Moran_Local(gdf_su_od_old["suicide_rate"],  w_old, permutations=999, seed=42)
lisa_od_old = Moran_Local(gdf_su_od_old["overdose_rate"], w_old, permutations=999, seed=42)
lisa_su_now = Moran_Local(gdf_su_od_now["suicide_rate"],  w_now, permutations=999, seed=42)
lisa_od_now = Moran_Local(gdf_su_od_now["overdose_rate"], w_now, permutations=999, seed=42)

gdf_su_od_old["lisa_su"] = cluster_labels(lisa_su_old)
gdf_su_od_old["lisa_od"] = cluster_labels(lisa_od_old)
gdf_su_od_now["lisa_su"] = cluster_labels(lisa_su_now)
gdf_su_od_now["lisa_od"] = cluster_labels(lisa_od_now)

CONUS_BOUNDS = (-2.4e6, 150000, 2.35e6, 3.22e6)  # correct EPSG:5070 bounds
def set_extent(ax):
    ax.set_xlim(CONUS_BOUNDS[0], CONUS_BOUNDS[2])
    ax.set_ylim(CONUS_BOUNDS[1], CONUS_BOUNDS[3])
    ax.set_aspect("equal")
    ax.axis("off")

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.patch.set_facecolor("#f8f8f6")

configs = [
    (axes[0,0], gdf_su_od_old, "lisa_su", "Suicide 1999–2005"),
    (axes[0,1], gdf_su_od_old, "lisa_od", "Overdose 1999–2005"),
    (axes[1,0], gdf_su_od_now, "lisa_su", "Suicide 2018–2024"),
    (axes[1,1], gdf_su_od_now, "lisa_od", "Overdose 2018–2024"),
]

for ax, gd, lcol, title in configs:
    counties[~counties["fips"].isin(gd["fips"])].plot(
        ax=ax, color="#e8e8e8", linewidth=0.1, edgecolor="white"
    )
    gd["_color"] = gd[lcol].map(CLUSTER_COLORS)
    gd.plot(ax=ax, color=gd["_color"], linewidth=0.1, edgecolor="white")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
    set_extent(ax)

legend_patches = [
    mpatches.Patch(color="#d7191c", label="HH (hot spot)"),
    mpatches.Patch(color="#2c7bb6", label="LL (cold spot)"),
    mpatches.Patch(color="#fdae61", label="HL"), mpatches.Patch(color="#abd9e9", label="LH"),
    mpatches.Patch(color="#cccccc", label="Not significant"),
]
axes[1,0].legend(handles=legend_patches, loc="lower left", fontsize=8, framealpha=0.9)

plt.suptitle("LISA Clusters: Suicide vs. Overdose, 1999–2005 vs. 2018–2024\n"
             "Was the geographic split always there, or did it emerge with the opioid epidemic?",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("fig_temporal.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig_temporal.png")
print("\nDone.")
