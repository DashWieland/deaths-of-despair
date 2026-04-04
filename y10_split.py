"""
Y10-Y14 split analysis.

Tests whether "undetermined intent" drug poisoning (Y10-Y14) has a different
geographic pattern than clean unintentional overdose (X40-X44).

If Y10-Y14 tracks with suicide geography (BV Moran's I > 0 with suicide),
it suggests coroners are misclassifying suicides as undetermined — and our
main overdose axis is contaminated. If Y10-Y14 tracks with X40-X44, they
share a supply-chain geography and lumping them was correct.
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

suicide      = load_wonder("suicide_county_2018_2024.csv",           "suicide_rate")
unintent     = load_wonder("overdose_unintentional_county_2018_2024.csv", "unintent_rate")
undetermined = load_wonder("overdose_undetermined_county_2018_2024.csv",  "undeter_rate")
overdose     = load_wonder("overdose_county_2018_2024.csv",          "overdose_rate")  # combined, for reference

print("Row counts (counties with data above suppression threshold):")
print(f"  Suicide (X60-X84):           {len(suicide)}")
print(f"  Unintentional (X40-X44):     {len(unintent)}")
print(f"  Undetermined (Y10-Y14):      {len(undetermined)}")
print(f"  Combined overdose (X40+Y10): {len(overdose)}")

df = (suicide
      .merge(unintent,     on="fips", how="outer")
      .merge(undetermined, on="fips", how="outer")
      .merge(overdose,     on="fips", how="outer"))

# What fraction of combined overdose rate is Y10-Y14?
df["y_fraction"] = df["undeter_rate"] / df["overdose_rate"]
valid_frac = df["y_fraction"].dropna()
print(f"\nY10-Y14 as fraction of combined overdose rate:")
print(f"  Counties with both:  {len(valid_frac)}")
print(f"  Mean fraction:       {valid_frac.mean():.3f}")
print(f"  Median fraction:     {valid_frac.median():.3f}")
print(f"  Counties where Y10-Y14 > 30% of total: {(valid_frac > 0.3).sum()}")

# ── Spatial setup ──────────────────────────────────────────────────────────────

print("\nLoading shapefile...")
counties = gpd.read_file(
    "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
)
counties = counties.rename(columns={"GEOID": "fips"})
EXCLUDE = {"02", "15", "60", "66", "69", "72", "78"}
counties = counties[~counties["STATEFP"].isin(EXCLUDE)].copy()
counties = counties.to_crs("EPSG:5070")

# Working set: counties with both suicide and unintentional overdose
gdf = counties.merge(df, on="fips", how="left")
gdf_su_ui = gdf[gdf["suicide_rate"].notna() & gdf["unintent_rate"].notna()].copy().reset_index(drop=True)

# Smaller set: all four axes present
gdf_all = gdf[gdf["suicide_rate"].notna() & gdf["unintent_rate"].notna() &
              gdf["undeter_rate"].notna()].copy().reset_index(drop=True)

print(f"\nSpatial subsets:")
print(f"  Suicide + unintentional: {len(gdf_su_ui)} counties")
print(f"  All four axes:           {len(gdf_all)} counties (Y10-Y14 suppressed elsewhere)")

w_su_ui = libpysal.weights.Queen.from_dataframe(gdf_su_ui, silence_warnings=True)
w_su_ui.transform = "r"

w_all = libpysal.weights.Queen.from_dataframe(gdf_all, silence_warnings=True)
w_all.transform = "r"

# ── Univariate Moran's I ───────────────────────────────────────────────────────

print("\nComputing Moran's I (999 permutations)...")
mi_su  = Moran(gdf_su_ui["suicide_rate"],  w_su_ui, permutations=999)
mi_ui  = Moran(gdf_su_ui["unintent_rate"], w_su_ui, permutations=999)
mi_ud  = Moran(gdf_all["undeter_rate"],    w_all,   permutations=999)

print("\n" + "="*60)
print("UNIVARIATE MORAN'S I")
print("="*60)
print(f"{'Axis':<35} {'I':>8} {'p':>8}")
print("-"*60)
print(f"{'Suicide (X60-X84)':<35} {mi_su.I:>8.4f} {mi_su.p_sim:>8.4f}")
print(f"{'Unintentional OD (X40-X44)':<35} {mi_ui.I:>8.4f} {mi_ui.p_sim:>8.4f}")
print(f"{'Undetermined (Y10-Y14)':<35} {mi_ud.I:>8.4f} {mi_ud.p_sim:>8.4f}")

# ── Bivariate Moran's I ────────────────────────────────────────────────────────

print("\nComputing bivariate Moran's I (999 permutations)...")

# On the larger set: suicide vs unintentional OD
bv_su_ui  = Moran_BV(gdf_su_ui["suicide_rate"],  gdf_su_ui["unintent_rate"], w_su_ui, permutations=999)
bv_ui_su  = Moran_BV(gdf_su_ui["unintent_rate"], gdf_su_ui["suicide_rate"],  w_su_ui, permutations=999)

# On the smaller set: all cross-pairs with Y10-Y14
bv_su_ud  = Moran_BV(gdf_all["suicide_rate"],  gdf_all["undeter_rate"],  w_all, permutations=999)
bv_ud_su  = Moran_BV(gdf_all["undeter_rate"],  gdf_all["suicide_rate"],  w_all, permutations=999)
bv_ui_ud  = Moran_BV(gdf_all["unintent_rate"], gdf_all["undeter_rate"],  w_all, permutations=999)
bv_ud_ui  = Moran_BV(gdf_all["undeter_rate"],  gdf_all["unintent_rate"], w_all, permutations=999)

print("\n" + "="*65)
print("BIVARIATE MORAN'S I")
print("="*65)
print(f"{'Pair':<40} {'BV I':>8} {'p':>8}")
print("-"*65)
print(f"{'Suicide → Unintentional OD neighbors':<40} {bv_su_ui.I:>8.4f} {bv_su_ui.p_sim:>8.4f}")
print(f"{'Unintentional OD → Suicide neighbors':<40} {bv_ui_su.I:>8.4f} {bv_ui_su.p_sim:>8.4f}")
print(f"{'Suicide → Undetermined neighbors':<40} {bv_su_ud.I:>8.4f} {bv_su_ud.p_sim:>8.4f}")
print(f"{'Undetermined → Suicide neighbors':<40} {bv_ud_su.I:>8.4f} {bv_ud_su.p_sim:>8.4f}")
print(f"{'Unintentional → Undetermined neighbors':<40} {bv_ui_ud.I:>8.4f} {bv_ui_ud.p_sim:>8.4f}")
print(f"{'Undetermined → Unintentional neighbors':<40} {bv_ud_ui.I:>8.4f} {bv_ud_ui.p_sim:>8.4f}")
print("="*65)

# ── Figure: LISA maps ──────────────────────────────────────────────────────────

print("\nGenerating figure...")

lisa_su = Moran_Local(gdf_su_ui["suicide_rate"],  w_su_ui, permutations=999, seed=42)
lisa_ui = Moran_Local(gdf_su_ui["unintent_rate"], w_su_ui, permutations=999, seed=42)
lisa_ud = Moran_Local(gdf_all["undeter_rate"],    w_all,   permutations=999, seed=42)

CLUSTER_COLORS = {"HH": "#d7191c", "LL": "#2c7bb6", "LH": "#abd9e9", "HL": "#fdae61", "NS": "#cccccc"}

def cluster_labels(lisa, sig=0.05):
    labels = np.full(len(lisa.q), "NS", dtype=object)
    quad_map = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    for i, q in enumerate(lisa.q):
        if lisa.p_sim[i] < sig:
            labels[i] = quad_map.get(q, "NS")
    return labels

gdf_su_ui = gdf_su_ui.copy()
gdf_su_ui["lisa_su"] = cluster_labels(lisa_su)
gdf_su_ui["lisa_ui"] = cluster_labels(lisa_ui)
gdf_all = gdf_all.copy()
gdf_all["lisa_ud"] = cluster_labels(lisa_ud)

CONUS_BOUNDS = (-2.4e6, 150000, 2.35e6, 3.22e6)  # correct EPSG:5070 bounds
def set_extent(ax):
    ax.set_xlim(CONUS_BOUNDS[0], CONUS_BOUNDS[2])
    ax.set_ylim(CONUS_BOUNDS[1], CONUS_BOUNDS[3])
    ax.set_aspect("equal")
    ax.axis("off")

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.patch.set_facecolor("#f8f8f6")

lisa_configs = [
    (gdf_su_ui, "lisa_su", "Suicide (X60–X84)"),
    (gdf_su_ui, "lisa_ui", "Unintentional OD (X40–X44)"),
    (gdf_all,   "lisa_ud", "Undetermined (Y10–Y14)"),
]

for ax, (gd, lcol, title) in zip(axes, lisa_configs):
    counties[~counties["fips"].isin(gd["fips"])].plot(
        ax=ax, color="#e8e8e8", linewidth=0.1, edgecolor="white"
    )
    gd["_color"] = gd[lcol].map(CLUSTER_COLORS)
    gd.plot(ax=ax, color=gd["_color"], linewidth=0.1, edgecolor="white")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    set_extent(ax)

legend_patches = [
    mpatches.Patch(color="#d7191c", label="HH (hot spot)"),
    mpatches.Patch(color="#2c7bb6", label="LL (cold spot)"),
    mpatches.Patch(color="#fdae61", label="HL"),
    mpatches.Patch(color="#abd9e9", label="LH"),
    mpatches.Patch(color="#cccccc", label="Not significant / no data"),
]
axes[0].legend(handles=legend_patches, loc="lower left", fontsize=8, framealpha=0.9)

plt.suptitle("LISA Clusters: Suicide vs. Overdose Sub-components (p < 0.05)\n"
             "Key question: does Y10–Y14 look like X40–X44 or like X60–X84?",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("fig_y10_split.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig_y10_split.png")
print("\nDone.")
