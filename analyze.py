import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
import libpysal
from esda import Moran, Moran_BV, Moran_Local

# ── 1. Load and clean WONDER CSVs ──────────────────────────────────────────────

def load_wonder(path, rate_col="suicide_rate"):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip().str.strip('"')
    # Keep only rows with valid 5-digit FIPS
    df = df[df["County Code"].str.match(r"^\d{5}$", na=False)].copy()
    df = df.rename(columns={"County Code": "fips", "County": "county_name"})
    df["deaths"] = pd.to_numeric(df["Deaths"], errors="coerce")
    df["population"] = pd.to_numeric(df["Population"], errors="coerce")
    df["crude_rate"] = pd.to_numeric(df["Crude Rate"], errors="coerce")
    df = df[["fips", "county_name", "deaths", "population", "crude_rate"]].copy()
    df = df.rename(columns={"crude_rate": rate_col, "deaths": f"deaths_{rate_col}", "population": f"pop_{rate_col}"})
    print(f"  {path.split('/')[-1]}: {len(df)} counties with data")
    return df

print("Loading data...")
suicide  = load_wonder("suicide_county_2018_2024.csv",      "suicide_rate")
overdose = load_wonder("overdose_county_2018_2024.csv",     "overdose_rate")
k70      = load_wonder("alcohol_liver_county_2018_2024.csv","k70_rate")

# ── 2. Merge ───────────────────────────────────────────────────────────────────

merged = (
    suicide[["fips", "county_name", "suicide_rate"]]
    .merge(overdose[["fips", "overdose_rate"]], on="fips", how="outer")
    .merge(k70[["fips", "k70_rate"]],          on="fips", how="outer")
)

n_all   = merged[["suicide_rate","overdose_rate","k70_rate"]].dropna().shape[0]
n_su_od = merged[["suicide_rate","overdose_rate"]].dropna().shape[0]
n_k70   = merged[["k70_rate"]].dropna().shape[0]
print(f"\nMerge coverage:")
print(f"  All three axes:       {n_all} counties")
print(f"  Suicide + overdose:   {n_su_od} counties")
print(f"  K70 only:             {n_k70} counties")

# ── 3. Spatial weights ─────────────────────────────────────────────────────────

print("\nLoading county shapefile...")
counties = gpd.read_file(
    "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
)
counties = counties.rename(columns={"GEOID": "fips"})

# Contiguous 48 states only
EXCLUDE = {"02", "15", "60", "66", "69", "72", "78"}
counties = counties[~counties["STATEFP"].isin(EXCLUDE)].copy()
counties = counties.to_crs("EPSG:5070")  # Albers Equal Area

# Join death data onto geometry
gdf = counties.merge(merged, on="fips", how="left")

# ── Full dataset (suicide + overdose) ──
gdf_su_od = gdf[gdf["suicide_rate"].notna() & gdf["overdose_rate"].notna()].copy().reset_index(drop=True)

# ── K70 subset ──
gdf_k70 = gdf[gdf["k70_rate"].notna() & gdf["suicide_rate"].notna() & gdf["overdose_rate"].notna()].copy().reset_index(drop=True)

print(f"\nSpatial subsets:")
print(f"  Suicide + overdose: {len(gdf_su_od)} counties")
print(f"  All three axes:     {len(gdf_k70)} counties")

print("Building spatial weights...")
w_su_od = libpysal.weights.Queen.from_dataframe(gdf_su_od, silence_warnings=True)
w_su_od.transform = "r"
w_k70 = libpysal.weights.Queen.from_dataframe(gdf_k70, silence_warnings=True)
w_k70.transform = "r"

# ── 4. Univariate Moran's I ────────────────────────────────────────────────────

print("\nComputing univariate Moran's I (999 permutations)...")
mi_su  = Moran(gdf_su_od["suicide_rate"],  w_su_od, permutations=999)
mi_od  = Moran(gdf_su_od["overdose_rate"], w_su_od, permutations=999)
mi_k70 = Moran(gdf_k70["k70_rate"],        w_k70,   permutations=999)

# ── 5. Bivariate Moran's I ─────────────────────────────────────────────────────

print("Computing bivariate Moran's I (999 permutations each)...")
bv_su_od = Moran_BV(gdf_su_od["suicide_rate"],  gdf_su_od["overdose_rate"], w_su_od, permutations=999)
bv_od_su = Moran_BV(gdf_su_od["overdose_rate"], gdf_su_od["suicide_rate"],  w_su_od, permutations=999)
bv_su_k7 = Moran_BV(gdf_k70["suicide_rate"],    gdf_k70["k70_rate"],        w_k70,   permutations=999)
bv_k7_su = Moran_BV(gdf_k70["k70_rate"],        gdf_k70["suicide_rate"],    w_k70,   permutations=999)
bv_od_k7 = Moran_BV(gdf_k70["overdose_rate"],   gdf_k70["k70_rate"],        w_k70,   permutations=999)
bv_k7_od = Moran_BV(gdf_k70["k70_rate"],        gdf_k70["overdose_rate"],   w_k70,   permutations=999)

# ── 6. LISA clusters ───────────────────────────────────────────────────────────

print("Computing LISA clusters...")
lisa_su = Moran_Local(gdf_su_od["suicide_rate"],  w_su_od, permutations=999, seed=42)
lisa_od = Moran_Local(gdf_su_od["overdose_rate"], w_su_od, permutations=999, seed=42)
lisa_k7 = Moran_Local(gdf_k70["k70_rate"],        w_k70,   permutations=999, seed=42)

def cluster_labels(lisa, sig=0.05):
    sig_mask = lisa.p_sim < sig
    labels = np.full(len(lisa.q), "NS", dtype=object)
    quad_map = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    for i, q in enumerate(lisa.q):
        if sig_mask[i]:
            labels[i] = quad_map.get(q, "NS")
    return labels

gdf_su_od = gdf_su_od.copy()
gdf_su_od["lisa_su"] = cluster_labels(lisa_su)
gdf_su_od["lisa_od"] = cluster_labels(lisa_od)

gdf_k70 = gdf_k70.copy()
gdf_k70["lisa_k7"] = cluster_labels(lisa_k7)

# ── Print summary ──────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("UNIVARIATE MORAN'S I")
print("="*65)
print(f"{'Axis':<20} {'I':>8} {'p':>8} {'HH':>6} {'LL':>6}")
print("-"*65)

def hh_ll(labels):
    return (labels == "HH").sum(), (labels == "LL").sum()

hh_su, ll_su = hh_ll(gdf_su_od["lisa_su"])
hh_od, ll_od = hh_ll(gdf_su_od["lisa_od"])
hh_k7, ll_k7 = hh_ll(gdf_k70["lisa_k7"])

print(f"{'Suicide':<20} {mi_su.I:>8.4f} {mi_su.p_sim:>8.4f} {hh_su:>6} {ll_su:>6}")
print(f"{'Overdose':<20} {mi_od.I:>8.4f} {mi_od.p_sim:>8.4f} {hh_od:>6} {ll_od:>6}")
print(f"{'K70 (liver)':<20} {mi_k70.I:>8.4f} {mi_k70.p_sim:>8.4f} {hh_k7:>6} {ll_k7:>6}")

print("\n" + "="*65)
print("BIVARIATE MORAN'S I  (x → neighbors' y)")
print("="*65)
print(f"{'Pair':<35} {'BV I':>8} {'p':>8}")
print("-"*65)
print(f"{'Suicide → Overdose neighbors':<35} {bv_su_od.I:>8.4f} {bv_su_od.p_sim:>8.4f}")
print(f"{'Overdose → Suicide neighbors':<35} {bv_od_su.I:>8.4f} {bv_od_su.p_sim:>8.4f}")
print(f"{'Suicide → K70 neighbors':<35} {bv_su_k7.I:>8.4f} {bv_su_k7.p_sim:>8.4f}")
print(f"{'K70 → Suicide neighbors':<35} {bv_k7_su.I:>8.4f} {bv_k7_su.p_sim:>8.4f}")
print(f"{'Overdose → K70 neighbors':<35} {bv_od_k7.I:>8.4f} {bv_od_k7.p_sim:>8.4f}")
print(f"{'K70 → Overdose neighbors':<35} {bv_k7_od.I:>8.4f} {bv_k7_od.p_sim:>8.4f}")
print("="*65)

# ── 7. Figures ─────────────────────────────────────────────────────────────────

CONUS_BOUNDS = (-2.4e6, 150000, 2.35e6, 3.22e6)  # correct EPSG:5070 bounds

def set_conus_extent(ax):
    ax.set_xlim(CONUS_BOUNDS[0], CONUS_BOUNDS[2])
    ax.set_ylim(CONUS_BOUNDS[1], CONUS_BOUNDS[3])
    ax.set_aspect("equal")
    ax.axis("off")

# ── Figure 1: Choropleth triptych ──
print("\nGenerating figures...")
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.patch.set_facecolor("#f8f8f6")

configs = [
    (gdf_su_od, "suicide_rate",  "Suicide (X60–X84)",         "#4d9de0"),
    (gdf_su_od, "overdose_rate", "Drug Overdose (X40–X44, Y10–Y14)", "#e84855"),
    (gdf_k70,   "k70_rate",      "Alcoholic Liver Disease (K70)",    "#a44a3f"),
]

for ax, (gd, col, title, color) in zip(axes, configs):
    # Background (all CONUS counties as light grey)
    counties[~counties["fips"].isin(gd["fips"])].plot(
        ax=ax, color="#e8e8e8", linewidth=0.1, edgecolor="white"
    )
    gd.plot(
        column=col, ax=ax, scheme="quantiles", k=5,
        cmap="YlOrRd", linewidth=0.1, edgecolor="white", legend=False, missing_kwds={"color": "#cccccc"}
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    set_conus_extent(ax)

plt.suptitle("Deaths of Despair: Three Axes, 2018–2024\n(Crude rate per 100K, county-level, 5-quantile classification)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("fig_choropleths.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig_choropleths.png")

# ── Figure 2: Bivariate scatter ──
CLUSTER_COLORS = {"HH": "#d7191c", "LL": "#2c7bb6", "LH": "#abd9e9", "HL": "#fdae61", "NS": "#cccccc"}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor("#f8f8f6")

scatter_configs = [
    ("suicide_rate", "overdose_rate", "lisa_su", gdf_su_od,
     "Suicide rate (per 100K)", "Overdose rate (per 100K)", bv_su_od.I, bv_su_od.p_sim),
    ("suicide_rate", "k70_rate", "lisa_k7", gdf_k70,
     "Suicide rate (per 100K)", "K70 rate (per 100K)", bv_su_k7.I, bv_su_k7.p_sim),
    ("overdose_rate", "k70_rate", "lisa_k7", gdf_k70,
     "Overdose rate (per 100K)", "K70 rate (per 100K)", bv_od_k7.I, bv_od_k7.p_sim),
]

for ax, (xcol, ycol, lcol, gd, xlabel, ylabel, bvi, bvp) in zip(axes, scatter_configs):
    gd_plot = gd[[xcol, ycol, lcol]].dropna()
    colors = [CLUSTER_COLORS[c] for c in gd_plot[lcol]]
    ax.scatter(gd_plot[xcol], gd_plot[ycol], c=colors, s=4, alpha=0.6, linewidths=0)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    sig = "***" if bvp < 0.001 else ("**" if bvp < 0.01 else ("*" if bvp < 0.05 else "ns"))
    ax.set_title(f"Bivariate Moran's I = {bvi:.4f}{sig}", fontsize=11)
    ax.set_facecolor("#f8f8f6")

legend_patches = [mpatches.Patch(color=v, label=k) for k, v in CLUSTER_COLORS.items() if k != "NS"]
axes[0].legend(handles=legend_patches, title="LISA (suicide)", fontsize=8, title_fontsize=8)
plt.suptitle("Spatial Cross-Correlation Between Axes\n(dot color = suicide LISA cluster of that county)",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("fig_scatter.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig_scatter.png")

# ── Figure 3: LISA cluster maps ──
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.patch.set_facecolor("#f8f8f6")

lisa_configs = [
    (gdf_su_od, "lisa_su", "Suicide LISA Clusters"),
    (gdf_su_od, "lisa_od", "Overdose LISA Clusters"),
    (gdf_k70,   "lisa_k7", "K70 LISA Clusters"),
]

for ax, (gd, lcol, title) in zip(axes, lisa_configs):
    counties[~counties["fips"].isin(gd["fips"])].plot(
        ax=ax, color="#e8e8e8", linewidth=0.1, edgecolor="white"
    )
    gd["_color"] = gd[lcol].map(CLUSTER_COLORS)
    gd.plot(ax=ax, color=gd["_color"], linewidth=0.1, edgecolor="white")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    set_conus_extent(ax)

legend_patches = [
    mpatches.Patch(color="#d7191c", label="HH (hot spot)"),
    mpatches.Patch(color="#2c7bb6", label="LL (cold spot)"),
    mpatches.Patch(color="#fdae61", label="HL (high-low outlier)"),
    mpatches.Patch(color="#abd9e9", label="LH (low-high outlier)"),
    mpatches.Patch(color="#cccccc", label="Not significant"),
]
axes[0].legend(handles=legend_patches, loc="lower left", fontsize=8, framealpha=0.9)
plt.suptitle("LISA Spatial Cluster Maps (p < 0.05, Queen contiguity, 999 permutations)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("fig_lisa.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig_lisa.png")

print("\nDone.")
