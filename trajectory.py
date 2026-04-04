"""
Trajectory of suicide-overdose spatial co-clustering, 1999-2024.

Computes bivariate Moran's I between suicide and overdose rates for
four non-overlapping 7-year windows, then plots the collapse.

Windows:
  1999-2005  pre-epidemic
  2006-2012  OxyContin era peak
  2013-2019  prescription crackdown → heroin → fentanyl
  2018-2024  fentanyl era (overlaps 2013-2019 by 1yr; separate WONDER DB)

Case & Deaton's landmark paper (2015) drew primarily on 1999-2013 data.
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

ERAS = [
    ("1999–2005", "suicide_county_1999_2005.csv",      "overdose_county_1999_2005.csv",      "alcohol_liver_county_1999_2005.csv"),
    ("2006–2012", "suicide_county_2006_2012.csv",      "overdose_county_2006_2012.csv",      "alcohol_liver_county_2006_2012.csv"),
    ("2013–2019", "suicide_county_2013_2019.csv",      "overdose_county_2013_2019.csv",      "alcohol_liver_county_2013_2019.csv"),
    ("2018–2024", "suicide_county_2018_2024.csv",      "overdose_county_2018_2024.csv",      "alcohol_liver_county_2018_2024.csv"),
]

# ── Spatial setup (load once) ──────────────────────────────────────────────────

print("Loading shapefile...")
counties = gpd.read_file(
    "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
)
counties = counties.rename(columns={"GEOID": "fips"})
EXCLUDE = {"02", "15", "60", "66", "69", "72", "78"}
counties = counties[~counties["STATEFP"].isin(EXCLUDE)].copy()
counties = counties.to_crs("EPSG:5070")

# ── Compute Moran statistics for each era ─────────────────────────────────────

results = []

for era, su_f, od_f, k7_f in ERAS:
    print(f"\n{era}...")
    su = load_wonder(su_f, "suicide_rate")
    od = load_wonder(od_f, "overdose_rate")
    k7 = load_wonder(k7_f, "k70_rate")

    merged = su.merge(od, on="fips", how="outer").merge(k7, on="fips", how="outer")
    gdf = counties.merge(merged, on="fips", how="left")

    gdf_su_od = gdf[gdf["suicide_rate"].notna() & gdf["overdose_rate"].notna()].copy().reset_index(drop=True)
    gdf_all   = gdf[gdf["suicide_rate"].notna() & gdf["overdose_rate"].notna() & gdf["k70_rate"].notna()].copy().reset_index(drop=True)

    w      = libpysal.weights.Queen.from_dataframe(gdf_su_od, silence_warnings=True); w.transform = "r"
    w_all  = libpysal.weights.Queen.from_dataframe(gdf_all,   silence_warnings=True); w_all.transform = "r"

    mi_su = Moran(gdf_su_od["suicide_rate"],  w,     permutations=999)
    mi_od = Moran(gdf_su_od["overdose_rate"], w,     permutations=999)
    mi_k7 = Moran(gdf_all["k70_rate"],        w_all, permutations=999)

    bv_su_od = Moran_BV(gdf_su_od["suicide_rate"],  gdf_su_od["overdose_rate"], w,     permutations=999)
    bv_od_su = Moran_BV(gdf_su_od["overdose_rate"], gdf_su_od["suicide_rate"],  w,     permutations=999)
    bv_su_k7 = Moran_BV(gdf_all["suicide_rate"],    gdf_all["k70_rate"],        w_all, permutations=999)
    bv_od_k7 = Moran_BV(gdf_all["overdose_rate"],   gdf_all["k70_rate"],        w_all, permutations=999)

    mean_su = gdf_su_od["suicide_rate"].mean()
    mean_od = gdf_su_od["overdose_rate"].mean()
    mean_k7 = gdf_all["k70_rate"].mean()

    results.append({
        "era":      era,
        "n_su_od":  len(gdf_su_od),
        "mi_su":    mi_su.I,  "p_su": mi_su.p_sim,
        "mi_od":    mi_od.I,  "p_od": mi_od.p_sim,
        "mi_k7":    mi_k7.I,  "p_k7": mi_k7.p_sim,
        "bv_su_od": bv_su_od.I, "p_bv_su_od": bv_su_od.p_sim,
        "bv_od_su": bv_od_su.I, "p_bv_od_su": bv_od_su.p_sim,
        "bv_su_k7": bv_su_k7.I, "p_bv_su_k7": bv_su_k7.p_sim,
        "bv_od_k7": bv_od_k7.I, "p_bv_od_k7": bv_od_k7.p_sim,
        "mean_su":  mean_su,
        "mean_od":  mean_od,
        "mean_k7":  mean_k7,
        "gdf_su_od": gdf_su_od,
        "w":         w,
    })
    print(f"  BV I(suicide,overdose) = {bv_su_od.I:.4f}  p={bv_su_od.p_sim:.4f}")
    print(f"  BV I(suicide,K70)      = {bv_su_k7.I:.4f}  p={bv_su_k7.p_sim:.4f}")

# ── Summary table ──────────────────────────────────────────────────────────────

print("\n" + "="*90)
print("TRAJECTORY SUMMARY")
print("="*90)
print(f"{'Era':<12} {'BV I(su,od)':>12} {'p':>7} {'BV I(su,k7)':>12} {'p':>7} "
      f"{'mean OD':>9} {'mean SU':>9}")
print("-"*90)
for r in results:
    sig_od = "***" if r["p_bv_su_od"] < 0.001 else ("**" if r["p_bv_su_od"] < 0.01 else
             ("*" if r["p_bv_su_od"] < 0.05 else "ns"))
    sig_k7 = "***" if r["p_bv_su_k7"] < 0.001 else ("**" if r["p_bv_su_k7"] < 0.01 else
             ("*" if r["p_bv_su_k7"] < 0.05 else "ns"))
    print(f"{r['era']:<12} {r['bv_su_od']:>12.4f} {sig_od:>7} {r['bv_su_k7']:>12.4f} {sig_k7:>7} "
          f"{r['mean_od']:>9.2f} {r['mean_su']:>9.2f}")

# ── Figure 1: BV I trajectory ──────────────────────────────────────────────────

print("\nGenerating figures...")

# X positions: midpoint year of each window
x_pos    = [2002, 2009, 2016, 2021]
x_labels = ["1999–2005\n(pre-epidemic)", "2006–2012\n(OxyContin peak)",
            "2013–2019\n(fentanyl transition)", "2018–2024\n(fentanyl era)"]

bvi_su_od = [r["bv_su_od"] for r in results]
bvi_su_k7 = [r["bv_su_k7"] for r in results]
p_su_od   = [r["p_bv_su_od"] for r in results]

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#f8f8f6")
ax.set_facecolor("#f8f8f6")

ax.plot(x_pos, bvi_su_od, "o-", color="#e84855", linewidth=2.5,
        markersize=9, label="BV I: Suicide × Overdose", zorder=3)
ax.plot(x_pos, bvi_su_k7, "s--", color="#4d9de0", linewidth=2,
        markersize=8, label="BV I: Suicide × K70", zorder=3)

# Significance markers
for x, y, p in zip(x_pos, bvi_su_od, p_su_od):
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    ax.annotate(sig, (x, y), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=11, color="#e84855")

ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_ylabel("Bivariate Moran's I", fontsize=11)
ax.set_ylim(-0.05, 0.45)
ax.legend(fontsize=10, loc="upper right")
ax.grid(axis="y", alpha=0.3)

# Annotate C&D publication
ax.axvspan(1999, 2013, alpha=0.07, color="#e84855", label="C&D data window")
ax.text(2006, 0.41, "Case & Deaton\ndata window\n(1999–2013)", fontsize=8,
        color="#c0392b", ha="center", style="italic")

ax.set_title("Collapse of Suicide–Overdose Spatial Co-clustering, 1999–2024\n"
             "Bivariate Moran's I: how much do high-suicide areas predict high-overdose neighbors?",
             fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("fig_trajectory.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig_trajectory.png")

# ── Figure 2: LISA comparison — overdose clusters across eras ─────────────────

CLUSTER_COLORS = {"HH":"#d7191c","LL":"#2c7bb6","LH":"#abd9e9","HL":"#fdae61","NS":"#cccccc"}

def cluster_labels(lisa, sig=0.05):
    labels = np.full(len(lisa.q), "NS", dtype=object)
    quad_map = {1:"HH", 2:"LH", 3:"LL", 4:"HL"}
    for i, q in enumerate(lisa.q):
        if lisa.p_sim[i] < sig:
            labels[i] = quad_map.get(q, "NS")
    return labels

CONUS_BOUNDS = (-2.4e6, 150000, 2.35e6, 3.22e6)  # correct EPSG:5070 bounds
def set_extent(ax):
    ax.set_xlim(CONUS_BOUNDS[0], CONUS_BOUNDS[2])
    ax.set_ylim(CONUS_BOUNDS[1], CONUS_BOUNDS[3])
    ax.set_aspect("equal"); ax.axis("off")

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.patch.set_facecolor("#f8f8f6")
axes_flat = axes.flatten()

for ax, r in zip(axes_flat, results):
    lisa_od = Moran_Local(r["gdf_su_od"]["overdose_rate"], r["w"], permutations=999, seed=42)
    gd = r["gdf_su_od"].copy()
    gd["_color"] = cluster_labels(lisa_od)
    gd["_color"] = gd["_color"].map(CLUSTER_COLORS)

    counties[~counties["fips"].isin(gd["fips"])].plot(
        ax=ax, color="#e8e8e8", linewidth=0.1, edgecolor="white")
    gd.plot(ax=ax, color=gd["_color"], linewidth=0.1, edgecolor="white")

    bvi = r["bv_su_od"]
    p   = r["p_bv_su_od"]
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    ax.set_title(f"Overdose LISA — {r['era']}\nBV I(suicide,overdose)={bvi:.3f}{sig}  "
                 f"mean rate={r['mean_od']:.1f}",
                 fontsize=10, fontweight="bold", pad=6)
    set_extent(ax)

legend_patches = [
    mpatches.Patch(color="#d7191c", label="HH (overdose hot spot)"),
    mpatches.Patch(color="#2c7bb6", label="LL (cold spot)"),
    mpatches.Patch(color="#fdae61", label="HL"), mpatches.Patch(color="#abd9e9", label="LH"),
    mpatches.Patch(color="#cccccc", label="Not significant"),
]
axes_flat[2].legend(handles=legend_patches, loc="lower left", fontsize=8, framealpha=0.9)

plt.suptitle("Overdose LISA Clusters Across Four Eras\n"
             "Watch Appalachia and New England consolidate as Mountain West fades",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("fig_trajectory_lisa.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig_trajectory_lisa.png")
print("\nDone.")
