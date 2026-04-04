"""
Publication-quality figures for the deaths of despair essay.

Three figures:
  1. Bivariate choropleth (suicide × overdose) — the visual argument
     Shows geographic divergence: Mountain West suicide vs. Appalachia overdose
  2. BV I trajectory — polished version of fig_temporal.png
  3. Regime classification map — A/B/C/D county geography
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import libpysal
from esda import Moran_BV, Moran_Local

# ── Shared helpers ─────────────────────────────────────────────────────────────

def load_wonder(path, rate_col):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip().str.strip('"')
    df = df[df["County Code"].str.match(r"^\d{5}$", na=False)].copy()
    df = df.rename(columns={"County Code": "fips"})
    df["rate"] = pd.to_numeric(df["Crude Rate"], errors="coerce")
    return df[["fips", "rate"]].rename(columns={"rate": rate_col})

def load_counties():
    counties = gpd.read_file(
        "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
    )
    counties = counties.rename(columns={"GEOID": "fips"})
    EXCLUDE = {"02", "15", "60", "66", "69", "72", "78"}
    return counties[~counties["STATEFP"].isin(EXCLUDE)].to_crs("EPSG:5070")

CONUS_BOUNDS = (-2.4e6, 150000, 2.35e6, 3.22e6)  # correct EPSG:5070 bounds
BG = "#f4f1ec"

def set_extent(ax):
    ax.set_xlim(CONUS_BOUNDS[0], CONUS_BOUNDS[2])
    ax.set_ylim(CONUS_BOUNDS[1], CONUS_BOUNDS[3])
    ax.set_aspect("equal")
    ax.axis("off")

# ── Load data ──────────────────────────────────────────────────────────────────

print("Loading data...")
suicide  = load_wonder("suicide_county_2018_2024.csv",      "suicide_rate")
overdose = load_wonder("overdose_county_2018_2024.csv",     "overdose_rate")
k70      = load_wonder("alcohol_liver_county_2018_2024.csv","k70_rate")

counties = load_counties()

merged = (suicide
          .merge(overdose, on="fips", how="outer")
          .merge(k70,      on="fips", how="outer"))
gdf = counties.merge(merged, on="fips", how="left")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Bivariate choropleth — suicide × overdose
# ═══════════════════════════════════════════════════════════════════════════════
#
# Steven's 3×3 bivariate scheme:
#   x-axis = overdose class (1=low → 3=high)  → teal hue
#   y-axis = suicide class  (1=low → 3=high)  → purple hue
#   both high = dark blue (rare; the "despair" zone)
#
# Key visual: Mountain West counties appear purple (high suicide, low overdose)
#             Appalachia/New England appear teal (high overdose, low suicide)
# ───────────────────────────────────────────────────────────────────────────────

print("\nFigure 1: Bivariate choropleth...")

# Steven's bivariate palette
# Rows: od class 1,2,3  Cols: su class 1,2,3
BIVAR = {
    (1, 1): "#e8e8e8",   # low od, low su — pale grey
    (2, 1): "#ace4e4",   # mid od, low su — light teal
    (3, 1): "#5ac8c8",   # high od, low su — teal (Appalachia overdose)
    (1, 2): "#dfb0d6",   # low od, mid su — light purple
    (2, 2): "#a5b3cc",   # mid od, mid su — slate
    (3, 2): "#5698b9",   # high od, mid su — blue-grey
    (1, 3): "#be64ac",   # low od, high su — purple (Mountain West)
    (2, 3): "#8c62aa",   # mid od, high su — mid purple-blue
    (3, 3): "#3b4994",   # high od, high su — dark blue (both high)
}

sub = gdf[gdf["suicide_rate"].notna() & gdf["overdose_rate"].notna()].copy().reset_index(drop=True)

# Tertile classification
su_cuts = sub["suicide_rate"].quantile([1/3, 2/3]).values
od_cuts = sub["overdose_rate"].quantile([1/3, 2/3]).values

def classify(val, cuts):
    if val <= cuts[0]: return 1
    if val <= cuts[1]: return 2
    return 3

sub["su_class"] = sub["suicide_rate"].apply(lambda v: classify(v, su_cuts))
sub["od_class"] = sub["overdose_rate"].apply(lambda v: classify(v, od_cuts))
sub["bv_color"] = sub.apply(lambda r: BIVAR[(r["od_class"], r["su_class"])], axis=1)

# Suppressed / missing counties
missing = gdf[gdf["suicide_rate"].isna() | gdf["overdose_rate"].isna()].copy()

fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

missing.plot(ax=ax, color="#d8d8d8", linewidth=0.05, edgecolor="white")
sub.plot(ax=ax, color=sub["bv_color"], linewidth=0.05, edgecolor="white")

set_extent(ax)

# Legend: 3×3 color matrix inset
legend_ax = fig.add_axes([0.04, 0.04, 0.09, 0.15])
legend_ax.set_facecolor(BG)
legend_ax.set_xlim(0, 3)
legend_ax.set_ylim(0, 3)
for od_c in [1, 2, 3]:
    for su_c in [1, 2, 3]:
        rect = plt.Rectangle(
            (od_c - 1, su_c - 1), 1, 1,
            color=BIVAR[(od_c, su_c)], linewidth=0
        )
        legend_ax.add_patch(rect)
legend_ax.set_xticks([0, 1.5, 3])
legend_ax.set_xticklabels(["Low", "", "High"], fontsize=7, color="#444444")
legend_ax.set_yticks([0, 1.5, 3])
legend_ax.set_yticklabels(["Low", "", "High"], fontsize=7, color="#444444")
legend_ax.set_xlabel("Overdose →", fontsize=8, color="#444444", labelpad=2)
legend_ax.set_ylabel("Suicide →", fontsize=8, color="#444444", labelpad=2)
legend_ax.tick_params(length=0)
for spine in legend_ax.spines.values():
    spine.set_visible(False)

# Annotations pointing to the two regimes
ax.annotate(
    "Mountain West:\nhigh suicide,\nlow overdose",
    xy=(-1.1e6, 2.1e6), xytext=(-1.9e6, 1.2e6),
    fontsize=9, color="#7b3f9e", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#7b3f9e", lw=1.5),
    ha="center"
)
ax.annotate(
    "Appalachia &\nNew England:\nhigh overdose,\nlow suicide",
    xy=(1.25e6, 1.85e6), xytext=(1.95e6, 1.2e6),
    fontsize=9, color="#3a9494", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#3a9494", lw=1.5),
    ha="center"
)

# Missing data note
fig.text(0.04, 0.03, "Grey = suppressed (< 10 deaths in either series)",
         fontsize=7, color="#888888")

ax.set_title(
    "The Two Americas of Despair: Suicide vs. Overdose Geography, 2018–2024\n"
    "Bivariate Moran's I = 0.016 (p = 0.15) — spatially independent crises, "
    "not one unified epidemic",
    fontsize=13, fontweight="bold", pad=12, color="#1a1a1a"
)

plt.savefig("fig_pub_bivariate.png", dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig_pub_bivariate.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: BV I trajectory — polished
# ═══════════════════════════════════════════════════════════════════════════════

print("\nFigure 2: BV I trajectory...")

ERAS = [
    ("1999–2005", "suicide_county_1999_2005.csv",  "overdose_county_1999_2005.csv",  "alcohol_liver_county_1999_2005.csv"),
    ("2006–2012", "suicide_county_2006_2012.csv",  "overdose_county_2006_2012.csv",  "alcohol_liver_county_2006_2012.csv"),
    ("2013–2019", "suicide_county_2013_2019.csv",  "overdose_county_2013_2019.csv",  "alcohol_liver_county_2013_2019.csv"),
    ("2018–2024", "suicide_county_2018_2024.csv",  "overdose_county_2018_2024.csv",  "alcohol_liver_county_2018_2024.csv"),
]

X_MID    = [2002, 2009, 2016, 2021]
X_LABELS = [
    "1999–2005\npre-epidemic",
    "2006–2012\nOxyContin peak",
    "2013–2019\nfentanyl transition",
    "2018–2024\nfentanyl era",
]

results = []
for era, su_f, od_f, k7_f in ERAS:
    print(f"  {era}...")
    su = load_wonder(su_f, "suicide_rate")
    od = load_wonder(od_f, "overdose_rate")
    k7 = load_wonder(k7_f, "k70_rate")

    m = su.merge(od, on="fips", how="outer").merge(k7, on="fips", how="outer")
    g = counties.merge(m, on="fips", how="left")

    g_su_od = g[g["suicide_rate"].notna() & g["overdose_rate"].notna()].copy().reset_index(drop=True)
    g_all   = g[g["suicide_rate"].notna() & g["overdose_rate"].notna() & g["k70_rate"].notna()].copy().reset_index(drop=True)

    w     = libpysal.weights.Queen.from_dataframe(g_su_od, silence_warnings=True); w.transform = "r"
    w_all = libpysal.weights.Queen.from_dataframe(g_all,   silence_warnings=True); w_all.transform = "r"

    bv_od = Moran_BV(g_su_od["suicide_rate"], g_su_od["overdose_rate"], w,     permutations=999)
    bv_k7 = Moran_BV(g_all["suicide_rate"],   g_all["k70_rate"],        w_all, permutations=999)

    results.append({
        "era": era,
        "bv_od": bv_od.I, "p_od": bv_od.p_sim,
        "bv_k7": bv_k7.I, "p_k7": bv_k7.p_sim,
        "mean_od": g_su_od["overdose_rate"].mean(),
    })
    print(f"    BV I(su × od)={bv_od.I:.3f} p={bv_od.p_sim:.3f}  "
          f"BV I(su × k70)={bv_k7.I:.3f} p={bv_k7.p_sim:.3f}")

bvi_od = [r["bv_od"] for r in results]
bvi_k7 = [r["bv_k7"] for r in results]
p_od   = [r["p_od"]  for r in results]
mean_od = [r["mean_od"] for r in results]

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# C&D data window
ax.axvspan(1999, 2013, alpha=0.10, color="#e84855", zorder=1)
ax.text(2006, 0.50,
        "Case & Deaton\ndata window\n(1999–2013)",
        fontsize=9, color="#b03030", ha="center", style="italic", va="top")

# Lines
ax.plot(X_MID, bvi_k7, "s--", color="#4d9de0", linewidth=2.2,
        markersize=9, label="Suicide × K70 (alcohol liver)", zorder=4, markeredgewidth=0)
ax.plot(X_MID, bvi_od, "o-",  color="#e84855", linewidth=2.5,
        markersize=11, label="Suicide × Overdose", zorder=5, markeredgewidth=0)

# Significance markers above points
SIG_OFFSET = 0.020
for x, y, p in zip(X_MID, bvi_od, p_od):
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    ax.text(x, y + SIG_OFFSET, sig, ha="center", fontsize=11,
            color="#e84855", fontweight="bold")

# Mean overdose rate on secondary axis
ax2 = ax.twinx()
ax2.set_facecolor(BG)
ax2.plot(X_MID, mean_od, "^:", color="#888888", linewidth=1.5,
         markersize=7, markeredgewidth=0, alpha=0.7, label="Mean OD rate (per 100K)")
ax2.set_ylabel("County mean overdose rate (per 100K)", fontsize=10, color="#888888")
ax2.tick_params(axis="y", labelcolor="#888888")
ax2.set_ylim(0, 35)

# Zero line
ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle=":")

ax.set_xticks(X_MID)
ax.set_xticklabels(X_LABELS, fontsize=10)
ax.set_xlim(1998, 2024)
ax.set_ylabel("Bivariate Moran's I", fontsize=11)
ax.set_ylim(-0.04, 0.54)
ax.grid(axis="y", alpha=0.25, linewidth=0.8)

# Combine legends
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=10, loc="upper right", framealpha=0.9,
          facecolor=BG, edgecolor="#cccccc")

# Annotation: the peak
ax.annotate(
    "Peak co-clustering\n(OxyContin era)",
    xy=(2009, 0.277), xytext=(2009, 0.38),
    ha="center", fontsize=9, color="#c03030",
    arrowprops=dict(arrowstyle="->", color="#c03030", lw=1.3)
)

# Annotation: the collapse
ax.annotate(
    "Fentanyl geography\nsplits from suicide",
    xy=(2021, 0.016), xytext=(2019.5, 0.09),
    ha="center", fontsize=9, color="#c03030",
    arrowprops=dict(arrowstyle="->", color="#c03030", lw=1.3)
)

ax.set_title(
    "Collapse of Suicide–Overdose Spatial Co-clustering, 1999–2024\n"
    "K70 (alcohol liver disease) remains coupled to suicide throughout; overdose decouples completely",
    fontsize=12, fontweight="bold", pad=12, color="#1a1a1a"
)

plt.savefig("fig_pub_trajectory.png", dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig_pub_trajectory.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Regime classification map
# ═══════════════════════════════════════════════════════════════════════════════

print("\nFigure 3: Regime classification map...")

# Reload current era
su = load_wonder("suicide_county_2018_2024.csv",      "suicide_rate")
od = load_wonder("overdose_county_2018_2024.csv",     "overdose_rate")
k7 = load_wonder("alcohol_liver_county_2018_2024.csv","k70_rate")

m_all = su.merge(od, on="fips", how="outer").merge(k7, on="fips", how="outer")

def tercile(series):
    cuts = series.quantile([1/3, 2/3]).values
    return pd.cut(series, bins=[-np.inf, cuts[0], cuts[1], np.inf],
                  labels=[0, 1, 2]).astype("Int64")

m_all["t_su"] = tercile(m_all["suicide_rate"])
m_all["t_od"] = tercile(m_all["overdose_rate"])
m_all["t_k7"] = tercile(m_all["k70_rate"])

def assign_regime(row):
    su = int(row["t_su"]) if not pd.isna(row["t_su"]) else None
    od = int(row["t_od"]) if not pd.isna(row["t_od"]) else None
    k7 = int(row["t_k7"]) if not pd.isna(row["t_k7"]) else None
    if su is None or od is None:
        return "Insufficient data"
    if su == 2 and od == 2 and k7 == 2:
        return "C: All high"
    if su == 2 and (k7 is None or k7 == 2) and od != 2:
        return "A: Chronic\n(suicide + K70)"
    if od == 2 and su != 2 and (k7 is None or k7 != 2):
        return "B: Supply-chain\n(overdose)"
    if su == 0 and od == 0 and (k7 is None or k7 == 0):
        return "D: All low"
    return "Mixed"

m_all["regime"] = m_all.apply(assign_regime, axis=1)

REGIME_COLORS = {
    "A: Chronic\n(suicide + K70)": "#be64ac",    # purple
    "B: Supply-chain\n(overdose)": "#5ac8c8",     # teal
    "C: All high":                  "#3b4994",    # dark blue
    "D: All low":                   "#d8e4f0",    # pale blue-grey
    "Mixed":                        "#c8c8c8",    # grey
    "Insufficient data":            "#e8e8e8",    # light grey
}

gdf_r = counties.merge(m_all[["fips","regime"]], on="fips", how="left")
gdf_r["regime"] = gdf_r["regime"].fillna("Insufficient data")
gdf_r["color"] = gdf_r["regime"].map(REGIME_COLORS)

# Count per regime for legend labels
counts = m_all["regime"].value_counts()

fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

gdf_r.plot(ax=ax, color=gdf_r["color"], linewidth=0.05, edgecolor="white")
set_extent(ax)

# Legend
legend_order = [
    "A: Chronic\n(suicide + K70)",
    "B: Supply-chain\n(overdose)",
    "C: All high",
    "D: All low",
    "Mixed",
    "Insufficient data",
]
patches = []
for regime in legend_order:
    n = counts.get(regime, 0)
    label = f"{regime.replace(chr(10), ' ')}  (n={n})" if n > 0 else regime.replace("\n", " ")
    patches.append(mpatches.Patch(color=REGIME_COLORS[regime], label=label))

ax.legend(handles=patches, loc="lower left", fontsize=9,
          framealpha=0.92, facecolor=BG, edgecolor="#cccccc",
          title="Death regime", title_fontsize=10)

ax.set_title(
    "County Death Regimes, 2018–2024\n"
    "Regime A (purple): Mountain West chronic despair  |  "
    "Regime B (teal): Appalachia/New England supply-chain poisoning",
    fontsize=12, fontweight="bold", pad=12, color="#1a1a1a"
)

fig.text(
    0.5, 0.01,
    "Regimes assigned by tertile rank on each axis. "
    "Poverty rates: Regime A 13.9%, Regime B 14.3% (p = 0.76, ns) — economic divergence explains neither crisis.",
    ha="center", fontsize=8, color="#666666"
)

plt.savefig("fig_pub_regimes.png", dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig_pub_regimes.png")

print("\nDone. Publication figures:")
print("  fig_pub_bivariate.png  — bivariate choropleth (suicide × overdose)")
print("  fig_pub_trajectory.png — BV I trajectory across four eras")
print("  fig_pub_regimes.png    — regime classification map")
