"""
Export structured data for native web figure rendering.

Outputs three files to web/data/:
  trajectory.json      — BV I series across four eras, both pairs, with
                         permutation-null 95% ranges as uncertainty bands
  bivariate_lisa.json  — per-county bivariate LISA (suicide × overdose, 2018-24)
  regimes.json         — per-county regime classification (A, B, or null)

County geometry note: analysis used Census TIGER/Line 2023 full-resolution
shapefile (tl_2023_us_county.zip). For web rendering, use the 20m cartographic
boundary file instead — same county set, simplified for display:
  CDN: https://cdn.jsdelivr.net/npm/us-atlas@3/counties-10m.json
  Census: https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_20m.zip
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import libpysal
from esda import Moran_BV, Moran_Local_BV

OUT_DIR = Path("web/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ERAS = [
    ("1999-2005", 2002, "data/suicide_county_1999_2005.csv",  "data/overdose_county_1999_2005.csv",  "data/alcohol_liver_county_1999_2005.csv"),
    ("2006-2012", 2009, "data/suicide_county_2006_2012.csv",  "data/overdose_county_2006_2012.csv",  "data/alcohol_liver_county_2006_2012.csv"),
    ("2013-2019", 2016, "data/suicide_county_2013_2019.csv",  "data/overdose_county_2013_2019.csv",  "data/alcohol_liver_county_2013_2019.csv"),
    ("2018-2024", 2021, "data/suicide_county_2018_2024.csv",  "data/overdose_county_2018_2024.csv",  "data/alcohol_liver_county_2018_2024.csv"),
]

PERMS = 9999

def load_wonder(path, rate_col):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip().str.strip('"')
    df = df[df["County Code"].str.match(r"^\d{5}$", na=False)].copy()
    df = df.rename(columns={"County Code": "fips"})
    df["rate"] = pd.to_numeric(df["Crude Rate"], errors="coerce")
    return df[["fips", "rate"]].rename(columns={"rate": rate_col})

# ── Load shapefile once ────────────────────────────────────────────────────────

print("Loading shapefile...")
counties = gpd.read_file(
    "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
)
counties = counties.rename(columns={"GEOID": "fips"})
EXCLUDE = {"02", "15", "60", "66", "69", "72", "78"}
counties = counties[~counties["STATEFP"].isin(EXCLUDE)].to_crs("EPSG:5070")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Trajectory data
# ══════════════════════════════════════════════════════════════════════════════

print("\nComputing trajectory BV I series...")

traj_su_od = []
traj_su_k7 = []

for era, midpoint, su_f, od_f, k7_f in ERAS:
    print(f"  {era}...")
    su = load_wonder(su_f, "suicide_rate")
    od = load_wonder(od_f, "overdose_rate")
    k7 = load_wonder(k7_f, "k70_rate")

    merged = su.merge(od, on="fips", how="outer").merge(k7, on="fips", how="outer")
    gdf = counties.merge(merged, on="fips", how="left")

    g_su_od = gdf[gdf["suicide_rate"].notna() & gdf["overdose_rate"].notna()].copy().reset_index(drop=True)
    g_all   = gdf[gdf["suicide_rate"].notna() & gdf["overdose_rate"].notna() & gdf["k70_rate"].notna()].copy().reset_index(drop=True)

    w     = libpysal.weights.Queen.from_dataframe(g_su_od, silence_warnings=True); w.transform = "r"
    w_all = libpysal.weights.Queen.from_dataframe(g_all,   silence_warnings=True); w_all.transform = "r"

    bv_od = Moran_BV(g_su_od["suicide_rate"], g_su_od["overdose_rate"], w,     permutations=PERMS)
    bv_k7 = Moran_BV(g_all["suicide_rate"],   g_all["k70_rate"],        w_all, permutations=PERMS)

    # 95% range from permutation null (2.5th–97.5th percentile)
    # Note: these are the null distribution bounds, not bootstrap CIs around the estimate
    def null_range(bv):
        sim = np.array(bv.sim)
        return float(np.percentile(sim, 2.5)), float(np.percentile(sim, 97.5))

    od_lo, od_hi = null_range(bv_od)
    k7_lo, k7_hi = null_range(bv_k7)

    traj_su_od.append({
        "period": era, "midpoint": midpoint,
        "bvi": round(bv_od.I, 4), "p": round(bv_od.p_sim, 4),
        "null_ci_lo": round(od_lo, 4), "null_ci_hi": round(od_hi, 4),
    })
    traj_su_k7.append({
        "period": era, "midpoint": midpoint,
        "bvi": round(bv_k7.I, 4), "p": round(bv_k7.p_sim, 4),
        "null_ci_lo": round(k7_lo, 4), "null_ci_hi": round(k7_hi, 4),
    })
    print(f"    su×od={bv_od.I:.4f} (p={bv_od.p_sim:.4f})  su×k70={bv_k7.I:.4f} (p={bv_k7.p_sim:.4f})")

trajectory = {
    "_note": (
        "null_ci_lo/hi are the 2.5th–97.5th percentile of the permutation null "
        "distribution (not bootstrap CIs around the estimate). Use for visual "
        "reference bands showing the range of values consistent with spatial "
        "randomness, not as uncertainty around the BV I point estimate."
    ),
    "suicide_overdose": traj_su_od,
    "suicide_k70": traj_su_k7,
}

traj_path = OUT_DIR / "trajectory.json"
traj_path.write_text(json.dumps(trajectory, indent=2))
print(f"\n  Saved {traj_path}  ({traj_path.stat().st_size // 1024}KB)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Bivariate LISA classifications (suicide × overdose, 2018–2024)
# ══════════════════════════════════════════════════════════════════════════════

print("\nComputing bivariate LISA (suicide × overdose, 2018–2024)...")

su18 = load_wonder("data/suicide_county_2018_2024.csv",  "suicide_rate")
od18 = load_wonder("data/overdose_county_2018_2024.csv", "overdose_rate")
merged18 = su18.merge(od18, on="fips", how="outer")
gdf18 = counties.merge(merged18, on="fips", how="left")
gdf18 = gdf18[gdf18["suicide_rate"].notna() & gdf18["overdose_rate"].notna()].copy().reset_index(drop=True)

w18 = libpysal.weights.Queen.from_dataframe(gdf18, silence_warnings=True); w18.transform = "r"

lisa_bv = Moran_Local_BV(gdf18["suicide_rate"], gdf18["overdose_rate"], w18, permutations=PERMS, seed=42)

# q: 1=HH, 2=LH, 3=LL, 4=HL  (focal=suicide, lag=overdose)
QUAD = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}

bv_lisa_out = []
for i, fips in enumerate(gdf18["fips"]):
    sig = lisa_bv.p_sim[i] < 0.05
    bv_class = QUAD.get(lisa_bv.q[i], "NS") if sig else "NS"
    bv_lisa_out.append({"fips": str(fips), "bv_class": bv_class})

# Summary
from collections import Counter
counts = Counter(r["bv_class"] for r in bv_lisa_out)
print(f"  n={len(bv_lisa_out)}  " + "  ".join(f"{k}:{v}" for k, v in sorted(counts.items())))

lisa_path = OUT_DIR / "bivariate_lisa.json"
lisa_path.write_text(json.dumps(bv_lisa_out))
print(f"  Saved {lisa_path}  ({lisa_path.stat().st_size // 1024}KB)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Regime classifications (2018–2024)
# ══════════════════════════════════════════════════════════════════════════════

print("\nComputing regime classifications...")

k70_18 = load_wonder("data/alcohol_liver_county_2018_2024.csv", "k70_rate")
all18 = su18.merge(od18, on="fips", how="outer").merge(k70_18, on="fips", how="outer")

def tercile(series):
    cuts = series.quantile([1/3, 2/3]).values
    return pd.cut(series, bins=[-np.inf, cuts[0], cuts[1], np.inf],
                  labels=[0, 1, 2]).astype("Int64")

all18["t_su"] = tercile(all18["suicide_rate"])
all18["t_od"] = tercile(all18["overdose_rate"])
all18["t_k7"] = tercile(all18["k70_rate"])

def assign_regime(row):
    su = int(row["t_su"]) if not pd.isna(row["t_su"]) else None
    od = int(row["t_od"]) if not pd.isna(row["t_od"]) else None
    k7 = int(row["t_k7"]) if not pd.isna(row["t_k7"]) else None
    if su is None or od is None:
        return None
    if su == 2 and od == 2 and k7 == 2:
        return "C"
    if su == 2 and (k7 is None or k7 == 2) and od != 2:
        return "A"
    if od == 2 and su != 2 and (k7 is None or k7 != 2):
        return "B"
    if su == 0 and od == 0 and (k7 is None or k7 == 0):
        return "D"
    return None

all18["regime"] = all18.apply(assign_regime, axis=1)

# Export all counties that have at least a suicide or overdose rate (include nulls)
# Full CONUS county list from shapefile for complete coverage
all_fips = counties["fips"].tolist()
regime_lookup = dict(zip(all18["fips"], all18["regime"]))

regimes_out = [
    {"fips": f, "regime": regime_lookup.get(f)}
    for f in all_fips
]

counts_r = Counter(r["regime"] for r in regimes_out)
print(f"  n={len(regimes_out)}  " + "  ".join(f"{k}:{v}" for k, v in sorted(counts_r.items(), key=lambda x: str(x[0]))))

regimes_path = OUT_DIR / "regimes.json"
regimes_path.write_text(json.dumps(regimes_out))
print(f"  Saved {regimes_path}  ({regimes_path.stat().st_size // 1024}KB)")

print("\nDone.")
print(f"\nGeometry note: analysis used tl_2023_us_county.zip (TIGER/Line full resolution).")
print(f"For web rendering, use the 20m cartographic file instead:")
print(f"  CDN:    https://cdn.jsdelivr.net/npm/us-atlas@3/counties-10m.json")
print(f"  Census: https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_20m.zip")
print(f"Join on FIPS (5-digit string). us-atlas uses numeric FIPS — zero-pad to 5 chars before joining.")
