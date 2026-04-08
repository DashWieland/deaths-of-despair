"""
AIAN decomposition.

Regime A (chronic: suicide+K70) has mean 5% AIAN but median 0.8% —
a handful of high-AIAN counties pull the mean. This tests whether the
altitude + gun + isolation pattern holds in predominantly WHITE Regime A
counties once tribal-lands counties are set aside.

If the pattern holds: two distinct sub-populations inside Regime A.
If the pattern collapses: tribal lands are doing most of the work.
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

# ── Load all data sources ──────────────────────────────────────────────────────

# Death rates
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

deaths = (suicide
          .merge(overdose, on="fips", how="outer")
          .merge(k70,      on="fips", how="outer"))

# Elevation
with open("data/elevation_cache.json") as f:
    elevations = json.load(f)
deaths["elevation_m"] = deaths["fips"].map(elevations)

# Firearm fraction
firearm = load_wonder("data/firearm_suicide_county_2018_2024.csv","firearm_rate")
deaths = deaths.merge(firearm, on="fips", how="left")
deaths["firearm_fraction"] = (deaths["firearm_rate"] / deaths["suicide_rate"]).clip(0, 1)

# ACS demographics (re-fetch key variables)
import requests
VARS = {
    "B02001_001E": "race_total",
    "B02001_004E": "race_aian",
    "B02001_002E": "race_white",
    "B01001_001E": "pop_total",
}
var_str = ",".join(VARS.keys())
print("Fetching ACS demographics...")
resp = requests.get(
    f"https://api.census.gov/data/2023/acs/acs5?get=NAME,{var_str}&for=county:*&in=state:*",
    timeout=60
)
raw = resp.json()
acs = pd.DataFrame(raw[1:], columns=raw[0])
acs["fips"] = acs["state"] + acs["county"]
acs = acs.rename(columns=VARS)
for col in VARS.values():
    acs[col] = pd.to_numeric(acs[col], errors="coerce").replace(-666666666, np.nan)
acs["pct_aian"]  = acs["race_aian"]  / acs["race_total"] * 100
acs["pct_white"] = acs["race_white"] / acs["race_total"] * 100

df = deaths.merge(acs[["fips","pct_aian","pct_white","pop_total"]], on="fips", how="left")

# ── Assign regimes ─────────────────────────────────────────────────────────────

def tercile(series):
    cuts = series.quantile([1/3, 2/3]).values
    return pd.cut(series, bins=[-np.inf, cuts[0], cuts[1], np.inf],
                  labels=[0, 1, 2]).astype("Int64")

df["t_suicide"]  = tercile(df["suicide_rate"])
df["t_overdose"] = tercile(df["overdose_rate"])
df["t_k70"]      = tercile(df["k70_rate"])

def assign_regime(row):
    su = int(row["t_suicide"]) if not pd.isna(row["t_suicide"]) else None
    od = int(row["t_overdose"]) if not pd.isna(row["t_overdose"]) else None
    k7 = int(row["t_k70"]) if not pd.isna(row["t_k70"]) else None
    if su is None or od is None:
        return "Missing"
    if su == 2 and od == 2 and k7 == 2:
        return "C: High all"
    if su == 2 and (k7 is None or k7 == 2) and od != 2:
        return "A: Chronic (suicide+K70)"
    if od == 2 and su != 2 and (k7 is None or k7 != 2):
        return "B: Supply-chain (overdose)"
    if su == 0 and od == 0 and (k7 is None or k7 == 0):
        return "D: Low all"
    return "Other"

df["regime"] = df.apply(assign_regime, axis=1)

# ── Split Regime A into high-AIAN vs white-majority ───────────────────────────

AIAN_THRESHOLD = 5.0  # % — counties above this are tribal-lands-influenced

regime_a = df[df["regime"] == "A: Chronic (suicide+K70)"].copy()
regime_a["subgroup"] = np.where(
    regime_a["pct_aian"] >= AIAN_THRESHOLD,
    "A-AIAN (≥5% AIAN)",
    "A-White (<5% AIAN)"
)

print(f"\nRegime A breakdown (threshold: {AIAN_THRESHOLD}% AIAN):")
print(regime_a["subgroup"].value_counts().to_string())

# ── Predictors in each subgroup ────────────────────────────────────────────────

print("\n" + "="*75)
print("REGIME A SUBGROUP PROFILES: predictors of chronic death regime")
print("="*75)

predictors = [
    ("elevation_m",       "Mean elevation (m)"),
    ("firearm_fraction",  "Firearm suicide fraction"),
    ("pct_white",         "% White"),
    ("pct_aian",          "% AIAN"),
    ("suicide_rate",      "Suicide rate"),
    ("k70_rate",          "K70 rate"),
    ("overdose_rate",     "Overdose rate"),
]

print(f"\n{'Variable':<30} {'A-AIAN median':>15} {'A-White median':>15} {'p (MW)':>10}")
print("-"*75)

a_aian  = regime_a[regime_a["subgroup"] == "A-AIAN (≥5% AIAN)"]
a_white = regime_a[regime_a["subgroup"] == "A-White (<5% AIAN)"]

for col, label in predictors:
    v_aian  = a_aian[col].dropna()
    v_white = a_white[col].dropna()
    if len(v_aian) < 3 or len(v_white) < 3:
        continue
    _, p = stats.mannwhitneyu(v_aian, v_white, alternative="two-sided")
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"{label:<30} {v_aian.median():>15.2f} {v_white.median():>15.2f} {p:>8.4f} {sig}")

# ── Key question: does altitude still predict suicide in white-only Regime A? ──

print("\n" + "="*65)
print("ELEVATION-SUICIDE CORRELATION WITHIN REGIME A SUBGROUPS")
print("="*65)

for subgroup, sub_df in [("A-AIAN",  a_aian),
                          ("A-White", a_white),
                          ("Regime B (overdose)", df[df["regime"]=="B: Supply-chain (overdose)"])]:
    for col, label in [("suicide_rate","Suicide"), ("overdose_rate","Overdose")]:
        sub = sub_df[["elevation_m", col]].dropna()
        if len(sub) < 10:
            continue
        r, p = stats.spearmanr(sub["elevation_m"], sub[col])
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  {subgroup:<35} {label:<10} ρ={r:>6.3f}  p={p:.4f} {sig}  n={len(sub)}")

# ── Figure ─────────────────────────────────────────────────────────────────────

print("\nGenerating figure...")

COLORS = {
    "A-AIAN (≥5% AIAN)":  "#7b2d8b",
    "A-White (<5% AIAN)": "#4d9de0",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#f8f8f6")

# Panel 1: elevation vs suicide, split by subgroup
ax = axes[0]
for sg, color in COLORS.items():
    sub = regime_a[regime_a["subgroup"] == sg][["elevation_m","suicide_rate"]].dropna()
    ax.scatter(sub["elevation_m"], sub["suicide_rate"],
               s=15, alpha=0.6, color=color, label=sg, linewidths=0)
    if len(sub) > 10:
        m, b, *_ = stats.linregress(sub["elevation_m"], sub["suicide_rate"])
        xr = np.array([sub["elevation_m"].min(), sub["elevation_m"].max()])
        ax.plot(xr, m*xr+b, color=color, linewidth=2)

ax.set_xlabel("Elevation (m)", fontsize=10)
ax.set_ylabel("Suicide rate (per 100K)", fontsize=10)
ax.set_title("Regime A: elevation vs. suicide\nby AIAN subgroup", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.set_facecolor("#f8f8f6")

# Panel 2: elevation vs K70, split by subgroup
ax = axes[1]
for sg, color in COLORS.items():
    sub = regime_a[regime_a["subgroup"] == sg][["elevation_m","k70_rate"]].dropna()
    ax.scatter(sub["elevation_m"], sub["k70_rate"],
               s=15, alpha=0.6, color=color, label=sg, linewidths=0)
    if len(sub) > 10:
        m, b, *_ = stats.linregress(sub["elevation_m"], sub["k70_rate"])
        xr = np.array([sub["elevation_m"].min(), sub["elevation_m"].max()])
        ax.plot(xr, m*xr+b, color=color, linewidth=2)

ax.set_xlabel("Elevation (m)", fontsize=10)
ax.set_ylabel("K70 rate (per 100K)", fontsize=10)
ax.set_title("Regime A: elevation vs. K70\nby AIAN subgroup", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.set_facecolor("#f8f8f6")

plt.suptitle("Regime A Decomposition: Tribal Lands vs. White Rural Mountain West\n"
             "Does the altitude signal hold in predominantly white counties?",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("figures/fig_aian_decomp.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig_aian_decomp.png")
print("\nDone.")
