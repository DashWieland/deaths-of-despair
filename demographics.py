"""
Demographic profiling of spatial regimes.

Assigns each county to a regime based on its death rate terciles:
  - Regime A: high suicide + high K70, low overdose  {chronic self-destruction}
  - Regime B: low suicide + low K70, high overdose   {supply-chain poisoning}
  - Regime C: high on all three                      {convergence}
  - Other: everything else (mid-range, mixed)

Pulls ACS 2023 5-year county-level estimates via Census API.
Compares mean demographics across regimes.
"""

import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── 1. Pull ACS demographics via Census API ────────────────────────────────────

VARS = {
    "B19013_001E": "median_income",
    "B15003_001E": "edu_total",
    "B15003_022E": "edu_ba",
    "B15003_023E": "edu_ma",
    "B15003_024E": "edu_prof",
    "B15003_025E": "edu_phd",
    "B02001_001E": "race_total",
    "B02001_002E": "race_white",
    "B02001_004E": "race_aian",
    "B17001_001E": "pov_total",
    "B17001_002E": "pov_below",
    "B01002_001E": "median_age",
    "B05012_001E": "nativity_total",
    "B05012_003E": "nativity_foreign",
    "B01001_001E": "pop_total",
}

var_str = ",".join(VARS.keys())
url = (
    f"https://api.census.gov/data/2023/acs/acs5"
    f"?get=NAME,{var_str}&for=county:*&in=state:*"
)

print("Fetching ACS 2023 5-year county demographics...")
resp = requests.get(url, timeout=60)
resp.raise_for_status()
raw = resp.json()

header = raw[0]
rows   = raw[1:]
acs = pd.DataFrame(rows, columns=header)

# Build FIPS and rename
acs["fips"] = acs["state"] + acs["county"]
acs = acs.rename(columns=VARS)
acs = acs.drop(columns=["state", "county", "NAME"])

for col in VARS.values():
    acs[col] = pd.to_numeric(acs[col], errors="coerce").replace(-666666666, np.nan)

# Derived variables
acs["pct_ba_plus"]    = (acs["edu_ba"] + acs["edu_ma"] + acs["edu_prof"] + acs["edu_phd"]) / acs["edu_total"] * 100
acs["pct_white"]      = acs["race_white"] / acs["race_total"] * 100
acs["pct_aian"]       = acs["race_aian"]  / acs["race_total"] * 100
acs["pct_poverty"]    = acs["pov_below"]  / acs["pov_total"]  * 100
acs["pct_foreign"]    = acs["nativity_foreign"] / acs["nativity_total"] * 100

DEMO_COLS = ["median_income", "pct_ba_plus", "pct_white", "pct_aian",
             "pct_poverty", "median_age", "pct_foreign"]

print(f"  {len(acs)} counties retrieved")

# ── 2. Load death rates ────────────────────────────────────────────────────────

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

# ── 3. Assign regimes via terciles ─────────────────────────────────────────────

def tercile(series):
    """Return 0/1/2 (low/mid/high) tercile assignment."""
    cuts = series.quantile([1/3, 2/3]).values
    return pd.cut(series, bins=[-np.inf, cuts[0], cuts[1], np.inf],
                  labels=[0, 1, 2]).astype("Int64")

deaths["t_suicide"]  = tercile(deaths["suicide_rate"])
deaths["t_overdose"] = tercile(deaths["overdose_rate"])
deaths["t_k70"]      = tercile(deaths["k70_rate"])

def assign_regime(row):
    su = int(row["t_suicide"]) if not pd.isna(row["t_suicide"]) else None
    od = int(row["t_overdose"]) if not pd.isna(row["t_overdose"]) else None
    k7 = int(row["t_k70"]) if not pd.isna(row["t_k70"]) else None
    if su is None or od is None:
        return "Missing"
    # Convergence: high on all three (where k70 available)
    if su == 2 and od == 2 and k7 == 2:
        return "C: High all"
    # Chronic self-destruction: high suicide + high K70, not high overdose
    if su == 2 and (k7 is None or k7 == 2) and od != 2:
        return "A: Chronic (suicide+K70)"
    # Supply-chain poisoning: high overdose, low/mid suicide + K70
    if od == 2 and su != 2 and (k7 is None or k7 != 2):
        return "B: Supply-chain (overdose)"
    # Low on everything
    if su == 0 and od == 0 and (k7 is None or k7 == 0):
        return "D: Low all"
    return "Other"

deaths["regime"] = deaths.apply(assign_regime, axis=1)

regime_counts = deaths["regime"].value_counts()
print("\nRegime assignment:")
for r, n in regime_counts.items():
    print(f"  {r:<35} {n}")

# ── 4. Merge with demographics ─────────────────────────────────────────────────

df = deaths.merge(acs[["fips"] + DEMO_COLS], on="fips", how="left")

# ── 5. Profile table ───────────────────────────────────────────────────────────

REGIME_ORDER = [
    "A: Chronic (suicide+K70)",
    "B: Supply-chain (overdose)",
    "C: High all",
    "D: Low all",
    "Other",
]

LABELS = {
    "median_income": "Median HH income ($)",
    "pct_ba_plus":   "% Bachelor's+",
    "pct_white":     "% White non-Hispanic (approx)",
    "pct_aian":      "% American Indian / AK Native",
    "pct_poverty":   "% Below poverty line",
    "median_age":    "Median age",
    "pct_foreign":   "% Foreign-born",
}

print("\n" + "="*90)
print("MEAN DEMOGRAPHICS BY REGIME")
print("="*90)

active_regimes = [r for r in REGIME_ORDER if r in df["regime"].values]
header = f"{'Variable':<35}" + "".join(f"{r[:18]:>18}" for r in active_regimes)
print(header)
print("-"*90)

profile = {}
for col, label in LABELS.items():
    row_vals = {}
    for regime in active_regimes:
        sub = df[df["regime"] == regime][col].dropna()
        row_vals[regime] = sub.mean()
    profile[col] = row_vals
    vals_str = "".join(f"{v:>18.1f}" for v in row_vals.values())
    print(f"{label:<35}{vals_str}")

print("-"*90)
n_row = f"{'n counties':<35}" + "".join(
    f"{(df['regime']==r).sum():>18}" for r in active_regimes
)
print(n_row)

# ── 6. Statistical tests (A vs B pairwise) ────────────────────────────────────

print("\n" + "="*70)
print("REGIME A vs B: Mann-Whitney U tests")
print("="*70)
print(f"{'Variable':<35} {'A median':>10} {'B median':>10} {'p':>10}")
print("-"*70)

a_df = df[df["regime"] == "A: Chronic (suicide+K70)"]
b_df = df[df["regime"] == "B: Supply-chain (overdose)"]

for col, label in LABELS.items():
    a_vals = a_df[col].dropna()
    b_vals = b_df[col].dropna()
    if len(a_vals) < 5 or len(b_vals) < 5:
        continue
    _, p = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"{label:<35} {a_vals.median():>10.1f} {b_vals.median():>10.1f} {p:>8.4f} {sig}")

# ── 7. Figure: demographic profiles ───────────────────────────────────────────

print("\nGenerating demographic profile figure...")

plot_vars = ["median_income", "pct_ba_plus", "pct_white", "pct_aian", "pct_poverty", "median_age"]
plot_labels = [LABELS[v] for v in plot_vars]

REGIME_COLORS = {
    "A: Chronic (suicide+K70)":   "#4d9de0",
    "B: Supply-chain (overdose)":  "#e84855",
    "C: High all":                 "#7b2d8b",
    "D: Low all":                  "#aaaaaa",
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor("#f8f8f6")
axes = axes.flatten()

for ax, col, label in zip(axes, plot_vars, plot_labels):
    plot_regimes = [r for r in ["A: Chronic (suicide+K70)", "B: Supply-chain (overdose)",
                                "C: High all", "D: Low all"] if r in df["regime"].values]
    data_by_regime = [df[df["regime"] == r][col].dropna().values for r in plot_regimes]
    colors = [REGIME_COLORS[r] for r in plot_regimes]
    short_labels = ["A: Chronic\n(suicide+K70)", "B: Supply-chain\n(overdose)",
                    "C: High all", "D: Low all"][:len(plot_regimes)]

    bp = ax.boxplot(data_by_regime, patch_artist=True, medianprops={"color": "black", "linewidth": 2},
                    whiskerprops={"linewidth": 1}, capprops={"linewidth": 1},
                    flierprops={"marker": ".", "markersize": 2, "alpha": 0.3})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(plot_regimes) + 1))
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_facecolor("#f8f8f6")
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Demographic Profiles by Spatial Regime\n(ACS 2023 5-year estimates, county-level medians)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("figures/fig_demographics.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  Saved fig_demographics.png")
print("\nDone.")
