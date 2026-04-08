"""
Gun ownership proxy: firearm suicide fraction
(firearm suicides X72-X74 / total suicides X60-X84)

Validated proxy for county-level gun ownership (Cook & Ludwig).
Tests whether it separates regimes and predicts each death axis.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── Load data ──────────────────────────────────────────────────────────────────

def load_wonder(path, rate_col):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip().str.strip('"')
    df = df[df["County Code"].str.match(r"^\d{5}$", na=False)].copy()
    df = df.rename(columns={"County Code": "fips"})
    df["rate"]   = pd.to_numeric(df["Crude Rate"], errors="coerce")
    df["deaths"] = pd.to_numeric(df["Deaths"],     errors="coerce")
    return df[["fips", "rate", "deaths"]].rename(
        columns={"rate": rate_col, "deaths": f"deaths_{rate_col}"}
    )

suicide  = load_wonder("data/suicide_county_2018_2024.csv",      "suicide_rate")
firearm  = load_wonder("data/firearm_suicide_county_2018_2024.csv", "firearm_rate")
overdose = load_wonder("data/overdose_county_2018_2024.csv",     "overdose_rate")
k70      = load_wonder("data/alcohol_liver_county_2018_2024.csv","k70_rate")

df = (suicide
      .merge(firearm[["fips","firearm_rate","deaths_firearm_rate"]], on="fips", how="left")
      .merge(overdose[["fips","overdose_rate"]], on="fips", how="outer")
      .merge(k70[["fips","k70_rate"]],           on="fips", how="outer"))

# Firearm fraction: where both firearm and total suicide counts are present
df["firearm_fraction"] = df["firearm_rate"] / df["suicide_rate"]
# Clip to [0,1] — rounding artifacts can push slightly over 1
df["firearm_fraction"] = df["firearm_fraction"].clip(0, 1)

print(f"Counties with firearm fraction: {df['firearm_fraction'].notna().sum()}")
print(f"  Mean:   {df['firearm_fraction'].mean():.3f}")
print(f"  Median: {df['firearm_fraction'].median():.3f}")
print(f"  Range:  {df['firearm_fraction'].min():.3f} – {df['firearm_fraction'].max():.3f}")

# ── Correlations: firearm fraction vs. each death axis ────────────────────────

print("\n" + "="*65)
print("SPEARMAN CORRELATIONS: firearm fraction vs. death rates")
print("="*65)
print(f"{'Axis':<25} {'rho':>8} {'p':>10} {'n':>6}")
print("-"*65)

for col, label in [
    ("suicide_rate",  "Suicide"),
    ("k70_rate",      "K70 (liver)"),
    ("overdose_rate", "Overdose"),
]:
    sub = df[["firearm_fraction", col]].dropna()
    r, p = stats.spearmanr(sub["firearm_fraction"], sub[col])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"{label:<25} {r:>8.4f} {p:>10.4f} {sig}  (n={len(sub)})")

# ── Regime-level means ─────────────────────────────────────────────────────────

# Reconstruct regime assignments (mirror demographics.py logic)
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

print("\n" + "="*65)
print("FIREARM FRACTION BY REGIME")
print("="*65)
print(f"{'Regime':<35} {'median':>8} {'mean':>8} {'n':>6}")
print("-"*65)

order = ["A: Chronic (suicide+K70)", "B: Supply-chain (overdose)",
         "C: High all", "D: Low all"]
for regime in order:
    sub = df[df["regime"] == regime]["firearm_fraction"].dropna()
    if len(sub) > 0:
        print(f"{regime:<35} {sub.median():>8.3f} {sub.mean():>8.3f} {len(sub):>6}")

# Mann-Whitney A vs B
a = df[df["regime"] == "A: Chronic (suicide+K70)"]["firearm_fraction"].dropna()
b = df[df["regime"] == "B: Supply-chain (overdose)"]["firearm_fraction"].dropna()
_, p_ab = stats.mannwhitneyu(a, b, alternative="two-sided")
print(f"\nRegime A vs B Mann-Whitney p = {p_ab:.6f}")

# ── Figure ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor("#f8f8f6")

COLORS = {"A: Chronic (suicide+K70)": "#4d9de0",
          "B: Supply-chain (overdose)": "#e84855",
          "C: High all": "#7b2d8b",
          "D: Low all": "#aaaaaa"}

# Panel 1: firearm fraction vs suicide rate
ax = axes[0]
for regime, color in COLORS.items():
    sub = df[df["regime"] == regime][["firearm_fraction","suicide_rate"]].dropna()
    ax.scatter(sub["firearm_fraction"], sub["suicide_rate"],
               s=5, alpha=0.5, color=color, label=regime.split(":")[0], linewidths=0)
r, p = stats.spearmanr(df["firearm_fraction"].dropna(),
                        df.loc[df["firearm_fraction"].notna(), "suicide_rate"])
ax.set_xlabel("Firearm suicide fraction", fontsize=10)
ax.set_ylabel("Suicide rate (per 100K)", fontsize=10)
ax.set_title(f"vs. Suicide  ρ={r:.3f}***", fontsize=11, fontweight="bold")
ax.set_facecolor("#f8f8f6")

# Panel 2: firearm fraction vs K70
ax = axes[1]
for regime, color in COLORS.items():
    sub = df[df["regime"] == regime][["firearm_fraction","k70_rate"]].dropna()
    ax.scatter(sub["firearm_fraction"], sub["k70_rate"],
               s=5, alpha=0.5, color=color, linewidths=0)
sub2 = df[["firearm_fraction","k70_rate"]].dropna()
r2, p2 = stats.spearmanr(sub2["firearm_fraction"], sub2["k70_rate"])
sig2 = "***" if p2 < 0.001 else ("ns" if p2 >= 0.05 else "*")
ax.set_xlabel("Firearm suicide fraction", fontsize=10)
ax.set_ylabel("K70 rate (per 100K)", fontsize=10)
ax.set_title(f"vs. K70  ρ={r2:.3f}{sig2}", fontsize=11, fontweight="bold")
ax.set_facecolor("#f8f8f6")

# Panel 3: firearm fraction vs overdose
ax = axes[2]
for regime, color in COLORS.items():
    sub = df[df["regime"] == regime][["firearm_fraction","overdose_rate"]].dropna()
    ax.scatter(sub["firearm_fraction"], sub["overdose_rate"],
               s=5, alpha=0.5, color=color, linewidths=0)
sub3 = df[["firearm_fraction","overdose_rate"]].dropna()
r3, p3 = stats.spearmanr(sub3["firearm_fraction"], sub3["overdose_rate"])
sig3 = "***" if p3 < 0.001 else ("ns" if p3 >= 0.05 else "*")
ax.set_xlabel("Firearm suicide fraction", fontsize=10)
ax.set_ylabel("Overdose rate (per 100K)", fontsize=10)
ax.set_title(f"vs. Overdose  ρ={r3:.3f}{sig3}", fontsize=11, fontweight="bold")
ax.set_facecolor("#f8f8f6")

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=r.split(":")[0]+":"+r.split(":")[1])
                   for r, c in COLORS.items()]
axes[0].legend(handles=legend_elements, fontsize=8, loc="upper left")

plt.suptitle("Firearm Suicide Fraction (Gun Ownership Proxy) vs. Each Death Axis\n"
             "(dot color = spatial regime)", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("figures/fig_gun_proxy.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("\n  Saved fig_gun_proxy.png")
print("Done.")
